import contextlib
import torch
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict
import copy

# 引入 PEFT
from peft import LoraConfig, get_peft_model
# 【修改 1】：引入原版 BlipVisionModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BlipVisionModel

# 保留你原有的 ConvNeXt3D 导入
from model.conv_backbone import convnext_3d_tiny


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode does not change anymore."""
    return self

# ==========================================================
# 原生 PyTorch 文本引导感知重采样器 (完美替代 Q-Former)
# ==========================================================
class NativeTextConditionedQFormer(nn.Module):
    def __init__(self, vision_dim=768, text_dim=3584, embed_dim=768, out_dim=3584, num_queries=32, num_layers=4):
        super().__init__()
        self.num_queries = num_queries
        
        # 1. 降维投影
        self.vision_proj = nn.Linear(vision_dim, embed_dim)
        self.text_proj = nn.Linear(text_dim, embed_dim)

        # 2. 可学习的 Query Tokens
        self.query_tokens = nn.Parameter(torch.randn(1, num_queries, embed_dim) * 0.02)

        # 3. 核心交互网络：使用 PyTorch 原生 Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=8,
            dim_feedforward=embed_dim * 4,
            batch_first=True,
            norm_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers) 

        # 4. 输出投影
        self.out_proj = nn.Linear(embed_dim, out_dim)

    def forward(self, vision_feats, text_feats, text_mask=None):
        B = vision_feats.size(0)

        memory = self.vision_proj(vision_feats) 
        t_embeds = self.text_proj(text_feats)   

        queries = self.query_tokens.expand(B, -1, -1)
        tgt = torch.cat([queries, t_embeds], dim=1) 

        if text_mask is not None:
            query_mask = torch.zeros(B, self.num_queries, dtype=torch.bool, device=tgt.device)
            t_pad_mask = ~(text_mask.bool()) 
            tgt_key_padding_mask = torch.cat([query_mask, t_pad_mask], dim=1)
        else:
            tgt_key_padding_mask = None

        out = self.transformer(
            tgt=tgt,
            memory=memory,
            tgt_key_padding_mask=tgt_key_padding_mask
        )

        q_out = out[:, :self.num_queries, :] 
        return self.out_proj(q_out)


class CustomSlowFast(nn.Module):
    def __init__(self, T_out=8, local_weight_path=None):
        super().__init__()
        
        pytorchvideo_dir = '/data/TeamMember/lm/sy/project/models/pytorchvideo-main' 
        base = torch.hub.load(pytorchvideo_dir, 'slowfast_r50', source='local', pretrained=False)
        
        if local_weight_path is not None:
            state_dict = torch.load(local_weight_path, map_location='cpu')
            if 'model_state' in state_dict:
                state_dict = state_dict['model_state']
            base.load_state_dict(state_dict)
            print("Successfully loaded local SlowFast weights.")
            
        self.stem = base.blocks[0]
        self.stage1 = base.blocks[1]
        self.stage2 = base.blocks[2]
        self.stage3 = base.blocks[3]
        self.stage4 = base.blocks[4]
        
        self.pool = nn.AdaptiveAvgPool3d((T_out, 1, 1))

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        
        slow_pool = self.pool(x[0]) 
        fast_pool = self.pool(x[1]) 
        
        return torch.cat([slow_pool, fast_pool], dim=1)


class T2VQA(nn.Module):
    def __init__(self, args):
        super().__init__()
    
        # ---------- 基础配置 ----------
        self.T = args.get('clip_len', 8)  
        llm_model = args['llm_model']

        # ==========================================================
        # 1. 语言模型（LLM）: Qwen2.5-7B
        # ==========================================================
        self.llm_tokenizer = AutoTokenizer.from_pretrained(llm_model, trust_remote_code=True)
        if self.llm_tokenizer.pad_token is None:
            self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token

        self.llm_model = AutoModelForCausalLM.from_pretrained(
            llm_model, 
            torch_dtype=torch.bfloat16, 
            trust_remote_code=True
        )
        self.llm_model.resize_token_embeddings(len(self.llm_tokenizer))

        for name, param in self.llm_model.named_parameters():
            param.requires_grad = False
            
        # 【新增】：配置文件控制 LLM LoRA
        use_llm_lora = args.get('use_llm_lora', True)
        if use_llm_lora:
            peft_config = LoraConfig(
                task_type="CAUSAL_LM",
                inference_mode=False,
                r=16,          
                lora_alpha=32,
                lora_dropout=0.05,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"] 
            )
            self.llm_model = get_peft_model(self.llm_model, peft_config)
            print("=> LLM LoRA is ENABLED.")
            self.llm_model.print_trainable_parameters() 
        else:
            print("=> LLM LoRA is DISABLED.")

        target_words = [" excellent", " good", " fair", " poor", " bad"]
        word_ids = []
        for word in target_words:
            tokens = self.llm_tokenizer(word, add_special_tokens=False)['input_ids']
            word_ids.append(tokens[0])
            
        self.excellent_idx, self.good_idx, self.fair_idx, self.poor_idx, self.bad_idx = word_ids
        self.weights = torch.Tensor([[1], [2], [3], [4], [5]])

        hidden_size = self.llm_model.config.hidden_size

        # ==========================================================
        # 2. 空间语义分支 (原生 BLIP 编码器 + 配置文件控制的 LoRA)
        # ==========================================================
        blip_path = args.get('blip_weights', "Salesforce/blip-image-captioning-base") 
        # 只加载 BLIP 的视觉部分
        self.blip_vision = BlipVisionModel.from_pretrained(blip_path, torch_dtype=torch.bfloat16, local_files_only=True)

        # 初始冻结所有视觉参数
        for param in self.blip_vision.parameters():
            param.requires_grad = False 

        # 【核心修改】：配置文件控制 BLIP LoRA
        use_blip_lora = args.get('use_blip_lora', True)
        if use_blip_lora:
            peft_config_blip = LoraConfig(
                r=16, lora_alpha=32, lora_dropout=0.05,
                # 适配 BLIP (ViT) 架构的关键层
                target_modules=["qkv", "query", "key", "value", "dense", "projection"] 
            )
            self.blip_vision = get_peft_model(self.blip_vision, peft_config_blip)
            print("=> BLIP Vision LoRA is ENABLED.")
            self.blip_vision.print_trainable_parameters()
        else:
            print("=> BLIP Vision LoRA is DISABLED.")
            
        # 实例化原生 Q-Former
        self.qformer = NativeTextConditionedQFormer(
            vision_dim=self.blip_vision.config.hidden_size, # BLIP 通常为 768
            text_dim=hidden_size, 
            embed_dim=768, 
            out_dim=hidden_size,  
            num_queries=32,       
            num_layers=4          
        ).to(torch.bfloat16)

        # ==========================================================
        # 3. 运动与技术质量分支 (SlowFast-R50)
        # ==========================================================
        self.slowfast = CustomSlowFast(
            T_out=self.T, 
            local_weight_path='/data/TeamMember/lm/sy/project/models/models/SLOWFAST_8x8_R50.pyth' 
        )
        slowfast_dim = 2304 
        self.motion_proj = nn.Linear(slowfast_dim, hidden_size)

        # ==========================================================
        # 4. 美学质量分支 (ConvNeXt3D)
        # ==========================================================
        self.aesthetic_conv3d = convnext_3d_tiny(
            pretrained=args.get("conv_pretrained", False),
            in_22k=args.get("conv_in_22k", False),
            checkpoint=args.get("aesthetic_weights", None), 
        )
        convnext_dim = 768
        self.aesthetic_pool = nn.AdaptiveAvgPool3d((self.T, 1, 1))
        self.aesthetic_proj = nn.Linear(convnext_dim, hidden_size)

    def device(self):
        return list(self.parameters())[0].device

    def maybe_autocast(self, dtype=torch.bfloat16):
        enable_autocast = self.device() != torch.device("cpu")
        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()

    def prepare_slowfast_input(self, video):
        alpha = 4
        fast_path = video
        slow_path = video[:, :, ::alpha, :, :]
        return [slow_path, fast_path]

    def forward(self, data, caption, prompt):
        video = data['video']
        B, C, T, H, W = video.shape
        device = video.device

        video_aesthetic = data.get('video_aesthetic', video)

        # ==========================================================
        # 1. 空间语义分支 (原生 BLIP Vision)
        # ==========================================================
        frames = video.transpose(1, 2).reshape(B * T, C, H, W)
        frames = frames.to(self.blip_vision.dtype) 
        
        # A. 提取视觉特征（如果开启 LoRA，这里需要计算梯度，故移除 no_grad）
        vision_outputs = self.blip_vision(pixel_values=frames)
        image_embeds = vision_outputs.last_hidden_state # [B*T, num_patches, 768]
            
        # B. 准备文本条件
        expanded_captions = [cap for cap in caption for _ in range(T)]
        text_inputs = self.llm_tokenizer(
            expanded_captions, 
            padding=True, 
            truncation=True,
            max_length=40,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            text_feats = self.llm_model.get_input_embeddings()(text_inputs.input_ids)

        # C. 多模态交互
        qformer_out = self.qformer(
            vision_feats=image_embeds.to(text_feats.dtype),
            text_feats=text_feats,
            text_mask=text_inputs.attention_mask
        ) 
        
        # D. 提取输出并浓缩语义
        frame_semantic = qformer_out.mean(dim=1) 
        semantic_embeds = frame_semantic.view(B, T, -1) 

        # ==========================================================
        # 2. 运动分支 (SlowFast)
        # ==========================================================
        sf_input = self.prepare_slowfast_input(video)
        pooled_motion = self.slowfast(sf_input) 
        
        motion_tokens = pooled_motion.flatten(2).transpose(1, 2) 
        motion_embeds = self.motion_proj(motion_tokens) 

        # ==========================================================
        # 3. 美学分支 (ConvNeXt)
        # ==========================================================
        aes_features = self.aesthetic_conv3d(video_aesthetic) 
        pooled_aes = self.aesthetic_pool(aes_features) 
        
        aesthetic_tokens = pooled_aes.flatten(2).transpose(1, 2) 
        aesthetic_embeds = self.aesthetic_proj(aesthetic_tokens) 

        # ==========================================================
        # 4. Token 拼接与 Qwen 输入对齐
        # ==========================================================
        multimodal_embeds = torch.cat([motion_embeds, aesthetic_embeds, semantic_embeds], dim=1) 
        atts_multimodal = torch.ones(multimodal_embeds.size()[:-1], dtype=torch.long).to(device)

        full_prompt = [f"Context: {cap}. {prompt}" for cap in caption] if isinstance(caption, list) else [f"Context: {caption}. {prompt}"] * B
        
        llm_tokens = self.llm_tokenizer(
            full_prompt,
            padding="longest",
            return_tensors="pt"
        ).to(device)

        # ==========================================================
        # 5. 前向传播与 Logits 打分提取
        # ==========================================================
        with self.maybe_autocast():
            text_embeds = self.llm_model.get_input_embeddings()(llm_tokens.input_ids)
            
            inputs_embeds = torch.cat([multimodal_embeds.to(text_embeds.dtype), text_embeds], dim=1)
            attention_mask = torch.cat([atts_multimodal, llm_tokens.attention_mask], dim=1)

            outputs = self.llm_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
            )
            
        output_logits = outputs.logits[:, -1]
        
        lexcellent = output_logits[:, self.excellent_idx]
        lgood = output_logits[:, self.good_idx]
        lfair = output_logits[:, self.fair_idx]
        lpoor = output_logits[:, self.poor_idx]
        lbad = output_logits[:, self.bad_idx]
        
        q_pred = (torch.stack([lexcellent, lgood, lfair, lpoor, lbad]) / 100).softmax(0)
        weights = self.weights.expand(-1, q_pred.shape[1]).to(device)
        q_pred = torch.mul(q_pred, weights)
        q_pred = torch.sum(q_pred, dim=0)

        return q_pred


if __name__=="__main__":
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    
    mock_args = {
        'clip_len': 8,
        'llm_model': 'Qwen/Qwen2.5-7B-Instruct', 
        'blip_weights': 'Salesforce/blip-image-captioning-base',
        'use_blip_lora': True,
        'use_llm_lora': False,
    }
    
    model = T2VQA(args=mock_args).to(device)
    model.eval()
    
    caption = ['A random caption about a dog'] * 2
    prompt = 'Carefully watch the video and evaluate its quality. The overall quality of this video is'
    video = torch.randn(2, 3, 8, 224, 224).to(device)
    data = {'video': video}

    output = model(data, caption, prompt)
    print("Predicted Scores:", output)