import torch
from torch import nn
import contextlib
from collections import OrderedDict

# 引入 HuggingFace 新版核心库
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    Blip2Model, 
    Blip2Processor,
    BertModel,           
    BertTokenizer        
)
# 引入 LoRA 微调库
from peft import LoraConfig, get_peft_model, TaskType

# 导入你原有的视觉主干网络
from model.swin import swin_3d_tiny
from model.conv_backbone import convnext_3d_tiny

# ==========================================
# 基础池化与融合模块
# ==========================================
class CrossAttentionPooling(nn.Module):
    def __init__(self, text_dim, visual_dim, embed_dim, num_heads=8):
        super().__init__()
        self.q_proj = nn.Linear(text_dim, embed_dim)
        self.k_proj = nn.Linear(visual_dim, embed_dim)
        self.v_proj = nn.Linear(visual_dim, embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, text_feat, visual_tokens):
        q = self.q_proj(text_feat).unsqueeze(1) 
        k = self.k_proj(visual_tokens)
        v = self.v_proj(visual_tokens)
        attn_out, _ = self.attn(q, k, v)
        out = self.norm1(q + attn_out)
        ffn_out = self.ffn(out)
        out = self.norm2(out + ffn_out)
        return out.squeeze(1)

class GateMixer(nn.Module):
    def __init__(self, v_in_dim, c_in_dim, text_dim, d, token_len=32, prefix_len=8, out_dim=None):
        super().__init__()
        self.token_len = token_len
        self.prefix_len = prefix_len
        self.w1_v = nn.Linear(v_in_dim, d)
        self.w1_c = nn.Linear(c_in_dim, d)
        self.w_g = nn.Linear(2 * d + text_dim, d) 
        if prefix_len > 0:
            self.h_p = nn.Parameter(torch.zeros(1, prefix_len, d))
            nn.init.normal_(self.h_p, mean=0.0, std=0.02)
        else:
            self.h_p = None
        self.w2 = nn.Linear(d, out_dim or d)

    def forward(self, v_v, v_c, text_feat):
        h_v = self.w1_v(v_v).unsqueeze(1).expand(-1, self.token_len, -1)
        h_c = self.w1_c(v_c).unsqueeze(1).expand(-1, self.token_len, -1)
        text_feat_exp = text_feat.unsqueeze(1).expand(-1, self.token_len, -1)
        
        alpha_v = torch.sigmoid(self.w_g(torch.cat([h_v, h_c, text_feat_exp], dim=-1)))
        h = (1 - alpha_v) * h_v + alpha_v * h_c
        
        if self.h_p is not None:
            h = torch.cat([self.h_p.expand(h.size(0), -1, -1), h], dim=1)
        return self.w2(h)

# ==========================================
# 重构核心模型 (LLaMA-3 + BLIP-2 + BERT + 双路 LoRA)
# ==========================================
class T2VQA_Llama3_Blip2(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        embed_dim = args.get('embed_dim', 256)
        llm_model_path = args['llm_model'] 
        blip2_model_path = args.get('blip2_model', "Salesforce/blip2-opt-2.7b")
        bert_model_path = args.get('bert_weights', 'bert-base-uncased')
        
        # -----------------------------------------------------------
        # 1. 视觉编码器 (BLIP-2)
        # -----------------------------------------------------------
        self.blip2_processor = Blip2Processor.from_pretrained(blip2_model_path)
        self.blip2 = Blip2Model.from_pretrained(blip2_model_path, torch_dtype=torch.float16)
        
        # --- 动态控制视觉编码器 (ViT) 的 LoRA 微调 ---
        self.use_vision_lora = args.get('use_vision_lora', False)
        
        if self.use_vision_lora:
            print("=> 启用 LoRA 进行 视觉编码器 (ViT) 微调...")
            vision_lora_r = args.get('vision_lora_r', 16)
            vision_lora_alpha = args.get('vision_lora_alpha', 32)
            vision_lora_dropout = args.get('vision_lora_dropout', 0.05)
            
            # 使用 PEFT 包装 BLIP-2 的 Vision Model 部分
            vision_peft_config = LoraConfig(
                r=vision_lora_r,
                lora_alpha=vision_lora_alpha,
                lora_dropout=vision_lora_dropout,
                # 适配 BLIP-2 ViT 中典型线性层的名称
                target_modules=["qkv", "projection", "dense"] 
            )
            self.blip2.vision_model = get_peft_model(self.blip2.vision_model, vision_peft_config)
            self.blip2.vision_model.print_trainable_parameters()
        else:
            print("=> 禁用视觉 LoRA。BLIP-2 的视觉主干网络将被完全冻结。")
            for param in self.blip2.vision_model.parameters():
                param.requires_grad = False
                
        # Q-Former 桥接层参数量适中，通常全参微调以学习新的图文对齐关系
        for param in self.blip2.qformer.parameters():
            param.requires_grad = True

        blip2_hidden_size = self.blip2.config.qformer_config.hidden_size # 通常为 768

        # -----------------------------------------------------------
        # 2. 纯文本编码器 (BERT - 专用于提取 GateMixer 的锚点)
        # -----------------------------------------------------------
        self.bert_tokenizer = BertTokenizer.from_pretrained(bert_model_path)
        self.pure_text_encoder = BertModel.from_pretrained(bert_model_path)
        
        # 彻底冻结 BERT，它只提供纯文本语义，不参与训练更新，极省显存
        for param in self.pure_text_encoder.parameters():
            param.requires_grad = False
            
        text_hidden_size = self.pure_text_encoder.config.hidden_size # 通常为 768

        # -----------------------------------------------------------
        # 3. 技术质量分支 (Swin3D + Conv3D)
        # -----------------------------------------------------------
        self.swin3d = swin_3d_tiny()
        if args.get('swin_weights'):
            state_dict = torch.load(args['swin_weights'], map_location='cpu')['state_dict']
            i_state_dict = OrderedDict({k.replace("backbone.", ""): v for k, v in state_dict.items() if "head" not in k})
            self.swin3d.load_state_dict(i_state_dict, strict=False)
            
        self.conv3d = convnext_3d_tiny(checkpoint=args.get("conv_weights", None))

        # 交叉注意力池化器 (输入维度改为 BERT 的维度)
        self.swin_attn_pool = CrossAttentionPooling(text_dim=text_hidden_size, visual_dim=768, embed_dim=embed_dim)
        self.conv_attn_pool = CrossAttentionPooling(text_dim=text_hidden_size, visual_dim=768, embed_dim=embed_dim)

        self.gate_mixer = GateMixer(
            v_in_dim=embed_dim, c_in_dim=embed_dim, text_dim=text_hidden_size, d=embed_dim,
            token_len=32, prefix_len=8, out_dim=embed_dim
        )

        # -----------------------------------------------------------
        # 4. 语言模型 (LLaMA-3) 与 动态 LoRA 微调
        # -----------------------------------------------------------
        self.llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_path, padding_side="left")
        if self.llm_tokenizer.pad_token is None:
            self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token

        self.llm_model = AutoModelForCausalLM.from_pretrained(
            llm_model_path, 
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # 冻结 LLaMA-3 的所有基础参数
        for param in self.llm_model.parameters():
            param.requires_grad = False
            
        # 读取配置文件中的 LLM LoRA 设置
        self.use_lora = args.get('use_lora', True)
        
        if self.use_lora:
            print("=> 启用 LoRA 进行 大语言模型 (LLM) 微调...")
            lora_r = args.get('lora_r', 16)
            lora_alpha = args.get('lora_alpha', 32)
            lora_dropout = args.get('lora_dropout', 0.05)
            
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM, 
                inference_mode=False, 
                r=lora_r, 
                lora_alpha=lora_alpha, 
                lora_dropout=lora_dropout,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            )
            self.llm_model = get_peft_model(self.llm_model, peft_config)
            self.llm_model.print_trainable_parameters() 
        else:
            print("=> 禁用大语言模型 LoRA。LLM 的所有参数将被冻结。")

        # -----------------------------------------------------------
        # 5. 模态对齐投影层 (Projection Layers)
        # -----------------------------------------------------------
        llm_hidden_size = self.llm_model.config.hidden_size
        self.finetune_semantic_proj = nn.Linear(blip2_hidden_size, llm_hidden_size)
        self.finetune_fidelity_proj = nn.Linear(embed_dim, llm_hidden_size)

        # -----------------------------------------------------------
        # 6. 质量预测分数 Logits 提取
        # -----------------------------------------------------------
        target_words = [" excellent", " good", " fair", " poor", " bad"]
        self.quality_ids = []
        for word in target_words:
            tokens = self.llm_tokenizer(word, add_special_tokens=False).input_ids
            self.quality_ids.append(tokens[-1]) 
        
        self.weights = torch.Tensor([[1], [2], [3], [4], [5]])

    @property
    def device(self):
        return next(self.parameters()).device

    def maybe_autocast(self, dtype=torch.float16):
        if self.device.type != "cpu":
            return torch.amp.autocast('cuda', dtype=dtype)
        else:
            return contextlib.nullcontext()

    def forward(self, data, caption, prompt):
        video = data['video'] # [B, C, T, H, W]
        B, C, T, H, W = video.shape
        device = video.device

        # ==========================================
        # Step 1: 语义与视觉特征提取
        # ==========================================
        
        # 1.1 使用 BERT 提取纯文本的全局特征 (Semantic Anchor)
        text_inputs = self.bert_tokenizer(
            caption, padding='max_length', truncation=True, max_length=35, return_tensors="pt"
        ).to(device)
        
        with torch.no_grad(): 
            text_outputs = self.pure_text_encoder(
                input_ids=text_inputs.input_ids,
                attention_mask=text_inputs.attention_mask,
                return_dict=True,
            )
        global_text_feat = text_outputs.last_hidden_state[:, 0, :] # [B, text_dim]

        # 1.2 准备 BLIP-2 视频输入，按帧拉平
        video_flat = video.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)

        with self.maybe_autocast():
            qformer_features = self.blip2.get_qformer_features(pixel_values=video_flat)
            
        # Q-Former 输出: [B*T, num_query_tokens(32), blip2_hidden_size(768)]
        frame_semantic_tokens = qformer_features.mean(dim=1) # [B*T, 768]
        frame_semantic_tokens = frame_semantic_tokens.view(B, T, -1) # [B, T, 768]

        # ==========================================
        # Step 2: 技术质量分支提取
        # ==========================================
        with self.maybe_autocast():
            f_swin = self.swin3d(video) 
            f_conv = self.conv3d(video) 
            
        f_swin_flat = f_swin.view(B, f_swin.shape[1], -1).transpose(1, 2) 
        f_conv_flat = f_conv.view(B, f_conv.shape[1], -1).transpose(1, 2)
        
        pooled_swin = self.swin_attn_pool(global_text_feat, f_swin_flat) 
        pooled_conv = self.conv_attn_pool(global_text_feat, f_conv_flat) 

        # GateMixer 融合
        inputs_swin = self.gate_mixer(pooled_swin, pooled_conv, global_text_feat) # [B, len, embed_dim]

        # ==========================================
        # Step 3: LLM 投影与拼接
        # ==========================================
        semantic_embeds = self.finetune_semantic_proj(frame_semantic_tokens) # [B, T, llm_dim]
        fidelity_embeds = self.finetune_fidelity_proj(inputs_swin)           # [B, len, llm_dim]

        visual_inputs_embeds = torch.cat([fidelity_embeds, semantic_embeds], dim=1)
        visual_atts = torch.ones(visual_inputs_embeds.size()[:-1], dtype=torch.long).to(device)

        if isinstance(prompt, str):
            prompt = [prompt] * B 

        prompt_texts = [f"<|start_header_id|>user<|end_header_id|>\n\n{p}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nThe quality of this video is " for p in prompt]
        
        llm_tokens = self.llm_tokenizer(
            prompt_texts, padding="longest", return_tensors="pt", add_special_tokens=False
        ).to(device)

        with self.maybe_autocast():
            if self.use_lora:
                text_inputs_embeds = self.llm_model.get_base_model().get_input_embeddings()(llm_tokens.input_ids)
            else:
                text_inputs_embeds = self.llm_model.get_input_embeddings()(llm_tokens.input_ids)
            
            inputs_embeds = torch.cat([visual_inputs_embeds, text_inputs_embeds], dim=1)
            attention_mask = torch.cat([visual_atts, llm_tokens.attention_mask], dim=1)

            outputs = self.llm_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
            )

        # ==========================================
        # Step 4: 质量分数回归
        # ==========================================
        output_logits = outputs.logits[:, -1, :] 
        
        q_logits = output_logits[:, self.quality_ids] # [B, 5]
        q_pred = (q_logits / 100).softmax(dim=-1)     # [B, 5]
        
        weights_device = self.weights.to(device, dtype=q_pred.dtype)
        score = torch.matmul(q_pred, weights_device).squeeze(-1) # [B]

        return score


# ==========================================
# 本地快速测试模块
# ==========================================
if __name__=="__main__":
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    mock_args = {
        'llm_model': 'LLM-Research/Meta-Llama-3-8B-Instruct', 
        'blip2_model': 'Salesforce/blip2-opt-2.7b',
        'bert_weights': 'bert-base-uncased', 
        'embed_dim': 256,
        
        # LLM LoRA 配置
        'use_lora': True,
        'lora_r': 16,
        
        # --- 视觉编码器 LoRA 新增配置 ---
        'use_vision_lora': True,
        'vision_lora_r': 16,
    }
    print("=> 正在初始化模型 (测试模式)...")
    model = T2VQA_Llama3_Blip2(mock_args).to(device)
    model.eval()
    
    caption = 'A random caption'
    prompt = ['Please assess the quality of this video'] * 2 
    video = torch.randn(2, 3, 8, 224, 224).to(device)

    print("=> 正在执行 Forward 推理测试...")
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            output = model({'video': video}, caption, prompt)
            
    print("=> 测试成功! 模型输出分数:", output)