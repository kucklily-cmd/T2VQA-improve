import contextlib
import torch
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict
import copy
# 1. 在文件开头引入 PEFT
from peft import LoraConfig, get_peft_model
# 引入 Qwen 相关的 Auto 类和 CLIP
from transformers import AutoModelForCausalLM, AutoTokenizer, CLIPVisionModel
import torchvision.models.video as video_models
from transformers import Blip2Model, Blip2Processor # 【新增】BLIP-2 相关库
# 保留你原有的 ConvNeXt3D 导入
from model.conv_backbone import convnext_3d_tiny
# 1. 在文件开头增加以下导入
from transformers import AutoModelForCausalLM, AutoTokenizer, Blip2Model, AutoTokenizer as Blip2Tokenizer

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode does not change anymore."""
    return self
class TextConditionedQFormer(nn.Module): 
    def __init__(self, clip_dim=1024, text_dim=3584, embed_dim=768, out_dim=3584, num_queries=8):
        super().__init__()
        self.num_queries = num_queries
        
        # 1. 降维投影：在高维 (3584) 计算 Attention 显存开销极大，先降至 embed_dim
        self.video_proj = nn.Linear(clip_dim, embed_dim)
        self.text_proj = nn.Linear(text_dim, embed_dim)

        # 2. 可学习的 Query Tokens (代表模型自带的视觉信息提取模板)
        self.query_tokens = nn.Parameter(torch.zeros(1, num_queries, embed_dim))
        nn.init.normal_(self.query_tokens, std=0.02)

        # 3. 核心交互网络：使用 PyTorch 原生 Transformer Decoder
        # Query 和 Text 作为 Target (带问题)，Video 作为 Memory (找答案)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=8,
            dim_feedforward=embed_dim * 4,
            batch_first=True,
            norm_first=True,  # 推荐使用 norm_first 让训练更稳定
            activation='gelu'
        )
        # 3 层 Decoder 足够完成深度的图文语义对齐
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=3) 

        # 4. 输出投影：升维回 LLM 所需的特征维度
        self.out_proj = nn.Linear(embed_dim, out_dim)

    def forward(self, video_feats, text_feats, text_mask=None):
        B = video_feats.size(0)

        # [B, T, embed_dim]
        v_embeds = self.video_proj(video_feats) 
        # [B, L, embed_dim]
        t_embeds = self.text_proj(text_feats)   

        # 扩展 Query Tokens 匹配 batch size
        queries = self.query_tokens.expand(B, -1, -1) # [B, num_queries, embed_dim]

        # 关键操作：将 Query 和 Text 特征在序列维度拼接作为联合 Target
        # 这样在 Self-Attention 中 Query 能“看见”文本，再带着文本信息去 Cross-Attend 视频
        tgt = torch.cat([queries, t_embeds], dim=1) # [B, num_queries + L, embed_dim]

        # 处理文本的 padding mask
        if text_mask is not None:
            # Query 部分全 0 (不 mask)
            query_mask = torch.zeros(B, self.num_queries, dtype=torch.bool, device=tgt.device)
            # text_mask 是 1 为有效，Transformer 中 True 代表忽略，所以取反
            t_pad_mask = ~(text_mask.bool()) 
            tgt_key_padding_mask = torch.cat([query_mask, t_pad_mask], dim=1)
        else:
            tgt_key_padding_mask = None

        # 穿越 Transformer 提取特征
        out = self.transformer(
            tgt=tgt,
            memory=v_embeds,
            tgt_key_padding_mask=tgt_key_padding_mask
        )

        # 剥离出 Query 部分的输出作为最终的视觉特征 (抛弃附加的文本部分)
        q_out = out[:, :self.num_queries, :] # [B, num_queries, embed_dim]

        # 升维并返回
        return self.out_proj(q_out)
class CustomSlowFast(nn.Module):
    def __init__(self, T_out=8, local_weight_path=None):
        super().__init__()
        
        # 1. 离线加载 base 模型
        pytorchvideo_dir = '/data/TeamMember/lm/sy/project/models/pytorchvideo-main' 
        base = torch.hub.load(pytorchvideo_dir, 'slowfast_r50', source='local', pretrained=False)
        
        if local_weight_path is not None:
            state_dict = torch.load(local_weight_path, map_location='cpu')
            if 'model_state' in state_dict:
                state_dict = state_dict['model_state']
            base.load_state_dict(state_dict)
            print("Successfully loaded local SlowFast weights.")
            
        # 2. 终极物理隔离：手动且精准地只提取前 5 个特征模块
        # 彻底抛弃 base.blocks[5]（也就是那个会导致崩溃的固定池化分类头）
        self.stem = base.blocks[0]
        self.stage1 = base.blocks[1]
        self.stage2 = base.blocks[2]
        self.stage3 = base.blocks[3]
        self.stage4 = base.blocks[4]
        
        # 3. Token 平衡：强制将时空特征池化为 T_out 个时序 Token
        self.pool = nn.AdaptiveAvgPool3d((T_out, 1, 1))

    def forward(self, x):
        # 严格按顺序手动前向传播，彻底斩断与分类头的任何联系
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        
        # 此时 x 完美避开了分类头，输出最纯净的特征: [slow_features, fast_features]
        # x[0] 是慢分支 (T=2)，x[1] 是快分支 (T=8)
        
        # 自适应池化层会自动把 T=2 扩展对齐到 T=8
        slow_pool = self.pool(x[0]) 
        fast_pool = self.pool(x[1]) 
        
        # 拼接快慢分支特征，总维度恢复到 2304
        return torch.cat([slow_pool, fast_pool], dim=1)

class T2VQA(nn.Module):
    def __init__(self, args):
        super().__init__()
    
        # ---------- 基础配置 ----------
        self.T = args.get('clip_len', 8)  # 视频帧数/Token 数量基准
        llm_model = args['llm_model']

        # ==========================================================
        # 1. 语言模型（LLM）: Qwen2.5-7B
        # ==========================================================
        self.llm_tokenizer = AutoTokenizer.from_pretrained(llm_model, trust_remote_code=True)
        if self.llm_tokenizer.pad_token is None:
            self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token

        # 推荐使用 bfloat16 加载 Qwen，如果显卡支持，可以开启 flash_attention_2
        self.llm_model = AutoModelForCausalLM.from_pretrained(
            llm_model, 
            torch_dtype=torch.bfloat16, 
            trust_remote_code=True
        )
        self.llm_model.resize_token_embeddings(len(self.llm_tokenizer))

        # 【修改点 1】：冻结原 LLM 参数，但【不要】重写 disabled_train，因为 LoRA 层需要进入 train 模式
        for name, param in self.llm_model.named_parameters():
            param.requires_grad = False
            
        # 【新增点 1】：注入 LoRA 适配器
        peft_config = LoraConfig(
            task_type="CAUSAL_LM",
            inference_mode=False,
            r=16,          # LoRA 的秩，推荐 16 或 32
            lora_alpha=32,
            lora_dropout=0.05,
            # 针对 Qwen 等 LLaMA 架构的推荐 Target Modules
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"] 
        )
        self.llm_model = get_peft_model(self.llm_model, peft_config)
        self.llm_model.print_trainable_parameters() # 打印一下 LoRA 的参数量确认注入成功    
        # 提取 Qwen 的 5 个质量评价词 Token ID（必须带前导空格）
        target_words = [" excellent", " good", " fair", " poor", " bad"]
        word_ids = []
        for word in target_words:
            tokens = self.llm_tokenizer(word, add_special_tokens=False)['input_ids']
            word_ids.append(tokens[0])
            
        self.excellent_idx, self.good_idx, self.fair_idx, self.poor_idx, self.bad_idx = word_ids
        self.weights = torch.Tensor([[1], [2], [3], [4], [5]])

        # LLM 的隐藏层维度 (Qwen2.5-7B 通常为 3584)
        hidden_size = self.llm_model.config.hidden_size

        # ==========================================================
        # 2. 空间语义分支 (升级为 BLIP-2 + LoRA)
        # ==========================================================
        # 推荐使用 blip2-opt-2.7b 作为基座（我们不用它的OPT模型，只借用它的视觉和Q-Former权重）
        blip2_path = args.get('blip2_model', "Salesforce/blip2-opt-2.7b") 
        self.blip2_tokenizer = Blip2Tokenizer.from_pretrained(blip2_path)
        
        # 加载 Blip2Model (不含语言生成头)
        self.blip2 = Blip2Model.from_pretrained(blip2_path, torch_dtype=torch.bfloat16)

        # 冻结视觉编码器 (EVA-CLIP)
        for param in self.blip2.vision_model.parameters():
            param.requires_grad = False 
            
        # 【新增】：对 Q-Former 应用 LoRA
        # Q-Former 内部是类 BERT 结构，target_modules 使用 attention 的关键层
        peft_config_qformer = LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.05,
            target_modules=["query", "key", "value", "dense"] 
        )
        self.blip2.qformer = get_peft_model(self.blip2.qformer, peft_config_qformer)
        self.blip2.qformer.print_trainable_parameters() # 确认 Q-Former LoRA 注入成功

        # 投影层：将 Q-Former 输出 (通常为 768 维) 映射到 Qwen (3584 维)
        self.semantic_proj = nn.Linear(self.blip2.config.qformer_config.hidden_size, hidden_size)

        # ==========================================================
        # 3. 运动与技术质量分支 (SlowFast-R50)
        # ==========================================================
        # 传入你本地下载好的 .pyth 文件绝对路径
        # 如果你已经搞定了服务器联网，直接让它留空 (local_weight_path=None) 并在上面设 pretrained=True 即可
        self.slowfast = CustomSlowFast(
            T_out=self.T, 
            local_weight_path='/data/TeamMember/lm/sy/project/models/models/SLOWFAST_8x8_R50.pyth' 
        )
        slowfast_dim = 2304 
        
        # 视显存情况，可以微调 SlowFast，或者在此处冻结
        # for param in self.slowfast.parameters(): param.requires_grad = False
        
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
        # Token 平衡：强制将时空特征池化为 T 个时序 Token
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
        # PyTorchVideo 官方 slowfast_r50 要求的快慢分支比例 (alpha) 严格为 4
        alpha = 4
        
        # Fast pathway: 包含所有的 8 帧
        fast_path = video
        
        # Slow pathway: 严格按照固定步长 alpha 进行降采样
        # 这样 8 帧的视频会精确抽出 8 / 4 = 2 帧，完美对齐
        slow_path = video[:, :, ::alpha, :, :]
        
        return [slow_path, fast_path]
    def forward(self, data, caption, prompt):
        # video 形状应当是 [B, 3, T, 224, 224]
        video = data['video']
        B, C, T, H, W = video.shape
        device = video.device

        # 如果 DataLoader 提供了独立美学数据则用，否则复用基础数据
        video_aesthetic = data.get('video_aesthetic', video)

        # ==========================================================
        # 1. 空间语义分支 (BLIP-2 Vision + Q-Former)
        # ==========================================================
        frames = video.transpose(1, 2).reshape(B * T, C, H, W)
        
        # 确保数据类型对齐 (输入转为与 blip2 相同的 bfloat16)
        frames = frames.to(self.blip2.dtype) 
        
        # A. 提取视觉特征 (无梯度)
        with torch.no_grad():
            vision_outputs = self.blip2.vision_model(pixel_values=frames)
            image_embeds = vision_outputs[0] # [B*T, num_patches, hidden_size]
            
        # B. 准备 Q-Former 的文本条件
        # 因为 frames 维度是 B*T，我们需要把 batch 为 B 的 caption 复制 T 遍对齐
        expanded_captions = [cap for cap in caption for _ in range(T)]
        text_inputs = self.blip2_tokenizer(
            expanded_captions, 
            padding=True, 
            return_tensors="pt"
        ).to(device)

        # C. Q-Former 多模态交互 (带梯度更新 LoRA)
        # 1. 提取 BLIP-2 内置的 Query Tokens
        query_tokens = self.blip2.query_tokens.expand(B * T, -1, -1)
        
        # 2. 【新增】：手动拼接 Attention Mask
        # Query Tokens 永远是有效的，所以给它们生成全是 1 的 mask
        query_mask = torch.ones(
            B * T, query_tokens.shape[1], 
            dtype=text_inputs.attention_mask.dtype, 
            device=device
        )
        # 将 Query Mask (32) 和 Text Mask (L) 在序列维度拼接 -> 长度变为 32 + L
        qformer_attention_mask = torch.cat([query_mask, text_inputs.attention_mask], dim=1)

        # 3. 送入 Q-Former (传入拼接后的完整 Mask)
        qformer_outputs = self.blip2.qformer(
            query_embeds=query_tokens,
            input_ids=text_inputs.input_ids,
            attention_mask=qformer_attention_mask,  # <--- 使用补齐后的完整 Mask
            encoder_hidden_states=image_embeds,
        )
        
        # qformer_outputs.last_hidden_state 形状是 [B*T, num_queries(通常32), 768]
        # 对 32 个 Query 求均值，代表每一帧浓缩后的语义特征
        frame_semantic = qformer_outputs.last_hidden_state.mean(dim=1) # [B*T, 768]

        # 恢复时序维度并进行通道映射
       # 恢复时序维度，并动态对齐到投影层的数据类型 (Float32)
        semantic_tokens = frame_semantic.view(B, T, -1).to(self.semantic_proj.weight.dtype)
        semantic_embeds = self.semantic_proj(semantic_tokens) # [B, T, 3584]

        # ==========================================================
        # 2. 运动分支 (SlowFast) - 提取 T 个 Token
        # ==========================================================
        sf_input = self.prepare_slowfast_input(video)
        # [B, 2304, T, 1, 1]
        pooled_motion = self.slowfast(sf_input) 
        # [B, T, 2304]
        # 运动分支 (把 flatten(3) 改为 flatten(2))
        motion_tokens = pooled_motion.flatten(2).transpose(1, 2) 
        motion_embeds = self.motion_proj(motion_tokens) # 完美形状: [B, T, 3584

        # ==========================================================
        # 3. 美学分支 (ConvNeXt) - 提取 T 个 Token
        # ==========================================================
        # [B, 768, T', H', W']
        aes_features = self.aesthetic_conv3d(video_aesthetic) 
        # [B, 768, T, 1, 1]
        pooled_aes = self.aesthetic_pool(aes_features)
        # [B, T, 768]
        # 美学分支 (把 flatten(3) 改为 flatten(2))
        aesthetic_tokens = pooled_aes.flatten(2).transpose(1, 2) 
        aesthetic_embeds = self.aesthetic_proj(aesthetic_tokens) # 完美形状: [B, T, 3584]

        # ==========================================================
        # 4. Token 拼接与 Qwen 输入对齐
        # ==========================================================
        # 序列维度拼接，实现 3*T 的 Token 绝对平衡
        # 形状变为 [B, 3*T, 3584]
        multimodal_embeds = torch.cat([motion_embeds, aesthetic_embeds, semantic_embeds], dim=1) 
        atts_multimodal = torch.ones(multimodal_embeds.size()[:-1], dtype=torch.long).to(device)

        # 我们可以将传入的 caption 作为背景信息融合到 Prompt 中，丰富 Qwen 的理解上下文
        # 如果 train.py 里没有这么写，在此处合并最保险
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
            # 提取纯文本的 embedding
            text_embeds = self.llm_model.get_input_embeddings()(llm_tokens.input_ids)
            
            # 将多模态 Token 放在文本 Token 前面，并且必须统一 dtype(bfloat16)
            inputs_embeds = torch.cat([multimodal_embeds.to(text_embeds.dtype), text_embeds], dim=1)
            attention_mask = torch.cat([atts_multimodal, llm_tokens.attention_mask], dim=1)

            outputs = self.llm_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
            )
            
        # 提取整个序列预测下一个词的 Logits
        output_logits = outputs.logits[:, -1]
        
        # 抓取 5 个评价词的概率
        lexcellent = output_logits[:, self.excellent_idx]
        lgood = output_logits[:, self.good_idx]
        lfair = output_logits[:, self.fair_idx]
        lpoor = output_logits[:, self.poor_idx]
        lbad = output_logits[:, self.bad_idx]
        
        # 归一化并加权求和得出最终回归分数 (Softmax + Expected Value)
        q_pred = (torch.stack([lexcellent, lgood, lfair, lpoor, lbad]) / 100).softmax(0)
        weights = self.weights.expand(-1, q_pred.shape[1]).to(device)
        q_pred = torch.mul(q_pred, weights)
        q_pred = torch.sum(q_pred, dim=0)

        return q_pred


if __name__=="__main__":
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    
    # 模拟外部传入参数
    mock_args = {
        'clip_len': 8,
        'llm_model': 'Qwen/Qwen2.5-7B-Instruct', # 这里填写你的 Qwen 本地路径
        'clip_weights': 'openai/clip-vit-large-patch14',
    }
    
    model = T2VQA(args=mock_args).to(device)
    model.eval()
    
    # 模拟数据输入
    caption = ['A random caption about a dog'] * 2
    prompt = 'Carefully watch the video and evaluate its quality. The overall quality of this video is'
    video = torch.randn(2, 3, 8, 224, 224).to(device)
    data = {'video': video}

    with torch.no_grad():
        output = model(data, caption, prompt)
    print("Predicted Scores:", output)