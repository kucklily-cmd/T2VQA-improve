import contextlib
import torch
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict
import copy

# 引入 Qwen 相关的 Auto 类和 CLIP
from transformers import AutoModelForCausalLM, AutoTokenizer, CLIPVisionModel
import torchvision.models.video as video_models

# 保留你原有的 ConvNeXt3D 导入
from model.conv_backbone import convnext_3d_tiny

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode does not change anymore."""
    return self

class CustomSlowFast(nn.Module):
    def __init__(self, T_out=8, local_weight_path=None):
        super().__init__()
        
        # 正确加载 SlowFast 的方式：使用 PyTorch Hub 从 PyTorchVideo 加载
        # 如果服务器能联网，将 pretrained 改为 True 即可自动下载
        base = torch.hub.load('facebookresearch/pytorchvideo', 'slowfast_r50', pretrained=False)
        
        # 如果服务器断网，需要手动加载你刚才下载的 .pyth 文件
        # 例如：local_weight_path = "/data/models/SLOWFAST_8x8_R50.pyth"
        if local_weight_path is not None:
            state_dict = torch.load(local_weight_path, map_location='cpu')
            # PyTorchVideo 的实际权重被包裹在 'model_state' 键中
            if 'model_state' in state_dict:
                state_dict = state_dict['model_state']
            base.load_state_dict(state_dict)
            print("Successfully loaded local SlowFast weights.")
            
        # 移除最后的分类头 (blocks[6] 通常是 ResNetBasicHead)，保留前面的特征提取层
        self.blocks = nn.ModuleList(list(base.blocks.children())[:-1])
        
        # Token 平衡：强制将时空特征池化为 T_out 个时序 Token
        self.pool = nn.AdaptiveAvgPool3d((T_out, 1, 1))

    def forward(self, x):
        # x 接收一个列表: [slow_pathway, fast_pathway]
        for block in self.blocks:
            x = block(x)
            
        # x[0] 是慢分支特征 (2048维)，x[1] 是快分支特征 (256维)
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

        # 冻结 LLM 所有参数
        for name, param in self.llm_model.named_parameters():
            param.requires_grad = False
        self.llm_model = self.llm_model.eval()
        self.llm_model.train = disabled_train

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
        # 2. 空间语义分支 (CLIP)
        # ==========================================================
        clip_weights = args.get('clip_weights', 'openai/clip-vit-large-patch14')
        self.clip = CLIPVisionModel.from_pretrained(clip_weights)
        clip_dim = self.clip.config.hidden_size # ViT-Large 为 1024
        
        # 冻结 CLIP 参数
        for param in self.clip.parameters():
            param.requires_grad = False 
            
        self.semantic_proj = nn.Linear(clip_dim, hidden_size)

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
        # 构造 SlowFast 需要的快慢帧输入: [B, C, T, H, W] -> [slow_path, fast_path]
        # 假设 fast pathway 包含所有帧 (alpha=4 比例)
        fast_path = video
        # Slow pathway 使用下采样帧，通常采样间隔为 4
        # 为了防止视频过短，使用 ::max(1, T//4) 保证至少能提取帧
        stride = max(1, video.size(2) // 4)
        slow_path = video[:, :, ::stride, :, :]
        return [slow_path, fast_path]

    def forward(self, data, caption, prompt):
        # video 形状应当是 [B, 3, T, 224, 224]
        video = data['video']
        B, C, T, H, W = video.shape
        device = video.device

        # 如果 DataLoader 提供了独立美学数据则用，否则复用基础数据
        video_aesthetic = data.get('video_aesthetic', video)

        # ==========================================================
        # 1. 空间语义分支 (CLIP) - 提取 T 个 Token
        # ==========================================================
        # 将视频折叠为 [B*T, 3, H, W] 逐帧输入 CLIP
        frames = video.transpose(1, 2).reshape(B * T, C, H, W)
        with torch.no_grad():
            clip_outputs = self.clip(pixel_values=frames)
            # 使用全局 [CLS] token 表征整帧的语义 [B*T, 1024]
            semantic_features = clip_outputs.pooler_output 
            
        # 恢复批次和时序维度 [B, T, 1024]
        semantic_tokens = semantic_features.view(B, T, -1) 
        semantic_embeds = self.semantic_proj(semantic_tokens) # [B, T, 3584]

        # ==========================================================
        # 2. 运动分支 (SlowFast) - 提取 T 个 Token
        # ==========================================================
        sf_input = self.prepare_slowfast_input(video)
        # [B, 2304, T, 1, 1]
        pooled_motion = self.slowfast(sf_input) 
        # [B, T, 2304]
        motion_tokens = pooled_motion.flatten(3).transpose(1, 2) 
        motion_embeds = self.motion_proj(motion_tokens) # [B, T, 3584]

        # ==========================================================
        # 3. 美学分支 (ConvNeXt) - 提取 T 个 Token
        # ==========================================================
        # [B, 768, T', H', W']
        aes_features = self.aesthetic_conv3d(video_aesthetic) 
        # [B, 768, T, 1, 1]
        pooled_aes = self.aesthetic_pool(aes_features)
        # [B, T, 768]
        aesthetic_tokens = pooled_aes.flatten(3).transpose(1, 2) 
        aesthetic_embeds = self.aesthetic_proj(aesthetic_tokens) # [B, T, 3584]

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