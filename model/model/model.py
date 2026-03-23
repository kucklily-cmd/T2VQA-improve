from transformers import CLIPVisionModel, BertTokenizer
import contextlib
from transformers import LlamaForCausalLM, LlamaTokenizer, BertModel

import torch
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict

import copy

#from model.attention import Transformer3DModel
from model.blip import create_vit, init_tokenizer, load_checkpoint
from model.blip_pretrain import BLIP_Pretrain
from model.swin import swin_3d_tiny, SwinTransformer3D, SwinTransformer2D
from model.conv_backbone import convnext_3d_tiny


from torch.nn import TransformerDecoderLayer, TransformerDecoder
from timm.models.vision_transformer import vit_base_patch16_224



def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module



def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

class CrossAttentionPooling(nn.Module):
    def __init__(self, text_dim, visual_dim, embed_dim, num_heads=8):
        super().__init__()
        # 将文本特征投影为 Query
        self.q_proj = nn.Linear(text_dim, embed_dim)
        # 将视觉时空 Token 投影为 Key 和 Value
        self.k_proj = nn.Linear(visual_dim, embed_dim)
        self.v_proj = nn.Linear(visual_dim, embed_dim)
        
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        
        # 前馈网络 (FFN) 增加非线性表达能力
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, text_feat, visual_tokens):
        """
        text_feat: [B, text_dim] (Semantic Anchor)
        visual_tokens: [B, T*H*W, visual_dim] (Unpooled 3D features)
        """
        # [B, text_dim] -> [B, 1, embed_dim]
        q = self.q_proj(text_feat).unsqueeze(1) 
        # [B, T*H*W, visual_dim] -> [B, T*H*W, embed_dim]
        k = self.k_proj(visual_tokens)
        v = self.v_proj(visual_tokens)
        
        # 交叉注意力计算
        attn_out, _ = self.attn(q, k, v)  # 输出形状: [B, 1, embed_dim]
        
        # 残差连接与归一化
        out = self.norm1(q + attn_out)
        ffn_out = self.ffn(out)
        out = self.norm2(out + ffn_out)
        
        # 去掉序列维度，返回池化后的对齐特征: [B, embed_dim]
        return out.squeeze(1)
class GateMixer(nn.Module):
    def __init__(
        self,
        v_in_dim,
        c_in_dim,
        text_dim,  # 新增文本特征维度
        d,
        token_len=32,
        prefix_len=8,
        out_dim=None,
    ):
        super().__init__()
        self.token_len = token_len
        self.prefix_len = prefix_len
        self.w1_v = nn.Linear(v_in_dim, d)
        self.w1_c = nn.Linear(c_in_dim, d)
        # 门控全连接层现在接收：Swin特征(d) + Conv特征(d) + 文本特征(text_dim)
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
        
        # 将文本特征也扩展到序列长度参与门控计算
        text_feat_exp = text_feat.unsqueeze(1).expand(-1, self.token_len, -1)
        
        # 模型现在可以根据 Prompt 决定倾向于全局结构还是局部纹理
        alpha_v = torch.sigmoid(self.w_g(torch.cat([h_v, h_c, text_feat_exp], dim=-1)))
        h = (1 - alpha_v) * h_v + alpha_v * h_c
        
        if self.h_p is not None:
            h = torch.cat([self.h_p.expand(h.size(0), -1, -1), h], dim=1)
        return self.w2(h)

class T2VQA(nn.Module):
    # python的属性字段在init函数声明，self.xx = xx
    def __init__(self,
                 args):
        super().__init__()
    
        # ---------- 基础配置 ----------
        # 读取配置参数
        med_config = args['med_config']
        image_size = args['image_size']
        embed_dim = args['embed_dim']#不同模态嵌入维度
        llm_model = args['llm_model']


        clip_version = args.get('clip_weights', "openai/clip-vit-large-patch14")
        self.clip_vision = CLIPVisionModel.from_pretrained(clip_version)
        
        # 设置 CLIP 是否参与微调 (视显存情况而定)
        for param in self.clip_vision.parameters():
            param.requires_grad = False  # 或者设为 False 进行冻结

        
        
        self.pure_text_encoder = BertModel.from_pretrained(args['bert_weights'])
        # 修改后：使用 pure_text_encoder 获取维度
        self.finetune_text_proj = nn.Linear(self.pure_text_encoder.config.hidden_size, embed_dim)
       
        for param in self.pure_text_encoder.parameters():
            param.requires_grad = False
        # 因为移除了 BLIP，需要独立加载一个 BertTokenizer 配合 pure_text_encoder 使用
        self.text_tokenizer = BertTokenizer.from_pretrained(args['bert_weights'])
        # ==================== 新增: 逐帧交叉注意力池化器 ====================
        # 用来替代原先 BLIP 的 text_encoder，将 CLIP 的空间特征与全局文本 Token 融合
        clip_hidden_size = self.clip_vision.config.hidden_size
        # 修改后：同样使用 pure_text_encoder 获取文本维度
        text_hidden_size = self.pure_text_encoder.config.hidden_size
        self.frame_attn_pool = CrossAttentionPooling(
            text_dim=text_hidden_size, 
            visual_dim=clip_hidden_size, 
            embed_dim=embed_dim
        )

        self.llm_tokenizer = LlamaTokenizer.from_pretrained(llm_model, use_fast=False)
        self.llm_model = LlamaForCausalLM.from_pretrained(
            llm_model, torch_dtype=torch.float16
        )
    
        self.llm_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.llm_tokenizer.add_special_tokens({'bos_token': '</s>'})
        self.llm_tokenizer.add_special_tokens({'eos_token': '</s>'})
        self.llm_tokenizer.add_special_tokens({'unk_token': '</s>'})

        #添加新标记的时候同时拓展词嵌入层
        self.llm_model.resize_token_embeddings(len(self.llm_tokenizer))

        llm_safetensors_index = args.get("llm_safetensors_index", None)
        if llm_safetensors_index:
            self._load_llm_from_safetensors_index(
                llm_safetensors_index,
                prefix_to_strip=args.get("llm_safetensors_prefix_to_strip", "llm."),
            )

        self.finetune_semantic_proj = nn.Linear(embed_dim, self.llm_model.config.hidden_size)
        self.finetune_fidelity_proj = nn.Linear(embed_dim, self.llm_model.config.hidden_size)
        
        #保证llm在训练过程中不变化
        for name, param in self.llm_model.named_parameters():#获取里面所有变量（模型参数nn.Parameter）
                param.requires_grad = False#关闭梯度
        self.llm_model = self.llm_model.eval()
        self.llm_model.train = disabled_train

        # 最终从 LLM 的 vocab logits 中取这 5 个词的打分
        # 词表中五个单词转换为数字列表
        self.excellent_idx, self.good_idx, self.fair_idx, self.poor_idx, self.bad_idx = self.llm_tokenizer(["excellent", "good","fair", "poor", "bad"])['input_ids']
        self.excellent_idx = self.excellent_idx[1]
        self.good_idx = self.good_idx[1]
        self.fair_idx = self.fair_idx[1]
        self.poor_idx = self.poor_idx[1]
        self.bad_idx = self.bad_idx[1]

        # ---------- 技术质量分支（Swin3D） ----------
        # 用 3D Swin 从视频 clip 中抽取技术质量/时空结构表征，并扩展成固定长度的 query token（32）
        self.swin3d = swin_3d_tiny()
        state_dict = torch.load(args['swin_weights'], map_location='cpu')
        state_dict = state_dict['state_dict']
        
        #我的状态字典，有序状态字典
        # 传入状态字典可以和我的模型名字对齐
        i_state_dict = OrderedDict()
        for key in state_dict.keys():
            if "head" in key:
                continue
            if "cls" in key:
                tkey = key.replace("cls", "vqa")
            elif "backbone" in key:
                tkey = key.replace("backbone.", "")
                i_state_dict[tkey] = state_dict[key]
            else:
                i_state_dict[key] = state_dict[key]
            
        print(self.swin3d.load_state_dict(i_state_dict, strict=False))
        
        #自适应平均池化，指定输出的尺寸
        # self.swin_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

        self.conv3d = convnext_3d_tiny(
            pretrained=args.get("conv_pretrained", False),
            in_22k=args.get("conv_in_22k", False),
            checkpoint=args.get("conv_weights", None),
        )
        # self.conv_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

        # 新增交叉注意力池化器 (假设 Swin3D 和 ConvNext3D 输出通道都是 768)
        text_hidden_size = self.pure_text_encoder.config.hidden_size
        self.swin_attn_pool = CrossAttentionPooling(text_dim=text_hidden_size, visual_dim=768, embed_dim=embed_dim)
        self.conv_attn_pool = CrossAttentionPooling(text_dim=text_hidden_size, visual_dim=768, embed_dim=embed_dim)

        self.gate_mixer = GateMixer(
            v_in_dim=embed_dim,    # 注意这里改为了 embed_dim，因为 attn_pool 输出是 embed_dim
            c_in_dim=embed_dim,    
            text_dim=text_hidden_size, # 传入文本维度
            d=embed_dim,
            token_len=args.get("gatemixer_token_len", 32),
            prefix_len=args.get("gatemixer_prefix_len", 8),
            out_dim=embed_dim,
        )

        # 将 5 个等级映射到数值权重（1~5），用于把 5 个词的概率加权成最终分数
        self.weights = torch.Tensor([[1], [2], [3], [4], [5]])


    def quality_regression(self,in_channels, middle_channels, out_channels):
        regression_block = nn.Sequential(
            nn.Linear(in_channels, middle_channels),
            nn.Linear(middle_channels, out_channels),          
        )

        return regression_block


    def device(self):
        return list(self.parameters())[0].device

    def maybe_autocast(self, dtype=torch.float16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()

    def _load_llm_from_safetensors_index(self, index_json_path: str, prefix_to_strip: str = "llm."):
        import json
        import os

        try:
            from safetensors.torch import load_file
        except Exception as e:
            raise ModuleNotFoundError(
                "Missing dependency `safetensors`. Install it to load *.safetensors shards."
            ) from e

        with open(index_json_path, "r", encoding="utf-8") as f:
            index = json.load(f)
        weight_map = index.get("weight_map", {})
        shard_to_keys = {}
        for k, shard_name in weight_map.items():
            if prefix_to_strip and not k.startswith(prefix_to_strip):
                continue
            shard_to_keys.setdefault(shard_name, []).append(k)

        base_dir = os.path.dirname(index_json_path)
        remapped_state = {}
        for shard_name, keys in shard_to_keys.items():
            shard_path = os.path.join(base_dir, shard_name)
            if not os.path.exists(shard_path):
                raise FileNotFoundError(f"Missing shard file: {shard_path}")
            shard_state = load_file(shard_path, device="cpu")
            for k in keys:
                new_k = k[len(prefix_to_strip):] if prefix_to_strip else k
                if k in shard_state:
                    remapped_state[new_k] = shard_state[k]

        self.llm_model.load_state_dict(remapped_state, strict=False)

    def forward(self, data, caption, prompt):
        video = data['video']

        text = self.text_tokenizer(caption, padding='max_length', truncation=True, max_length=35, return_tensors="pt").to(video.device)
        
        text_output = self.pure_text_encoder(text.input_ids, attention_mask=text.attention_mask, return_dict=True)
        global_text_feat = text_output.last_hidden_state[:, 0, :] # [B, text_dim]

        # 2. 技术质量特征：用 Cross-Attention Pooling 替代 AvgPool 展平
        f_swin = self.swin3d(video) # 原始形状: [B, C, T, H, W]
        B, C_s, T_s, H_s, W_s = f_swin.shape
        # 展平为 [B, T*H*W, C] 
        f_swin_flat = f_swin.view(B, C_s, -1).transpose(1, 2) 
        # 文本引导对齐
        pooled_swin = self.swin_attn_pool(global_text_feat, f_swin_flat) # [B, embed_dim]

        f_conv = self.conv3d(video) # 原始形状: [B, C, T, H, W]
        B, C_c, T_c, H_c, W_c = f_conv.shape
        # 展平为 [B, T*H*W, C]
        f_conv_flat = f_conv.view(B, C_c, -1).transpose(1, 2)
        # 文本引导对齐
        pooled_conv = self.conv_attn_pool(global_text_feat, f_conv_flat) # [B, embed_dim]

        # 3. 文本条件引导的 GateMixer
        # inputs_swin 此时包含高度对齐的时空技术质量 tokens
        inputs_swin = self.gate_mixer(pooled_swin, pooled_conv, global_text_feat)
        
        
        atts_swin = torch.ones(inputs_swin.size()[:-1], dtype=torch.long).to(video.device)

        inputs_llm = []

        # ---------- 新的：多帧语义 token 提取 (基于 CLIP 视觉特征 + Cross Attention) ----------
        for j in range(video.size(2)):
            image = video[:,:,j,:,:] # [B, 3, H, W]

            # 1. 使用 CLIP 提取逐帧的空间特征 (Spatial Features)
            # 输出包含 pooler_output 和 last_hidden_state
            clip_outputs = self.clip_vision(pixel_values=image)
            
            # 取出空间 Patch 特征: [B, N, clip_hidden_size]
            image_embeds = clip_outputs.last_hidden_state 

            # 2. 跨模态融合: 使用你定义的 CrossAttentionPooling
            # 让纯文本的 Semantic Anchor (global_text_feat) 去查询(Query)每一帧的 CLIP 空间特征(Key/Value)
            # 输出形状: [B, embed_dim]
            frame_semantic_token = self.frame_attn_pool(global_text_feat, image_embeds)

            inputs_llm.append(frame_semantic_token)

        semantic_tokens = torch.stack(inputs_llm, dim=1)
        semantic_tokens = self.finetune_semantic_proj(semantic_tokens)
        fidelity_tokens = self.finetune_fidelity_proj(inputs_swin)

        inputs_llm = torch.cat([fidelity_tokens, semantic_tokens], dim=1)
        atts_llm = torch.ones(inputs_llm.size()[:-1], dtype=torch.long).to(video.device)

        
        # LLM提示词转换为数字矩阵
        llm_tokens = self.llm_tokenizer(
        # ---------- 文本提示词 token（prompt） ----------
            [prompt] * video.size(0),# 将同一个字符串 prompt 重复B次，组成一个列表。
            padding="longest",# 自动补长
            return_tensors="pt"# 返回pt张量
        ).to(video.device)

        # 是否开启混合精度
        with self.maybe_autocast():
            # 调用 LLM 自带的嵌入层（Embedding Layer），将之前生成的数字编号（input_ids）映射为高维稠密向量。
            inputs_embeds = self.llm_model.get_input_embeddings()(llm_tokens.input_ids)
            
            # 将 Token (inputs_llm) 拼在文本 Token (inputs_embeds) 的前面
            inputs_embeds = torch.cat([inputs_llm.to(dtype=inputs_embeds.dtype), inputs_embeds], dim=1)
            
            #同样在序列维度（dim=1）上，将视觉部分的“全 1 掩码”和文本部分的“填充掩码”拼在一起。
            attention_mask = torch.cat([atts_llm, llm_tokens.attention_mask], dim=1)

            outputs = self.llm_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,

                )
        # 从 LLM 的输出中取 最后一个 Token（即提示词结束后的第一个预测位）的 Logits
        output_logits = outputs.logits[:, -1]

        # 拥有几万个词的概率分布中，精准挑出你最开始获取的那 5 个索引（excellent, good 等）对应的数值。
        lexcellent, lgood, lfair, lpoor, lbad = output_logits[:, self.excellent_idx], output_logits[:, self.good_idx], output_logits[:, self.fair_idx], output_logits[:,self.poor_idx], output_logits[:, self.bad_idx]

        #归一化
        q_pred = (torch.stack([lexcellent, lgood, lfair, lpoor, lbad]) / 100).softmax(0)

        #加权得分
        weights = self.weights.expand(-1, q_pred.shape[1]).to(video.device)
        q_pred = torch.mul(q_pred, weights)

        q_pred = torch.sum(q_pred, dim=0)

        return q_pred








if __name__=="__main__":
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    model = T2VQA(med_config='../configs/med_config.json', image_size = 224).to(device)
    model.eval()
    caption = 'A random caption'
    prompt = 'Please assess the quality of this image'
    video = torch.randn(2, 3, 8, 224, 224).to(device)

    with torch.no_grad():
        output = model(video, caption, prompt)
    print(output)        
