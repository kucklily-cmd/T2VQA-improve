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

        # ---------- 视觉-文本编码器（BLIP） ----------
        # 这里用 BLIP 的 text_encoder 读取 caption，并通过 cross-attn 融合每帧的视觉 token
        self.blip = BLIP_Pretrain(image_size = image_size, vit = 'large', embed_dim = embed_dim, med_config = med_config)
        # 反序列化python对象，加载 BLIP 预训练权重
        state_dict = torch.load(args['blip_weights'], map_location='cpu')

        # 将state_dict的内容键张量对加载到模型里面的对应参数，模型有关参数在model键下，False表示不严格匹配
        self.blip.load_state_dict(state_dict["model"], strict=False)

        for name, param in self.blip.named_parameters():
            if ("text_encoder" in name):
                # 是否计算梯度，反向传播是否更新
                param.requires_grad = True
            else:
                param.requires_grad = False

        # 把 BLIP text_encoder 输出投到 embed_dim（后续作为多帧语义 token）
        self.finetune_text_proj = nn.Linear(self.blip.text_encoder.config.hidden_size, embed_dim)
        # 新增一个纯文本编码器提取语义锚点
        # 加载标准的 bert-base-uncased，默认 add_cross_attention 是 False
        self.pure_text_encoder = BertModel.from_pretrained(args['bert_weights'])
        
        # 冻结这个纯文本编码器（视显存情况而定，建议冻结）
        for param in self.pure_text_encoder.parameters():
            param.requires_grad = False

        # ---------- 语言模型（LLM） ----------
        # LLM 本体冻结，仅用作“把多模态 token + 文本 prompt”映射到质量词的 logits
        self.llm_tokenizer = LlamaTokenizer.from_pretrained(llm_model, use_fast=False)
        self.llm_model = LlamaForCausalLM.from_pretrained(
            llm_model, torch_dtype=torch.float16
        )
        # 设置词表
        # 特殊标记的添加时为了确保LLM可以正确处理输入序列，开始结束和词汇表外的词汇
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
        text_hidden_size = self.blip.text_encoder.config.hidden_size
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
        # 接收双分支数据
        video_fidelity = data['video_fidelity'] 
        video_semantic = data['video_semantic'] 

        # 1. 优先获取全局纯文本特征作为 Semantic Anchor
        text = self.blip.tokenizer(caption, padding='max_length', truncation=True, max_length=35, return_tensors="pt").to(video_fidelity.device)
        text_output = self.pure_text_encoder(text.input_ids, attention_mask=text.attention_mask, return_dict=True)
        global_text_feat = text_output.last_hidden_state[:, 0, :] 

        # 2. 技术质量特征：喂入高帧率保真度张量
        f_swin = self.swin3d(video_fidelity) 
        B, C_s, T_s, H_s, W_s = f_swin.shape
        f_swin_flat = f_swin.view(B, C_s, -1).transpose(1, 2) 
        pooled_swin = self.swin_attn_pool(global_text_feat, f_swin_flat) 

        f_conv = self.conv3d(video_fidelity) 
        B, C_c, T_c, H_c, W_c = f_conv.shape
        f_conv_flat = f_conv.view(B, C_c, -1).transpose(1, 2)
        pooled_conv = self.conv_attn_pool(global_text_feat, f_conv_flat) 

        # 3. 文本条件引导的 GateMixer
        inputs_swin = self.gate_mixer(pooled_swin, pooled_conv, global_text_feat)
        atts_swin = torch.ones(inputs_swin.size()[:-1], dtype=torch.long).to(video_fidelity.device)

        inputs_llm = []
        text = self.blip.tokenizer(caption, padding='max_length', truncation=True, max_length=35, return_tensors="pt").to(video_fidelity.device)
        
        # ---------- 多帧语义 token：喂入低帧率语义张量 ----------
        for j in range(video_semantic.size(2)):
            image = video_semantic[:,:,j,:,:]
            image_embeds = self.blip.visual_encoder(image)
            image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(video_semantic.device)

            output = self.blip.text_encoder(text.input_ids,
                                                attention_mask = text.attention_mask,
                                                encoder_hidden_states = image_embeds,
                                                encoder_attention_mask = image_atts,
                                                return_dict = True,
                                            )
            output = self.finetune_text_proj(output.last_hidden_state[:,0,:])
            inputs_llm.append(output)

        semantic_tokens = torch.stack(inputs_llm, dim=1)
        semantic_tokens = self.finetune_semantic_proj(semantic_tokens)
        fidelity_tokens = self.finetune_fidelity_proj(inputs_swin)

        inputs_llm = torch.cat([fidelity_tokens, semantic_tokens], dim=1)
        atts_llm = torch.ones(inputs_llm.size()[:-1], dtype=torch.long).to(video_fidelity.device)

        llm_tokens = self.llm_tokenizer(
            [prompt] * video_fidelity.size(0),
            padding="longest",
            return_tensors="pt"
        ).to(video_fidelity.device)

        with self.maybe_autocast():
            inputs_embeds = self.llm_model.get_input_embeddings()(llm_tokens.input_ids)
            inputs_embeds = torch.cat([inputs_llm.to(dtype=inputs_embeds.dtype), inputs_embeds], dim=1)
            attention_mask = torch.cat([atts_llm, llm_tokens.attention_mask], dim=1)

            outputs = self.llm_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                )
                
        output_logits = outputs.logits[:, -1]
        lexcellent, lgood, lfair, lpoor, lbad = output_logits[:, self.excellent_idx], output_logits[:, self.good_idx], output_logits[:, self.fair_idx], output_logits[:,self.poor_idx], output_logits[:, self.bad_idx]

        # 强制将 FP16 的 logits 转换为 FP32 再进行除法和 softmax，防止底层数值溢出
        q_pred = (torch.stack([lexcellent, lgood, lfair, lpoor, lbad]).float() / 100).softmax(0)
        weights = self.weights.expand(-1, q_pred.shape[1]).to(video_fidelity.device)
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
