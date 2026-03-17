import contextlib
from transformers import LlamaForCausalLM, LlamaTokenizer#, BertLMHeadModel

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


class GateMixer(nn.Module):
    def __init__(
        self,
        v_in_dim,
        c_in_dim,
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
        self.w_g = nn.Linear(2 * d, d)
        if prefix_len > 0:
            self.h_p = nn.Parameter(torch.zeros(1, prefix_len, d))
            nn.init.normal_(self.h_p, mean=0.0, std=0.02)
        else:
            self.h_p = None
        self.w2 = nn.Linear(d, out_dim or d)

    def forward(self, v_v, v_c):
        h_v = self.w1_v(v_v).unsqueeze(1).expand(-1, self.token_len, -1)
        h_c = self.w1_c(v_c).unsqueeze(1).expand(-1, self.token_len, -1)
        alpha_v = torch.sigmoid(self.w_g(torch.cat([h_v, h_c], dim=-1)))
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
        self.swin_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

        self.conv3d = convnext_3d_tiny(
            pretrained=args.get("conv_pretrained", False),
            in_22k=args.get("conv_in_22k", False),
            checkpoint=args.get("conv_weights", None),
        )
        self.conv_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

        self.gate_mixer = GateMixer(
            v_in_dim=768,
            c_in_dim=768,
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

        # ---------- 技术质量 token（Swin3D -> 32 个 query token） ----------
        f_swin = self.swin3d(video)
        f_swin = self.swin_avg_pool(f_swin).view(video.size(0), -1)

        f_conv = self.conv3d(video)
        f_conv = self.conv_avg_pool(f_conv).view(video.size(0), -1)

        inputs_swin = self.gate_mixer(f_swin, f_conv).to(video.device)
        atts_swin = torch.ones(inputs_swin.size()[:-1], dtype=torch.long).to(video.device)

        inputs_llm = []

        #人类能看懂的句子（caption）转换成模型能处理的数字矩阵（Tokens），并统一所有句子的长度。
        #将字符串拆分成“词元”（Tokens）。例如把 "A cat is running" 拆解并映射为词表中的索引数字，如 [101, 134, 567, ...]。
        text = self.blip.tokenizer(caption, padding='max_length', truncation=True, max_length=35, 
                                  return_tensors="pt").to(video.device)
        
        # ---------- 多帧语义 token（逐帧：视觉 encoder + text_encoder cross-attn） ----------
        # 这个video是数据加载器封装的五个维度的那个
        for j in range(video.size(2)):#size（2）获取第三个维度的内容
            #遍历每一帧
            image = video[:,:,j,:,:]

            image_embeds = self.blip.visual_encoder(image)

            # 给图像 embedding 构造一个 全1的 attention mask
            image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(video.device)

            # 交叉注意力机制
            output = self.blip.text_encoder(text.input_ids,
                                                attention_mask = text.attention_mask,
                                                encoder_hidden_states = image_embeds,
                                                encoder_attention_mask = image_atts,
                                                return_dict = True,
                                            )

            # 取 [CLS] 作为该帧的语义摘要 token
            output = self.finetune_text_proj(output.last_hidden_state[:,0,:])


            inputs_llm.append(output)

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
