import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        if len(x.shape) == 4:
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
        elif len(x.shape) == 5:
            x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
        return x


class Block3D(nn.Module):
    def __init__(self, dim, drop_path=0.0, inflate_len=3, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv3d(
            dim,
            dim,
            kernel_size=(inflate_len, 7, 7),
            padding=(inflate_len // 2, 3, 3),
            groups=dim,
        )
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        residual = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 4, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 4, 1, 2, 3)
        x = residual + self.drop_path(x)
        return x


class ConvNeXt3D(nn.Module):
    def __init__(
        self,
        in_chans=3,
        inflate_strategy="131",
        depths=(3, 3, 9, 3),
        dims=(96, 192, 384, 768),
        drop_path_rate=0.0,
        layer_scale_init_value=1e-6,
    ):
        super().__init__()
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv3d(in_chans, dims[0], kernel_size=(2, 4, 4), stride=(2, 4, 4)),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv3d(dims[i], dims[i + 1], kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[
                    Block3D(
                        dim=dims[i],
                        inflate_len=int(inflate_strategy[j % len(inflate_strategy)]),
                        drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value,
                    )
                    for j in range(depths[i])
                ]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv3d, nn.Linear)):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def inflate_weights(self, s_state_dict):
        t_state_dict = self.state_dict()
        for key in t_state_dict.keys():
            if key not in s_state_dict:
                continue
            if t_state_dict[key].shape != s_state_dict[key].shape:
                if t_state_dict[key].ndim != 5 or s_state_dict[key].ndim != 4:
                    continue
                t = t_state_dict[key].shape[2]
                s_state_dict[key] = s_state_dict[key].unsqueeze(2).repeat(1, 1, t, 1, 1) / t
        self.load_state_dict(s_state_dict, strict=False)

    def forward_features(self, x, return_spatial=True):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        if return_spatial:
            x = self.norm(x.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)
            return x
        return self.norm(x.mean([-3, -2, -1]))

    def forward(self, x):
        return self.forward_features(x, return_spatial=True)


model_urls = {
    "convnext_tiny_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth",
    "convnext_tiny_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_22k_224.pth",
}


def convnext_3d_tiny(pretrained=False, in_22k=False, checkpoint=None, **kwargs):
    model = ConvNeXt3D(depths=(3, 3, 9, 3), dims=(96, 192, 384, 768), **kwargs)
    if pretrained:
        url = model_urls["convnext_tiny_22k"] if in_22k else model_urls["convnext_tiny_1k"]
        ckpt = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        model.inflate_weights(state)
    if checkpoint:
        ckpt = torch.load(checkpoint, map_location="cpu")
        state = ckpt.get("state_dict", ckpt)
        if isinstance(state, dict) and all(k.startswith("module.") for k in state.keys()):
            state = {k[len("module.") :]: v for k, v in state.items()}
        model.load_state_dict(state, strict=False)
    return model

