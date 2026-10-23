"""Image-conditioned, multiscale PromptIR-style adapters for source decoding.

No metadata, auxiliary predictions, or changes to the SAM mask token layout.
Scaled image features and decoder features jointly route each prompt mixture;
image features never enter the main decoder stream through a spatial skip.
Prompt banks and interaction layers are randomly initialized. A small nonzero
residual scale permits gradients to reach the routing/bank on the first step.
"""
from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass(frozen=True)
class PromptIRConfig:
    num_prompts: int = 5
    prompt_dim: int = 32
    prompt_size: int = 16
    condition_dim: int = 64
    hidden_dim: int = 64
    num_heads: int = 4
    init_scale: float = 0.1
    image_conditioning: bool = True
    image_channels: int = 16

    def __post_init__(self):
        for key in ('num_prompts','prompt_dim','prompt_size','condition_dim','hidden_dim','num_heads','image_channels'):
            if int(getattr(self,key)) < 1:raise ValueError(f'PromptIR {key} must be positive')
        if self.hidden_dim % self.num_heads:raise ValueError('PromptIR hidden_dim must divide into heads')
        if not 0 < self.init_scale <= 1:raise ValueError('PromptIR init_scale must be in (0,1]')


def normalize_prompt_ir_stages(stages: Sequence[str]) -> tuple[str,...]:
    allowed={'neck','stem','up1','up2','up3','up4','refine'}
    stages=tuple(stages)
    if not stages or len(set(stages))!=len(stages) or set(stages)-allowed:
        raise ValueError(f'PromptIR requires unique stages from {sorted(allowed)}, got {stages}')
    return stages


class PromptIRConditionEncoder(nn.Module):
    """Pool texture statistics from neck features, before source InstanceNorm."""
    def __init__(self, in_channels: int, condition_dim: int):
        super().__init__()
        self.features=nn.Sequential(nn.Conv2d(in_channels,32,1),nn.GELU(),
            nn.Conv2d(32,32,3,padding=1,groups=32),nn.GELU())
        self.project=nn.Sequential(nn.Linear(64,condition_dim),nn.GELU())

    def forward(self,x: Tensor) -> Tensor:
        x=self.features(x)
        stats=torch.cat((x.mean((-2,-1)),x.float().var((-2,-1),unbiased=False).add(1e-6).sqrt().to(x.dtype)),1)
        return self.project(stats)


_IMAGE_LEVELS = {'neck': 4, 'stem': 4, 'up1': 3, 'up2': 2,
                 'up3': 1, 'up4': 0, 'refine': 0}


def _masked_stats(x: Tensor, valid: Tensor) -> Tensor:
    """Per-channel spatial mean/std in float32, excluding invalid support."""
    values, weights = x.float(), valid.float()
    count = weights.sum((-2, -1)).clamp_min(1e-6)
    mean = (values * weights).sum((-2, -1)) / count
    variance = ((values - mean[..., None, None]).square() * weights).sum((-2, -1)) / count
    std = variance.clamp_min(0).add(1e-6).sqrt()
    return torch.cat((mean, std), dim=1).to(x.dtype)


class PromptIRImagePyramid(nn.Module):
    """Small CNN pyramid used exclusively for prompt routing, without skips.

    The first convolution runs at input resolution. Subsequent downsampling
    therefore pools learned features rather than shrinking the raw image first.
    No instance normalization removes the image's texture/amplitude statistics.
    """

    def __init__(self, base_channels: int, max_level: int):
        super().__init__()
        self.channels = (base_channels, max(1, 3 * base_channels // 2),
                         2 * base_channels, 3 * base_channels, 4 * base_channels)
        self.max_level = max_level
        self.blocks = nn.ModuleList()
        in_channels = 1
        for out_channels in self.channels[:max_level + 1]:
            self.blocks.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1), nn.GELU(),
                nn.Conv2d(out_channels, out_channels, 3, padding=1,
                          groups=out_channels), nn.GELU()))
            in_channels = out_channels

    def forward(self, images: Tensor, *, canvas_size: tuple[int, int],
                valid_mask: Tensor | None = None) -> tuple[tuple[Tensor, Tensor], ...]:
        if images.ndim != 4 or images.shape[1] not in (1, 3):
            raise ValueError('PromptIR images must be flat [N,1 or 3,H,W] scaled model inputs')
        finite = torch.isfinite(images).all(dim=1, keepdim=True)
        if valid_mask is not None:
            if valid_mask.shape != finite.shape:
                raise ValueError('PromptIR image valid_mask must have shape [N,1,H,W]')
            finite = finite & valid_mask.to(device=images.device, dtype=torch.bool)
        # A grayscale condition is shared across the source decoder's RGB/gray
        # input paths; caller-supplied validity survives prior nan_to_num.
        x = torch.nan_to_num(images.float(), nan=0., posinf=0., neginf=0.).mean(dim=1, keepdim=True)
        x = x * finite
        height, width = images.shape[-2:]
        canvas_h, canvas_w = canvas_size
        if height > canvas_h or width > canvas_w:
            raise ValueError('PromptIR input image exceeds the decoder padded canvas')
        padding = (0, canvas_w - width, 0, canvas_h - height)
        x = F.pad(x, padding)
        valid = F.pad(finite.float(), padding)
        features = []
        for level, block in enumerate(self.blocks):
            if level:
                support = F.avg_pool2d(valid, 2)
                # Prevent invalid/padded cells from biasing pooled features.
                x = F.avg_pool2d(x * valid, 2) / support.clamp_min(1e-6)
                valid = support
            x = block(x) * (valid > 0).to(x.dtype)
            features.append((x, valid))
        return tuple(features)


class PromptIRImageCondition(nn.Module):
    """Fuse aligned image/decoder features ONLY inside the routing branch."""

    def __init__(self, decoder_channels: int, image_channels: int, condition_dim: int):
        super().__init__()
        self.decoder_project = nn.Conv2d(decoder_channels, image_channels, 1)
        self.image_project = nn.Conv2d(image_channels, image_channels, 1)
        self.joint = nn.Sequential(nn.GELU(),
            nn.Conv2d(image_channels, image_channels, 3, padding=1, groups=image_channels), nn.GELU())
        self.project = nn.Sequential(nn.Linear(2 * image_channels, condition_dim), nn.GELU())

    def forward(self, decoder_features: Tensor, image_features: Tensor, valid: Tensor) -> Tensor:
        if image_features.shape[0] != decoder_features.shape[0] or image_features.shape[-2:] != decoder_features.shape[-2:]:
            raise ValueError('PromptIR image/decoder feature grids must align')
        support = (valid > 0).to(decoder_features.dtype)
        joint = (self.decoder_project(decoder_features) + self.image_project(image_features)) * support
        joint = self.joint(joint) * support
        return self.project(_masked_stats(joint, valid))


class _ChannelTransformer(nn.Module):
    """Restormer-style channel attention; no quadratic HW-by-HW attention."""
    def __init__(self, channels: int, heads: int):
        super().__init__();self.heads=heads
        self.norm1=nn.LayerNorm(channels);self.norm2=nn.LayerNorm(channels)
        self.qkv=nn.Conv2d(channels,3*channels,1,bias=False)
        self.qkv_dw=nn.Conv2d(3*channels,3*channels,3,padding=1,groups=3*channels,bias=False)
        self.temperature=nn.Parameter(torch.ones(heads,1,1))
        self.out=nn.Conv2d(channels,channels,1,bias=False)
        self.ffn_in=nn.Conv2d(channels,4*channels,1,bias=False)
        self.ffn_dw=nn.Conv2d(4*channels,4*channels,3,padding=1,groups=4*channels,bias=False)
        self.ffn_out=nn.Conv2d(2*channels,channels,1,bias=False)

    @staticmethod
    def norm(x,layer):return layer(x.permute(0,2,3,1)).permute(0,3,1,2).contiguous()

    def forward(self,x: Tensor) -> Tensor:
        b,c,h,w=x.shape
        q,k,v=self.qkv_dw(self.qkv(self.norm(x,self.norm1))).chunk(3,1)
        q=F.normalize(q.reshape(b,self.heads,c//self.heads,h*w),dim=-1)
        k=F.normalize(k.reshape(b,self.heads,c//self.heads,h*w),dim=-1)
        v=v.reshape(b,self.heads,c//self.heads,h*w)
        attn=((q@k.transpose(-2,-1))*self.temperature).softmax(-1)
        x=x+self.out((attn@v).reshape(b,c,h,w))
        a,b=self.ffn_dw(self.ffn_in(self.norm(x,self.norm2))).chunk(2,1)
        return x+self.ffn_out(F.gelu(a)*b)


class PromptIRBlock(nn.Module):
    def __init__(self, channels: int, config: PromptIRConfig, *, image_channels: int | None = None):
        super().__init__();self.channels=channels
        self.prompt_bank=nn.Parameter(torch.empty(config.num_prompts,config.prompt_dim,
                                                  config.prompt_size,config.prompt_size))
        nn.init.normal_(self.prompt_bank,std=.02)
        self.router=nn.Linear(channels+config.condition_dim,config.num_prompts)
        self.prompt_conv=nn.Conv2d(config.prompt_dim,config.prompt_dim,3,padding=1,bias=False)
        self.fuse=nn.Conv2d(channels+config.prompt_dim,config.hidden_dim,1)
        self.interaction=_ChannelTransformer(config.hidden_dim,config.num_heads)
        self.project_out=nn.Conv2d(config.hidden_dim,channels,1,bias=False)
        nn.init.normal_(self.project_out.weight,std=.01)
        self.scale=nn.Parameter(torch.tensor(float(config.init_scale)))
        self.image_condition = (PromptIRImageCondition(channels, image_channels, config.condition_dim)
                                if image_channels is not None else None)
        self.image_router = nn.Linear(config.condition_dim, config.num_prompts) if image_channels is not None else None

    def prompt_weights(self,x: Tensor,condition: Tensor,
                       image_context: tuple[Tensor, Tensor] | None = None) -> Tensor:
        logits = self.router(torch.cat((x.mean((-2,-1)),condition),1)).float()
        if self.image_condition is not None:
            if image_context is None:
                raise ValueError('Image-conditioned PromptIR requires scaled input images')
            image_features, valid = image_context
            image_condition = self.image_condition(x, image_features, valid)
            present = valid.sum((-2, -1)) > 0
            logits = logits + self.image_router(image_condition).float() * present
        return logits.softmax(-1)

    def forward(self,x: Tensor,condition: Tensor,
                image_context: tuple[Tensor, Tensor] | None = None) -> Tensor:
        weights=self.prompt_weights(x,condition,image_context)
        prompt=torch.einsum('bk,kchw->bchw',weights,self.prompt_bank.float()).to(x.dtype)
        prompt=F.interpolate(prompt,size=x.shape[-2:],mode='bilinear',align_corners=False)
        prompt=self.prompt_conv(prompt)
        delta=self.project_out(self.interaction(self.fuse(torch.cat((x,prompt),1))))
        return x+self.scale.to(x.dtype)*delta


class MultiScalePromptIR(nn.Module):
    def __init__(self, in_channels: int, stage_channels: Mapping[str,int], *,
                 stages: Sequence[str]=('stem','up1','up2'), config: Mapping | PromptIRConfig | None=None):
        super().__init__()
        self.config=config if isinstance(config,PromptIRConfig) else PromptIRConfig(**(config or {}))
        self.stages=normalize_prompt_ir_stages(stages)
        self.condition_encoder=PromptIRConditionEncoder(in_channels,self.config.condition_dim)
        self.image_pyramid = (PromptIRImagePyramid(self.config.image_channels,
            max(_IMAGE_LEVELS[stage] for stage in self.stages)) if self.config.image_conditioning else None)
        self.blocks=nn.ModuleDict({stage:PromptIRBlock(stage_channels[stage],self.config,
            image_channels=self.image_pyramid.channels[_IMAGE_LEVELS[stage]] if self.image_pyramid is not None else None)
            for stage in self.stages})
        # A tensor-only descriptor permits reconstruction from a bare state_dict.
        self.register_buffer('_architecture',torch.tensor([1,*[getattr(self.config,key) for key in
            ('num_prompts','prompt_dim','prompt_size','condition_dim','hidden_dim','num_heads')]],dtype=torch.int64))
        if self.image_pyramid is not None:
            # Keep v1's base descriptor/routers unchanged for warm-starting.
            self.register_buffer('_image_architecture', torch.tensor([1, self.config.image_channels], dtype=torch.int64))

    def prepare_image_condition(self, images: Tensor | None, *, canvas_size: tuple[int, int],
                                valid_mask: Tensor | None = None) -> dict[str, tuple[Tensor, Tensor]] | None:
        if self.image_pyramid is None:
            return None
        if images is None:
            raise ValueError('Image-conditioned PromptIR requires scaled input images')
        pyramid = self.image_pyramid(images, canvas_size=canvas_size, valid_mask=valid_mask)
        return {stage: pyramid[_IMAGE_LEVELS[stage]] for stage in self.stages}

    def forward(self,stage: str,x: Tensor,condition: Tensor,
                image_context: Mapping[str, tuple[Tensor, Tensor]] | None = None) -> Tensor:
        if stage not in self.blocks:
            return x
        return self.blocks[stage](x, condition, image_context[stage] if image_context is not None else None)


def prompt_ir_kwargs_from_state_dict(state: Mapping) -> dict:
    """Rebuild all PromptIR geometry without relying on a separate run config."""
    clean={}
    for key,value in state.items():
        key=str(key)
        while key.startswith(('module.','_orig_mod.')):key=key.split('.',1)[1]
        clean[key.replace('._orig_mod.','.')]=value
    prefix='decoder.prompt_ir.'
    if not any(k.startswith(prefix) for k in clean):return {'decoder_prompt_ir':False}
    key=prefix+'_architecture'
    if key not in clean:raise ValueError('PromptIR checkpoint is missing its architecture descriptor')
    values=clean[key].detach().cpu().tolist()
    if len(values)!=7 or values[0]!=1:raise ValueError(f'Unsupported PromptIR descriptor: {values}')
    names=('num_prompts','prompt_dim','prompt_size','condition_dim','hidden_dim','num_heads')
    config_values = dict(zip(names, map(int, values[1:])))
    image_key = prefix + '_image_architecture'
    config_values['image_conditioning'] = image_key in clean
    if image_key in clean:
        image_values = clean[image_key].detach().cpu().tolist()
        if len(image_values) != 2 or image_values[0] != 1:
            raise ValueError(f'Unsupported PromptIR image descriptor: {image_values}')
        config_values['image_channels'] = int(image_values[1])
    elif any(k.startswith(prefix + 'image_pyramid.') or '.image_condition.' in k or '.image_router.' in k
             for k in clean if k.startswith(prefix)):
        raise ValueError('PromptIR image checkpoint is missing its architecture descriptor')
    config=asdict(PromptIRConfig(**config_values))
    stages=tuple(stage for stage in ('neck','stem','up1','up2','up3','up4','refine')
                 if any(k.startswith(prefix+'blocks.'+stage+'.') for k in clean))
    return dict(decoder_prompt_ir=True,prompt_ir_stages=normalize_prompt_ir_stages(stages),prompt_ir_config=config)
