from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from perceiver.model.core import (
    DecoderConfig,
    EncoderConfig,
    FourierPositionEncoding,
    InputAdapter,
    OutputAdapter,
    PerceiverDecoder,
    PerceiverEncoder,
    PerceiverIO,
    PerceiverIOConfig,
    QueryProvider,
)


@dataclass
class AnomalyEncoderConfig(EncoderConfig):
    image_shape: Tuple[int, int, int] = (224, 224, 3)
    num_frequency_bands: int = 16
    params: Optional[str] = None


@dataclass
class AnomalyDecoderConfig(DecoderConfig):
    map_shape: Tuple[int, int] = (224,224)
    num_output_query_channels: int = 128 #MVTec数据小 256 decoder容易记忆
    num_output_channels: int = 1
    score_pool: str = "topk_mean"
    score_topk_ratio: float = 0.01

    # 新增：控制全局偏置和查询可训练性的配置
    use_global_bias: bool = False          # 默认关闭全局偏置
    trainable_query: bool = False          # 默认冻结空间查询


AnomalyDetectorConfig = PerceiverIOConfig[AnomalyEncoderConfig, AnomalyDecoderConfig]


class AnomalyImageInputAdapter(InputAdapter):
    """
    Input:  x (B, H, W, C)
    Output: (B, H*W, C + position_encoding_dim)
    """

    def __init__(self, image_shape: Tuple[int, int, int], num_frequency_bands: int):
        *spatial_shape, num_image_channels = image_shape
        position_encoding = FourierPositionEncoding(
            input_shape=spatial_shape,
            num_frequency_bands=num_frequency_bands,
        )
        super().__init__(
            num_input_channels=num_image_channels + position_encoding.num_position_encoding_channels()
        )
        self.image_shape = image_shape
        self.position_encoding = position_encoding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, *d = x.shape
        if tuple(d) != self.image_shape:
            raise ValueError(
                f"Input image shape {tuple(d)} different from required shape {self.image_shape}"
            )

        x_pos = self.position_encoding(b)
        x_seq = rearrange(x, "b h w c -> b (h w) c")  #图像展平成序列
        return torch.cat([x_seq, x_pos], dim=-1)    #拼接特征 +位置编码


class SpatialQueryProvider(nn.Module, QueryProvider):
    """
    Provides frozen spatial queries:
      output shape: (1, Hq*Wq, Q)

    This is intentionally frozen to reduce the risk that each output location
    memorizes a fixed anomaly heatmap template.
    """

    def __init__(self, map_shape: Tuple[int, int], num_query_channels: int, init_scale: float = 0.02, trainable_query: bool = False):
        super().__init__()
        self.map_shape = map_shape
        hq, wq = map_shape #保存查询图的 高度和宽度
        self._num_query_channels = num_query_channels

        self.query = nn.Parameter(
            torch.empty(hq * wq, num_query_channels),
            requires_grad=trainable_query,  # 关闭梯度更新
        )
        self._init_parameters(init_scale)

    # 参数初始化：正态分布
    def _init_parameters(self, init_scale: float):
        with torch.no_grad():
            self.query.normal_(mean=0.0, std=init_scale)

    @property  #接口属性  返回查询维度
    def num_query_channels(self) -> int:
        return self._num_query_channels

    def forward(self, x=None):
        # 返回固定查询  1 = batch size（所有图片共用同一套空间查询）
        return rearrange(self.query, "n c -> 1 n c")


class AnomalyMapOutputAdapter(OutputAdapter):
    """
    Input: decoder output (B, Hq*Wq, Q)
    Output:
      - anomaly_logits_lowres: (B, Hq, Wq, 1)
      - anomaly_logits:        (B, H, W, 1)
      - anomaly_prob:          (B, H, W, 1)
      - image_score:           (B,)
    """

    def __init__(
        self,
        image_shape: Tuple[int, int, int],
        map_shape: Tuple[int, int],
        num_output_query_channels: int,
        num_output_channels: int = 1,
        score_pool: str = "topk_mean",
        score_topk_ratio: float = 0.01,
        conv_channels:int = 96,  # 新增：卷积中间通道数
        use_global_bias: bool = False,
    ):
        super().__init__()
        self.image_shape = image_shape
        self.map_shape = map_shape
        self.score_pool = score_pool
        self.score_topk_ratio = score_topk_ratio
        self.use_global_bias=use_global_bias # 保存为实例属性

        # 投影到卷积特征空间
        self.proj_to_conv = nn.Linear(num_output_query_channels, conv_channels)
       
        # 增强的卷积 refine 模块（空洞卷积扩大感受野）
        self.conv_refine = nn.Sequential(
            nn.Conv2d(conv_channels, conv_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, conv_channels),
            nn.GELU(),

            nn.Conv2d(conv_channels, conv_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, conv_channels),
            nn.GELU(),
            # 第3层 空洞卷积 dilation=2  感受野扩大
            nn.Conv2d(conv_channels, conv_channels, kernel_size=3, padding=2, dilation=2),
            nn.GroupNorm(8, conv_channels),
            nn.GELU(),

            nn.Conv2d(conv_channels, conv_channels, kernel_size=3, padding=4, dilation=4),
            nn.GroupNorm(8, conv_channels),
            nn.GELU(),
            # 最后 1x1 卷积  把多通道特征 输出 1 个通道异常分数
            nn.Conv2d(conv_channels, num_output_channels, kernel_size=1),
        )

        for m in self.conv_refine.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight) # 卷积核初始化
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0) # 偏置初始化为0

        # 图像级偏置（global bias）: 为每张图学习一个整体偏移量
        if use_global_bias:
            self.global_bias_proj = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),          # 对序列维度池化: (B, Q, N) -> (B, Q, 1)
                nn.Flatten(start_dim=1),           # (B, Q)
                nn.Linear(num_output_query_channels, 1),
            )
            # 初始化偏置为0，让模型从零开始学习
            nn.init.zeros_(self.global_bias_proj[-1].weight)
            nn.init.zeros_(self.global_bias_proj[-1].bias)

    def _pool_image_score(self, logits_map: torch.Tensor) -> torch.Tensor:
        flat = rearrange(logits_map, "b h w c -> b (h w c)")

        if self.score_pool == "max":
            return flat.max(dim=1).values

        if self.score_pool == "mean":
            return flat.mean(dim=1)

        k = max(1, int(flat.shape[1] * self.score_topk_ratio))
        return torch.topk(flat, k=k, dim=1).values.mean(dim=1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        hq, wq = self.map_shape
        h, w, _ = self.image_shape
        
        #线性投影
        feat = self.proj_to_conv(x)
        feat = rearrange(feat, "b (hq wq) c -> b c hq wq", hq=hq, wq=wq)  #把序列重新变回 2D 图像特征图
        
        # Conv Refine：在低分辨率上做空间细化
        feat = self.conv_refine(feat)  # 输出: (B, 1, hq, wq)
        
        #图像级偏置
        if self.use_global_bias:
            # x shape: (B, N, Q) -> 计算全局偏置
            global_bias = self.global_bias_proj(x.transpose(1, 2))  # (B, 1)
            global_bias = global_bias.view(-1, 1, 1, 1)             # (B, 1, 1, 1)
            feat = feat + global_bias

        # 上采样到原图尺寸（若 map_shape == image_shape，则无插值）
        logits_full = F.interpolate(
            feat, size=(h, w), mode="bilinear", align_corners=False
        )
        logits_full = rearrange(logits_full, "b c h w -> b h w c")
        logits_lowres = rearrange(feat, "b c hq wq -> b hq wq c")

        # 计算两种图像级分数：基于 logits 和基于概率
        probs_full = torch.sigmoid(logits_full)

        # 保留原有的 image_score（基于 logits）用于 BCE 损失
        image_score_logits = self._pool_image_score(logits_full)
        # 基于概率的图像级分数，用于 AUROC 或监控
        image_score_prob = self._pool_image_score(probs_full)
        
        return {
            "anomaly_logits_lowres": logits_lowres,
            "anomaly_logits": logits_full,
            "anomaly_prob": probs_full,
            "image_score": image_score_logits,          # 保持向后兼容
            "image_score_prob": image_score_prob,       # 新增字段

        }


class AnomalyDetector(PerceiverIO):
    """
    Forward input:
      x: (B, H, W, C)

    Forward output:
      dict from AnomalyMapOutputAdapter
    """

    def __init__(self, config: AnomalyDetectorConfig):
        input_adapter = AnomalyImageInputAdapter(
            image_shape=config.encoder.image_shape,
            num_frequency_bands=config.encoder.num_frequency_bands,
        )

        encoder_kwargs = config.encoder.base_kwargs()
        if encoder_kwargs["num_cross_attention_qk_channels"] is None:
            encoder_kwargs["num_cross_attention_qk_channels"] = input_adapter.num_input_channels

        encoder = PerceiverEncoder(
            input_adapter=input_adapter,
            num_latents=config.num_latents,
            num_latent_channels=config.num_latent_channels,
            activation_checkpointing=config.activation_checkpointing,
            activation_offloading=config.activation_offloading,
            **encoder_kwargs,
        )

        output_query_provider = SpatialQueryProvider(
            map_shape=config.decoder.map_shape,
            num_query_channels=config.decoder.num_output_query_channels,
            init_scale=config.decoder.init_scale,
            trainable_query=config.decoder.trainable_query,
        )

        output_adapter = AnomalyMapOutputAdapter(
            image_shape=config.encoder.image_shape,
            map_shape=config.decoder.map_shape,
            num_output_query_channels=config.decoder.num_output_query_channels,
            num_output_channels=config.decoder.num_output_channels,
            score_pool=config.decoder.score_pool,
            score_topk_ratio=config.decoder.score_topk_ratio,
            use_global_bias=config.decoder.use_global_bias, # 关闭图像级偏置
        )

        decoder = PerceiverDecoder(
            output_adapter=output_adapter,
            output_query_provider=output_query_provider,
            num_latent_channels=config.num_latent_channels,
            activation_checkpointing=config.activation_checkpointing,
            activation_offloading=config.activation_offloading,
            **config.decoder.base_kwargs(
                exclude=(
                    "freeze",
                    "num_output_query_channels",
                    "num_output_channels",
                    "map_shape",
                    "score_pool",
                    "score_topk_ratio",
                    "use_global_bias",
                    "trainable_query",
                )
            ),
        )

        super().__init__(encoder, decoder)
        self.config = config

    def forward(self, x: torch.Tensor, pad_mask=None) -> Dict[str, torch.Tensor]:
        latents = self.encoder(x, pad_mask=pad_mask)
        return self.decoder(latents)