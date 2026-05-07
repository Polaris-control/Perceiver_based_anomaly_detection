from dataclasses import dataclass
from typing import Dict, Tuple

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
    image_shape: Tuple[int, int, int] = (256, 256, 3)
    num_frequency_bands: int = 64


@dataclass
class AnomalyDecoderConfig(DecoderConfig):
    # spatial query grid (low-res map). final map is resized to image_shape.
    map_shape: Tuple[int, int] = (64, 64)
    num_output_query_channels: int = 256
    num_output_channels: int = 1
    score_pool: str = "topk_mean"  # "max", "mean", "topk_mean"
    score_topk_ratio: float = 0.01


AnomalyDetectorConfig = PerceiverIOConfig[AnomalyEncoderConfig, AnomalyDecoderConfig]


class AnomalyImageInputAdapter(InputAdapter):
    """
    Input:  x (B, H, W, C) channels-last
    Output: (B, H*W, C + pos_dim)
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

        x_pos = self.position_encoding(b)              # (B, H*W, pos_dim)
        x_seq = rearrange(x, "b h w c -> b (h w) c")  # (B, H*W, C)
        return torch.cat([x_seq, x_pos], dim=-1)      # (B, H*W, C')


class SpatialQueryProvider(nn.Module, QueryProvider):
    """
    Provides trainable spatial queries:
      output shape (1, Hq*Wq, Q)
    """

    def __init__(self, map_shape: Tuple[int, int], num_query_channels: int, init_scale: float = 0.02):
        super().__init__()
        self.map_shape = map_shape
        hq, wq = map_shape
        self._num_query_channels = num_query_channels
        self.query = nn.Parameter(torch.empty(hq * wq, num_query_channels))
        self._init_parameters(init_scale)

    def _init_parameters(self, init_scale: float):
        with torch.no_grad():
            self.query.normal_(mean=0.0, std=init_scale)

    @property
    def num_query_channels(self) -> int:
        return self._num_query_channels

    def forward(self, x=None):
        return rearrange(self.query, "n c -> 1 n c")


class AnomalyMapOutputAdapter(OutputAdapter):
    """
    Input: decoder output (B, Hq*Wq, Q)
    Output dict:
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
    ):
        super().__init__()
        self.image_shape = image_shape
        self.map_shape = map_shape
        self.score_pool = score_pool
        self.score_topk_ratio = score_topk_ratio
        self.linear = nn.Linear(num_output_query_channels, num_output_channels)

    def _pool_image_score(self, logits_map: torch.Tensor) -> torch.Tensor:
        # logits_map: (B, H, W, 1)
        b, h, w, c = logits_map.shape
        flat = rearrange(logits_map, "b h w c -> b (h w c)")

        if self.score_pool == "max":
            return flat.max(dim=1).values
        if self.score_pool == "mean":
            return flat.mean(dim=1)

        # topk_mean
        k = max(1, int(flat.shape[1] * self.score_topk_ratio))
        topk = torch.topk(flat, k=k, dim=1).values
        return topk.mean(dim=1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        hq, wq = self.map_shape
        h, w, _ = self.image_shape

        logits_lowres = self.linear(x)  # (B, Hq*Wq, 1)
        logits_lowres = rearrange(logits_lowres, "b (hq wq) c -> b hq wq c", hq=hq, wq=wq)

        # interpolate in channels-first then return channels-last
        logits_ch_first = rearrange(logits_lowres, "b hq wq c -> b c hq wq")
        logits_full = F.interpolate(
            logits_ch_first,
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        )
        logits_full = rearrange(logits_full, "b c h w -> b h w c")

        probs = torch.sigmoid(logits_full)
        score = self._pool_image_score(logits_full)

        return {
            "anomaly_logits_lowres": logits_lowres,
            "anomaly_logits": logits_full,
            "anomaly_prob": probs,
            "image_score": score,
        }


class AnomalyDetector(PerceiverIO):
    """
    Forward input:
      x: (B, H, W, C), channels-last
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
        )

        output_adapter = AnomalyMapOutputAdapter(
            image_shape=config.encoder.image_shape,
            map_shape=config.decoder.map_shape,
            num_output_query_channels=config.decoder.num_output_query_channels,
            num_output_channels=config.decoder.num_output_channels,
            score_pool=config.decoder.score_pool,
            score_topk_ratio=config.decoder.score_topk_ratio,
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
                "map_shape", "score_pool", 
                "score_topk_ratio")),
        )

        super().__init__(encoder, decoder)
        self.config = config

    def forward(self, x: torch.Tensor, pad_mask=None) -> Dict[str, torch.Tensor]:
        latents = self.encoder(x, pad_mask=pad_mask)
        return self.decoder(latents)