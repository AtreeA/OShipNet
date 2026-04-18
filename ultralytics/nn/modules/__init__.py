# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
Ultralytics modules.

Example:
    Visualize a module with Netron.
    ```python
    from ultralytics.nn.modules import *
    import torch
    import os

    x = torch.ones(1, 128, 40, 40)
    m = Conv(128, 128)
    f = f'{m._get_name()}.onnx'
    torch.onnx.export(m, x, f)
    os.system(f'onnxsim {f} {f} && open {f}')
    ```
"""

from .block import *
from .conv import (
    EnConcat,

    CBAM,
    ChannelAttention,
    Concat,
    Conv,
    Conv2,
    ConvTranspose,
    DWConv,
    DWConvTranspose2d,
    Focus,
    GhostConv,
    LightConv,
    RepConv,
    SpatialAttention
)
from .head import OBB, Classify, Detect, Pose, RTDETRDecoder, Segment, WorldDetect, v10Detect
from .transformer import (
    AIFI,
    MLP,
    DeformableTransformerDecoder,
    DeformableTransformerDecoderLayer,
    LayerNorm2d,
    MLPBlock,
    MSDeformAttn,
    TransformerBlock,
    TransformerEncoderLayer,
    TransformerLayer,
)
# from .ScConv import ScConv
# from .DySample import DySample
from .EMA import EMA
# from .swin_transformer import SwinTransformerBlock,PatchEmbed,PatchMerging
# from .AKConv import AKConv
# from .qihnet import ShuffleNetV2,Conv_maxpool
# from .yolomamba import ConvSSM, SSM


#------------------------------------------------------------------------------
# from .multi_conv import multi_conv
from .inceptionnext import MetaNeXtStage,MetaNeXtBlock
__all__ = (
    "SDFM",
    "C3k2_FMB",
    "SPPF_LSKA_EMA",
    "A2C2f",
    "SPPF_LSKA_ATT",
    "SPPF_LSKA",
    "C3k2_Sc",
    "SPP_CA",
    "EnConcat",
    "multi_conv",
    "MetaNeXtStage",
    "MetaNeXtBlock",
    "C2PSA",
    "AGENT_SwinTransformerBlock",
    "C3k2",

    "DynamicFilter",

    "v10Detect",


    # "ConvSSM",
    # "SSM",


    "ShuffleNetV2",
    "Conv_maxpool",

    "DySample",
    "ScConv",
    "EMA",
    "SwinTransformerBlock",
    "PatchEmbed",
    "PatchMerging",
    "AKConv",




    "Conv",
    "Conv2",
    "LightConv",
    "RepConv",
    "DWConv",
    "DWConvTranspose2d",
    "ConvTranspose",
    "Focus",
    "GhostConv",
    "ChannelAttention",
    "SpatialAttention",
    "CBAM",
    "Concat",
    "TransformerLayer",
    "TransformerBlock",
    "MLPBlock",
    "LayerNorm2d",
    "DFL",
    "HGBlock",
    "HGStem",
    "SPP",
    "SPPF",
    "C1",
    "C2",
    "C3",
    "C2f",
    "C2fAttn",
    "C3x",
    "C3TR",
    "C3Ghost",
    "GhostBottleneck",
    "Bottleneck",
    "BottleneckCSP",
    "Proto",
    "Detect",
    "Segment",
    "Pose",
    "Classify",
    "TransformerEncoderLayer",
    "RepC3",
    "RTDETRDecoder",
    "AIFI",
    "DeformableTransformerDecoder",
    "DeformableTransformerDecoderLayer",
    "MSDeformAttn",
    "MLP",
    "ResNetLayer",
    "OBB",
    "WorldDetect",
    "ImagePoolingAttn",
    "ContrastiveHead",
    "BNContrastiveHead",
    "RepNCSPELAN4",
    "ADown",
    "SPPELAN",
    "CBFuse",
    "CBLinear",
)

