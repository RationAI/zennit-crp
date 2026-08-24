"""AttnLRP rule hooks — one module per source paper."""
from zennit_extensions.rules.attnlrp import (
    EpsilonAdd,
    EpsilonAddGradTimesInput,
    GammaGradInput,
    GradTimesInputMultiInputBasicHook,
    IdentityGradTimesInput,
    LayerNormEpsilon,
    MatmulAttnLRP,
    MatmulAttnLRPGradTimesInput,
    SoftmaxAttnLRP,
    TorchvisionEncoderBlockCanonizer,
)
from zennit_extensions.rules.bajger_contrib import AlphaBetaMatmul, ResidualL1
from zennit_extensions.rules.chefer2021 import (
    CheferAdd,
    CheferMatmul,
    safe_divide,
)
from zennit_extensions.rules.palrp import PosEmbedSink, RotaryRopeSink
from zennit_extensions.rules.residuals_otsuki2024 import ResidualRatio

__all__ = [
    "SoftmaxAttnLRP", "MatmulAttnLRP", "EpsilonAdd", "LayerNormEpsilon",
    "GradTimesInputMultiInputBasicHook",
    "GammaGradInput", "EpsilonAddGradTimesInput", "MatmulAttnLRPGradTimesInput",
    "IdentityGradTimesInput",
    "TorchvisionEncoderBlockCanonizer",
    "ResidualRatio", "CheferAdd", "CheferMatmul",
    "safe_divide",
    "AlphaBetaMatmul", "ResidualL1",
    "PosEmbedSink", "RotaryRopeSink",
]
