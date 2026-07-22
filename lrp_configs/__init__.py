"""Named LRP/CRP configurations for ViT relevance propagation.

Each configuration is a complete, frozen recipe (layer_map + canonizers +
rules + hyperparameters) behind one ``build()`` call. They exist to answer the
research question of *which propagation choices help* — value-path-only vs full
bilinear attention, PA-LRP on the positional embedding, how the skip connection
splits relevance, ε vs γ on the linears — by varying one building block at a
time off a common reference (:mod:`lrp_configs.attnlrp_gamma`).

Usage::

    import lrp_configs
    cfg = lrp_configs.get("cp_lrp_baseline")     # the LXT value-path baseline
    composite = cfg.composite()                  # fresh zennit composite
    fv_dir = cfg.fv_path(root="data", model_tag="vit_small_dsprites")

    lrp_configs.names()        # every registered config
    lrp_configs.all_configs()  # name -> LRPConfig

Add a configuration by dropping a new module here that calls
``register(LRPConfig(...))`` and importing it below.

Current set (all isolate one knob off ``attnlrp_gamma`` unless noted):

==============================  =====================================
name                            isolates
==============================  =====================================
cp_lrp_baseline                 baseline — value-path only (StopGradient Q/K)
attnlrp_gamma                   reference — full bilinear + γ linears
attnlrp_epsilon                 linear rule (ε vs γ)
attnlrp_gamma_palrp             positional-embedding PA-LRP
attnlrp_gamma_residual_none     residual split (gradient default)
attnlrp_gamma_residual_symmetric residual split (50/50)
attnlrp_gamma_residual_l1        residual split (sign-preserving, research only)
attnlrp_alpha2beta1             attention αβ emphasis (2,−1)
==============================  =====================================
"""
from __future__ import annotations

from ._base import LRPConfig, register, get, names, all_configs, SITES

# import every configuration module so it self-registers on package import
from . import cp_lrp_baseline          # noqa: F401
from . import attnlrp_gamma            # noqa: F401
from . import attnlrp_epsilon          # noqa: F401
from . import attnlrp_gamma_palrp      # noqa: F401
from . import attnlrp_gamma_residual_none        # noqa: F401
from . import attnlrp_gamma_residual_symmetric   # noqa: F401
from . import attnlrp_gamma_residual_l1          # noqa: F401
from . import attnlrp_alpha2beta1      # noqa: F401

__all__ = ["LRPConfig", "register", "get", "names", "all_configs", "SITES"]
