"""Named LRP/CRP configurations for ViT relevance propagation.

Each configuration is a complete, frozen recipe (layer_map + canonizers +
rules + hyperparameters) behind one ``build()`` call. Two recipes are carried:
``cp_lrp_baseline`` (the default, value-path-only CP-LRP) and
``attnlrp_baseline`` (AttnLRP exactly as published). The research question is
whether the value-path-only baseline surpasses faithful full-bilinear AttnLRP.

Usage::

    import lrp_configs
    cfg = lrp_configs.get("cp_lrp_baseline")     # the LXT value-path baseline
    composite = cfg.composite()                  # fresh zennit composite
    fv_dir = cfg.fv_path(root="data", model_tag="vit_small_dsprites")

    lrp_configs.names()        # every registered config
    lrp_configs.all_configs()  # name -> LRPConfig

Add a configuration by dropping a new module here that calls
``register(LRPConfig(...))`` and importing it below.

Current set:

==============================  =====================================
name                            recipe
==============================  =====================================
cp_lrp_baseline                 default — value-path only (StopGradient Q/K)
attnlrp_baseline                AttnLRP as published (Eq. 15 matmul + Prop. 3.1)
chefer_lrp                      Chefer CVPR'21 LRP — engine for the Chefer
                                benchmark row (not an LRP recipe under comparison)
==============================  =====================================
"""
from __future__ import annotations

from ._base import LRPConfig, register, get, names, all_configs, SITES

# import every configuration module so it self-registers on package import
from . import cp_lrp_baseline          # noqa: F401
from . import attnlrp_baseline         # noqa: F401
from . import chefer_lrp               # noqa: F401

__all__ = ["LRPConfig", "register", "get", "names", "all_configs", "SITES"]
