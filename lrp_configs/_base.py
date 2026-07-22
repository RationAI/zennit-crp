"""Registry plumbing for named LRP/CRP configurations.

A *configuration* is a complete, frozen recipe for propagating relevance
through a ViT: the layer_map + canonizers + rules + hyperparameters, bundled
behind one ``build()`` call that returns a ready zennit ``Composite``. Each
configuration lives in its own module under :mod:`lrp_configs` and registers
itself here, so notebooks and experiments can switch recipes by name and keep
their artefacts (FV indices, saliency maps, flipping parquets) from clashing.

Why a registry rather than ad-hoc composite construction: the research
question (paper §Concept-flipping / §Variants) is *which propagation choices
help* — value-path-only vs full bilinear attention, PA-LRP on the positional
embedding, how the skip connection splits relevance, ε vs γ on linears. Each
of those is one named config, varied one knob at a time off a common
reference, so the experiment can rank the building blocks rather than guess.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict

from zennit.core import Composite

# concept probe sites exposed by the attention-substitution canonizers
# (see AGENTS.md §3): the block's post-projection output, or the Q/K/V token
# sequences just after the qkv split.
SITES = ("proj_drop", "q_lrp_probe", "k_lrp_probe", "v_lrp_probe")


@dataclass(frozen=True)
class LRPConfig:
    """One named relevance-propagation recipe.

    Attributes
    ----------
    name : str
        Unique slug. Doubles as the on-disk namespace for every artefact
        derived from this config (FV index dir, flipping parquet suffix,
        saliency cache), so two configs never overwrite each other.
    description : str
        One line: which building blocks this recipe includes, stated against
        the common reference so the ablation it isolates is readable.
    build : Callable[[], Composite]
        Constructs a fresh composite. Called once per attribution context;
        never share a composite across ``composite.context(model)`` scopes.
    site : str
        Default concept probe site for this recipe (one of :data:`SITES`).
    isolates : str
        The single building block this config varies relative to the
        reference (``attnlrp_gamma``) — used to group results in the
        variant-ranking table. ``""`` for the reference / baseline itself.
    """

    name: str
    description: str
    build: Callable[[], Composite]
    site: str = "proj_drop"
    isolates: str = ""

    def composite(self) -> Composite:
        """Return a fresh composite for this configuration."""
        return self.build()

    def fv_path(self, root, model_tag: str) -> Path:
        """Per-(config, model) directory for FeatureVisualization indices, so
        indices computed under different recipes or models never collide:
        ``<root>/fv/<model_tag>/<config name>/``."""
        return Path(root) / "fv" / model_tag / self.name


_REGISTRY: Dict[str, LRPConfig] = {}


def register(config: LRPConfig) -> LRPConfig:
    """Add ``config`` to the global registry (rejecting duplicate names) and
    return it, so a module can do ``CONFIG = register(LRPConfig(...))``."""
    if config.name in _REGISTRY:
        raise ValueError(f"duplicate LRP config name {config.name!r}")
    _REGISTRY[config.name] = config
    return config


def get(name: str) -> LRPConfig:
    """Look up a registered configuration by name."""
    if name not in _REGISTRY:
        raise KeyError(f"unknown LRP config {name!r}; available: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def names() -> list:
    """All registered configuration names, sorted."""
    return sorted(_REGISTRY)


def all_configs() -> Dict[str, LRPConfig]:
    """Mapping of name → config for every registered configuration."""
    return dict(_REGISTRY)
