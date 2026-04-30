import torch
import numpy as np
from typing import List
import os
from pathlib import Path


def get_layer_names(model: torch.nn.Module, types: List):
    """
    Retrieves the layer names of all layers that belong to a torch.nn.Module type defined 
    in 'types'.

    Parameters
    ----------
    model: torch.nn.Module
    types: list of torch.nn.Module
        Layer types i.e. torch.nn.Conv2D

    Returns
    -------
    layer_names: list of strings


    """

    layer_names = []

    for name, layer in model.named_modules():
        for layer_definition in types:
            if isinstance(layer, layer_definition) or issubclass(layer.__class__, layer_definition):
                if name not in layer_names:
                    layer_names.append(name)

    return layer_names


def abs_norm(rel: torch.Tensor, stabilize=1e-10):
    """

    Parameter:
        rel: 1-D array
    """

    abs_sum = torch.sum(torch.abs(rel))

    return rel / (abs_sum + stabilize)

def max_norm(rel, stabilize=1e-10):
    
    return rel / (rel.max() + stabilize)


def get_output_shapes(model, single_sample: torch.tensor, record_layers: List[str]):
    """
    calculates the output shape of each layer using a forward pass.


    """

    output_shapes = {}

    def generate_hook(name):

        def shape_hook(module, input, output):
            output_shapes[name] = output.shape[1:]

        return shape_hook

    hooks = []
    for name, layer in model.named_modules():
        if name in record_layers:
            shape_hook = generate_hook(name)
            hooks.append(layer.register_forward_hook(shape_hook))

    _ = model(single_sample)

    [h.remove() for h in hooks]

    return output_shapes


_VARIANT_PREFIXES = ("RelMax_", "ActMax_", "RelStats_", "ActStats_")
_CONCEPT_MODES = ("head_", "token_")


def _legacy_path(path_folder):
    """
    Map a new-style variant folder (e.g. RelMax_head_sum_normed/) back to the
    legacy single-variant folder (RelMax_sum_normed/) for backward-compatible loads.
    Returns None if the folder name does not match the new layout.
    """
    p = Path(path_folder)
    parent = p.parent
    name = p.name
    for prefix in _VARIANT_PREFIXES:
        if name.startswith(prefix):
            tail = name[len(prefix):]
            for cm in _CONCEPT_MODES:
                if tail.startswith(cm):
                    return parent / Path(prefix + tail[len(cm):])
    return None


def _resolve_variant_folder(path_folder, *probe_files):
    """
    If `path_folder` lacks any of the probe files, fall back to the legacy folder
    (without the head_/token_ prefix). Returns the resolved Path.
    """
    p = Path(path_folder)
    if all((p / f).exists() for f in probe_files):
        return p
    legacy = _legacy_path(p)
    if legacy is not None and all((legacy / f).exists() for f in probe_files):
        return legacy
    return p


def load_maximization(path_folder, layer_name):

    filename = f"{layer_name}_"
    folder = _resolve_variant_folder(
        path_folder, filename + "data.npy", filename + "rel.npy", filename + "rf.npy")

    d_c_sorted = np.load(folder / Path(filename + "data.npy"), mmap_mode="r")
    rel_c_sorted = np.load(folder / Path(filename + "rel.npy"), mmap_mode="r")
    rf_c_sorted = np.load(folder / Path(filename + "rf.npy"), mmap_mode="r")

    return d_c_sorted, rel_c_sorted, rf_c_sorted

def load_stat_targets(path_folder):

    folder = _resolve_variant_folder(path_folder, "targets.npy")
    targets = np.load(folder / Path("targets.npy")).astype(np.int_)

    return targets


def load_statistics(path_folder, layer_name, target):

    filename = f"{target}_"
    folder = _resolve_variant_folder(
        path_folder,
        Path(layer_name) / Path(filename + "data.npy"),
        Path(layer_name) / Path(filename + "rel.npy"),
        Path(layer_name) / Path(filename + "rf.npy"),
    )

    d_c_sorted = np.load(folder / Path(layer_name) / Path(filename + "data.npy"), mmap_mode="r")
    rel_c_sorted = np.load(folder / Path(layer_name) / Path(filename + "rel.npy"), mmap_mode="r")
    rf_c_sorted = np.load(folder / Path(layer_name) / Path(filename + "rf.npy"), mmap_mode="r")

    return d_c_sorted, rel_c_sorted, rf_c_sorted


def load_receptive_field(path_folder, layer_name):

    filename = f"{layer_name}.npy"

    rf_array = np.load(Path(path_folder) / Path(filename), mmap_mode="r")

    return rf_array


def find_files(path=None):
    """
    Parameters:
        path: path analysis results

    """
    if path is None:
        path = os.getcwd()

    folders = os.listdir(path)

    r_max, a_max, r_stats, a_stats, rf = [], [], [], [], []
    for name in folders:
        found_path = str(Path(path) / Path(name))
        if "RelMax" in name:
            r_max.append(found_path)
        elif "ActMax" in name:
            a_max.append(found_path)
        elif "RelStats" in name:
            r_stats.append(found_path)
        elif "ActStats" in name:
            a_stats.append(found_path)
        elif "ReField" in name:
            rf.append(found_path)

    return r_max, a_max, r_stats, a_stats, rf
