"""Model-evaluation helpers shared by the experiments."""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch
from torch.utils.data import DataLoader, Subset


def select_correct(model, ds, classes: Sequence[int], n_per_class: int, device,
                   *, normalize=None, batch_size: int = 128, seed: int = 0,
                   max_scan: Optional[int] = None,
                   ) -> Dict[int, List[int]]:
    """Return ``{class: [dataset indices]}`` of up to ``n_per_class`` images per
    target class that the model classifies correctly. Scans the dataset in a
    fixed random order (so class-grouped datasets like dSprites are covered
    quickly) and stops once every target class is filled.

    ``max_scan`` caps how many images are examined: if a class is unfillable
    (e.g. a class the model never predicts correctly) the scan would
    otherwise crawl the *entire* dataset — for dSprites (737k images) that is
    ~20 min of CPU image decoding. With a cap the scan stops early and returns
    whatever filled (partial classes are expected and handled downstream).

    ``normalize`` (optional) is applied to each batch before the forward — pass
    it when the dataset yields un-normalized [0,1] images; leave ``None`` when the
    dataset transform already normalizes.
    """
    targets = set(classes)
    perm = torch.randperm(len(ds), generator=torch.Generator().manual_seed(seed)).tolist()
    loader = DataLoader(Subset(ds, perm), batch_size=batch_size, num_workers=0)
    sel: Dict[int, List[int]] = {c: [] for c in targets}
    pos = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            x = normalize(x) if normalize is not None else x
            pred = model(x).argmax(-1).cpu()
            for j in range(len(y)):
                c = int(y[j])
                if c in targets and pred[j] == c and len(sel[c]) < n_per_class:
                    sel[c].append(perm[pos + j])
            pos += len(y)
            if all(len(sel[c]) >= n_per_class for c in targets):
                break
            if max_scan is not None and pos >= max_scan:
                break
    return sel


__all__ = ["select_correct"]
