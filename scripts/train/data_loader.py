import numpy as np
import torch
from typing import Tuple
from scipy.sparse import csr_matrix
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

# Globals for worker sharing
_GLOBAL_X = None
_GLOBAL_Y = None


class IndexCSRDataset(Dataset):
    """Return only row index -> let collate handle the conversion."""

    def __init__(self, X: csr_matrix, y: np.ndarray):
        self.X = X
        self.y = y.astype(np.int64)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> int:
        return idx


def csr_worker_init(worker_id):
    """Share X and Y inside worker memory space."""
    global _GLOBAL_X, _GLOBAL_Y
    info = torch.utils.data.get_worker_info()
    if info is None:
        return
    ds = info.dataset
    _GLOBAL_X = ds.X
    _GLOBAL_Y = ds.y


def collate_csr(batch_indices):
    """Convert CSR → pinned float32 tensor."""
    global _GLOBAL_X, _GLOBAL_Y

    idx = np.asarray(batch_indices, dtype=np.int64)

    # convert slice (very fast)
    arr = _GLOBAL_X[idx].toarray().astype(np.float32)

    xb = torch.from_numpy(arr)
    yb = torch.from_numpy(_GLOBAL_Y[idx])

    # Pinned memory = optimal for GPU transfer
    if xb.is_floating_point():
        xb = xb.pin_memory()

    return xb, yb


def make_loaders(
    Xtr: csr_matrix,
    ytr: np.ndarray,
    Xte: csr_matrix,
    yte: np.ndarray,
    num_classes: int,
    args,
    device
) -> Tuple[DataLoader, DataLoader]:

    pin = (device.type == "cuda")
    workers = max(0, args.num_workers)

    train_ds = IndexCSRDataset(Xtr, ytr)
    test_ds = IndexCSRDataset(Xte, yte)

    common = dict(
        num_workers=workers,
        pin_memory=pin,
        persistent_workers=(workers > 0),
        worker_init_fn=csr_worker_init,
        collate_fn=collate_csr,
    )

    # Weighted sampler
    if args.weighted_sampler:
        binc = np.bincount(ytr, minlength=num_classes)
        w = 1 / np.clip(binc, 1, None)
        sample_w = w[ytr]
        sampler = WeightedRandomSampler(sample_w, len(ytr))
        tr = DataLoader(train_ds, batch_size=args.batch_size,
                        sampler=sampler, **common)
    else:
        tr = DataLoader(train_ds, batch_size=args.batch_size,
                        shuffle=True, **common)

    # Test batches are bigger (fast inference)
    te = DataLoader(test_ds, batch_size=max(
        256, args.batch_size), shuffle=False, **common)

    return tr, te
