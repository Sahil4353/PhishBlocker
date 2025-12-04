import numpy as np
import torch
from typing import Dict, Any, Optional, Tuple
from scipy.sparse import csr_matrix
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

# Globals for Windows multiprocessing
_GLOBAL_X = None
_GLOBAL_Y = None


class IndexCSRDataset(Dataset):
    """
    Instead of returning dense vectors,
    return index -> collate builds dense batch
    """

    def __init__(self, X: csr_matrix, y: np.ndarray):
        self.X = X
        self.y = y.astype(np.int64)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> int:
        return int(idx)


def csr_worker_init(worker_id):
    global _GLOBAL_X, _GLOBAL_Y
    info = torch.utils.data.get_worker_info()
    if info is None:
        return
    ds = info.dataset
    _GLOBAL_X = ds.X
    _GLOBAL_Y = ds.y


def collate_csr(batch_idx_list):
    global _GLOBAL_X, _GLOBAL_Y
    idx = np.fromiter(batch_idx_list, dtype=np.int64)
    Xb = torch.from_numpy(_GLOBAL_X[idx].toarray().astype(np.float32))
    yb = torch.from_numpy(_GLOBAL_Y[idx])
    return Xb, yb


def make_loaders(
    Xtr: csr_matrix,
    ytr: np.ndarray,
    Xte: csr_matrix,
    yte: np.ndarray,
    num_classes: int,
    args,
    device
) -> Tuple[DataLoader, DataLoader]:
    pin = device.type == "cuda"
    workers = args.num_workers if args.num_workers >= 0 else 2

    train_ds = IndexCSRDataset(Xtr, ytr)
    test_ds = IndexCSRDataset(Xte, yte)

    common = dict(
        num_workers=workers,
        pin_memory=pin,
        persistent_workers=workers > 0,
        collate_fn=collate_csr,
        worker_init_fn=csr_worker_init
    )

    if args.weighted_sampler:
        binc = np.bincount(ytr, minlength=num_classes)
        w = 1.0 / np.clip(binc, 1, None)
        sample_w = w[ytr]
        sampler = WeightedRandomSampler(
            sample_w.tolist(), num_samples=len(ytr))
        tr = DataLoader(train_ds, batch_size=args.batch_size,
                        sampler=sampler, **common)
    else:
        tr = DataLoader(train_ds, batch_size=args.batch_size,
                        shuffle=True, **common)

    te = DataLoader(test_ds, batch_size=max(
        256, args.batch_size), shuffle=False, **common)

    return tr, te
