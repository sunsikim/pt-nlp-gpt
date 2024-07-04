import numpy as np
import pathlib
import torch


class GPT2Dataset(torch.utils.data.Dataset):

    def __init__(self, data_dir: pathlib.Path, block_size: int, split: str):
        if split not in ("train", "val"):
            raise ValueError("split must be one of 'train', 'val'")
        self.data_path = data_dir.joinpath(f"{split}.bin")
        self.data = np.memmap(self.data_path, np.uint16, "r")
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        """
        (1) We recreate np.memmap every batch to avoid a memory leak, as per
        https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
        (2) In original nanoGPT code, x, y are pinned to move them to GPU asynchronously (non_blocking=True)
        However, this will be handled by 🤗 accelerate, so removed.
        """
        idx_ub = idx + self.block_size
        x = torch.from_numpy((self.data[idx: idx_ub]).astype(np.int64))
        y = torch.from_numpy((self.data[idx + 1: idx_ub + 1]).astype(np.int64))
        return x, y
