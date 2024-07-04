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


class BatchLoader:

    def __init__(
        self,
        data_dir: pathlib.Path,
        block_size: int,
        batch_size: int,
        device: str,
    ):
        self.data_dir = data_dir
        self.train_dataset = GPT2Dataset(data_dir, block_size, "train")
        self.val_dataset = GPT2Dataset(data_dir, block_size, "val")
        self.block_size = block_size
        self.batch_size = batch_size
        self.device = device

    def get_batch(self, split: str):
        data = self.train_dataset if split == "train" else self.val_dataset
        ix = torch.randint(len(data) - self.block_size, (self.batch_size,))
        xs, ys = [], []
        for i in ix:
            x, y = data[i]
            xs.append(x)
            ys.append(y)
        x = torch.stack(xs)
        y = torch.stack(ys)
        if self.device.startswith("cuda"):
            # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
            x, y = x.pin_memory().to(
                self.device, non_blocking=True
            ), y.pin_memory().to(self.device, non_blocking=True)
        else:
            x, y = x.to(self.device), y.to(self.device)
        return x, y
