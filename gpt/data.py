import numpy as np
import pathlib
import torch


class BatchLoader:

    def __init__(
        self,
        data_dir: pathlib.Path,
        block_size: int,
        batch_size: int,
        device: str,
    ):
        self.data_dir = data_dir
        self.block_size = block_size
        self.batch_size = batch_size
        self.device = device

    def get_batch(self, split: str):
        # We recreate np.memmap every batch to avoid a memory leak, as per
        # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
        if split not in ("train", "val"):
            raise ValueError("split must be one of 'train', 'val'")
        data = np.memmap(
            filename=self.data_dir.joinpath(f"{split}.bin"),
            dtype=np.uint16,
            mode="r",
        )

        ix = torch.randint(len(data) - self.block_size, (self.batch_size,))
        x = torch.stack(
            [
                torch.from_numpy((data[i: i + self.block_size]).astype(np.int64))
                for i in ix
            ]
        )
        y = torch.stack(
            [
                torch.from_numpy(
                    (data[i + 1: i + 1 + self.block_size]).astype(np.int64)
                )
                for i in ix
            ]
        )
        if self.device.startswith("cuda"):
            # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
            x, y = x.pin_memory().to(
                self.device, non_blocking=True
            ), y.pin_memory().to(self.device, non_blocking=True)
        else:
            x, y = x.to(self.device), y.to(self.device)
        return x, y
