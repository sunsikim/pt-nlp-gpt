import os
import pickle
import requests
import numpy as np
from jobs.configure import ExecutableJobs


class PreprocessJob(ExecutableJobs):
    """
    Prepare the Shakespeare dataset for character-level language modeling.
    So instead of encoding with GPT-2 BPE tokens, we just map characters to ints.
    Will save train.bin, val.bin containing the ids, and meta.pkl containing the
    encoder and decoder and some other related info.
    """

    def __init__(self, job_name="preprocess"):
        super().__init__(job_name=job_name)
        self._data_dir = self.temp_dir.joinpath("data")
        self._data_dir.mkdir(exist_ok=True, parents=True)

    def execute(self):
        """
        resulting dataset would have:
        # length of dataset in characters:  1115394
        # all the unique characters:
        #  !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz
        # vocab size: 65
        # train has 1003854 tokens
        # val has 111540 tokens
        """
        # download the tiny shakespeare dataset
        input_file_path = self._data_dir.joinpath("input.txt")
        if not os.path.exists(input_file_path):
            data_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
            with open(input_file_path, "w") as f:
                f.write(requests.get(data_url).text)

        with open(input_file_path, "r") as f:
            data = f.read()
        print(f"length of dataset in characters: {len(data):,}")

        # get all the unique characters that occur in this text
        chars = sorted(list(set(data)))
        vocab_size = len(chars)
        print("all the unique characters:", "".join(chars))
        print(f"vocab size: {vocab_size:,}")

        # create a mapping from characters to integers
        stoi = {ch: i for i, ch in enumerate(chars)}
        itos = {i: ch for i, ch in enumerate(chars)}

        def encode(s: str) -> list[int]:
            """
            encoder: take a string, output a list of integers
            :param s: string
            :return: list of integers
            """
            return [stoi[c] for c in s]

        # create the train and test splits
        n = len(data)
        train_data = data[: int(n * 0.9)]
        val_data = data[int(n * 0.9) :]

        # encode both to integers
        train_ids = encode(train_data)
        val_ids = encode(val_data)
        print(f"train has {len(train_ids):,} tokens")
        print(f"val has {len(val_ids):,} tokens")

        # export to bin files
        train_ids = np.array(train_ids, dtype=np.uint16)
        val_ids = np.array(val_ids, dtype=np.uint16)
        train_ids.tofile(self._data_dir.joinpath("train.bin"))
        val_ids.tofile(self._data_dir.joinpath("val.bin"))

        # save the meta information as well, to help us encode/decode later
        meta = {
            "vocab_size": vocab_size,
            "itos": itos,
            "stoi": stoi,
        }
        with open(self._data_dir.joinpath("meta.pkl"), "wb") as f:
            pickle.dump(meta, f)
