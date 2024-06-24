import os
import pickle
from contextlib import nullcontext
import torch
from gpt.model import GPT2Model
from jobs.configure import ExecutableJobs, GPT2Config


class EvaluateJob(ExecutableJobs):
    """
    Sample from a trained model
    """

    def __init__(self, job_name="evaluate-shakespeare"):
        super(EvaluateJob, self).__init__(job_name)
        self.data_type = job_name.split("-")[1]
        self.data_dir = self.temp_dir.joinpath(f"data/{self.data_type}")
        self.model_dir = self.temp_dir.joinpath(f"models/{self.data_type}")
        self.model_dir.mkdir(exist_ok=True, parents=True)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.context = (
            nullcontext()
            if not torch.cuda.is_available()
            else torch.amp.autocast(device_type="cuda", dtype=self.ptdtype)
        )

    def execute(self):
        torch.manual_seed(1337)
        torch.cuda.manual_seed(1337)
        torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
        torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn

        start = "\n"  # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
        # number of samples to draw
        num_samples = 10
        # number of tokens generated in each sample
        max_new_tokens = 500
        # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
        temperature = 0.8
        # retain only the top_k most likely tokens, clamp others to have 0 probability
        top_k = 200

        model = self.load_model()
        meta_path = self.data_dir.joinpath("meta.pkl")
        if os.path.exists(meta_path):
            print(f"Loading meta from {meta_path}...")
            with open(meta_path, "rb") as f:
                meta = pickle.load(f)
            stoi, itos = meta["stoi"], meta["itos"]
            encode = lambda s: [stoi[c] for c in s]
            decode = lambda l: "".join([itos[i] for i in l])

        # encode the beginning of the prompt
        if start.startswith("FILE:"):
            with open(start[5:], "r", encoding="utf-8") as f:
                start = f.read()
        start_ids = encode(start)
        x = torch.tensor(start_ids, dtype=torch.long, device=self.device)[None, ...]

        # run generation(model.generate has no_grad decorator, so omit it here)
        with self.context:
            for k in range(num_samples):
                y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
                print(decode(y[0].tolist()))
                print("---------------")

    def load_model(self):
        ckpt_path = os.path.join(self.model_dir, "ckpt.pt")
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        model_cfg = GPT2Config(**checkpoint["model_args"])
        model = GPT2Model(model_cfg)
        state_dict = checkpoint["model"]
        unwanted_prefix = "_orig_mod."
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        model.eval()
        model.to(self.device)
        model = torch.compile(model)
        return model

    @property
    def dtype(self) -> str:
        return "bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16"

    @property
    def ptdtype(self):
        return torch.bfloat16 if self.dtype == "bfloat16" else torch.float16
