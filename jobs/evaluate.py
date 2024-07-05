import pickle
import torch
import accelerate
import torch.nn.functional as F
from gpt.model import GPT2Model
from jobs.configure import ExecutableJobs, load_config


class EvaluateJob(ExecutableJobs):
    """
    Sample from a trained model
    """

    def __init__(self, job_name="evaluate"):
        super(EvaluateJob, self).__init__(job_name)
        self.data_dir = self.temp_dir.joinpath("data")
        self.model_dir = self.temp_dir.joinpath("models")
        self.num_samples = 10  # number of samples to draw
        self.max_new_tokens = 500  # number of tokens generated in each sample
        self.temperature = 0.8  # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
        self.top_k = 200  # retain only the top_k most likely tokens, clamp others to have 0 probability

        meta_path = self.data_dir.joinpath("meta.pkl")
        if meta_path.exists():
            with open(meta_path, "rb") as f:
                meta = pickle.load(f)
            self.stoi = meta.pop("stoi")
            self.itos = meta.pop("itos")
            self.model_cfg = load_config(self.project_dir, "model")
            self.model_cfg.vocab_size = len(self.itos)

    def execute(self):
        torch.manual_seed(1337)

        accelerator = accelerate.Accelerator()
        model = GPT2Model(self.model_cfg)
        accelerate.load_checkpoint_in_model(model, str(self.model_dir))
        model = accelerator.prepare(model)
        model.eval()

        with torch.no_grad():
            for k in range(self.num_samples):
                start_ids = self.encode("\n")  # begin sentence with \n
                input_ids = torch.tensor(start_ids, dtype=torch.long, device="cuda:0")[None, ...]

                for _ in range(self.max_new_tokens):
                    # if the sequence context is growing too long we must crop it at block_size
                    cropped_input_ids = (
                        input_ids
                        if input_ids.size(1) <= self.model_cfg.block_size
                        else input_ids[:, -self.model_cfg.block_size:]
                    )
                    # forward the model to get the logits for the index in the sequence
                    logits, _ = model(cropped_input_ids)
                    # pluck the logits at the final step and scale by desired temperature
                    logits = logits[:, -1, :] / self.temperature
                    # crop the logits to only the top k options
                    logit_lb = torch.topk(logits, min(self.top_k, logits.size(-1))).values
                    logits[logits < logit_lb[:, [-1]]] = -float("inf")
                    probs = F.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                    input_ids = torch.cat((input_ids, next_token), dim=1)

                print(self.decode(input_ids[0].tolist()))
                print("---------------")

    def encode(self, string: str) -> list[int]:
        return [self.stoi[char] for char in string]

    def decode(self, indices: list[int]) -> str:
        return "".join([self.itos[idx] for idx in indices])
