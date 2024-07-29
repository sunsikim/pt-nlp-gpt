import logging
import pathlib
from dataclasses import dataclass

logging.basicConfig(
    format="%(asctime)s (%(module)s) (%(funcName)s) : %(msg)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)


class ExecutableJobs:

    def __init__(self, job_name: str = None):
        self.name = job_name
        job_dir = pathlib.Path(__file__).parent
        self.project_dir = job_dir.parent
        self.temp_dir = self.project_dir.joinpath("tmp")
        self.temp_dir.mkdir(exist_ok=True, parents=True)

    def execute(self):
        raise NotImplementedError



@dataclass
class GPT2Config:
    """
    Configuration class for GPT2 Model
    :param block_size: size of context window
    :param vocab_size: GPT-2 defaults to 50257, but rounded up to 50304 for efficiency
    :param n_layer: number of transformer layers
    :param n_head: number of transformer heads
    :param n_embd: number of embedding dimension
    :param dropout: dropout rate(for pretraining 0 is good, for finetuning try 0.1+)
    :param bias: whether we should use bias inside LayerNorm and Linear layers
    """

    block_size: int = 256
    vocab_size: int = 50304
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.2
    bias: bool = False


@dataclass
class OptimizerConfig:
    """
    :param weight_decay: weight decay rate
    :param learning_rate: maximum learning rate
    :param beta1: AdamW optimizer parameter
    :param beta2: AdamW optimizer parameter
    """

    weight_decay: float = 1e-1
    learning_rate: float = 1e-3
    beta1: float = 0.9
    beta2: float = 0.99


@dataclass
class TrainerConfig:
    """
    :param batch_size: batch size
    :param grad_clip: clip gradients at this value, or disable if == 0.0
    :param gradient_accumulation_steps: used to simulate larger batch sizes
    :param max_iters: total number of training iterations
    :param eval_interval: number of iterations between evaluations
    :param eval_iters: number of steps used for evaluation
    :param log_interval: number of iterations between logging
    :param max_iters: total number of training iterations
    :param compile: whether to compile the model to be faster
    :param warmup_iters: number of warmup iterations
    :param always_save_checkpoint: if False, save checkpoint only when validation loss improves
    :param wandb_log: whether to log to wandb(disabled by default)
    :param wandb_project: project name for wandb
    :param wandb_run_name: wandb run name
    """

    batch_size: int = 64
    grad_clip: float = 1.0
    gradient_accumulation_steps: int = 1
    eval_interval: int = 250
    eval_iters: int = 200
    log_interval: int = 10
    max_iters: int = 5_000
    compile: bool = True
    warmup_iters: int = 100
    always_save_checkpoint: bool = False
    wandb_log: bool = False
    wandb_project: str = "owt"
    wandb_run_name: str = "gpt2"


def load_config(config_type: str):
    if config_type == "model":
        return GPT2Config()
    elif config_type == "optimizer":
        return OptimizerConfig()
    elif config_type == "trainer":
        return TrainerConfig()
    else:
        raise ValueError(f"Unknown config type: {config_type}")
