import inspect
import os
import pickle
import logging
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from dataclasses import asdict
from gpt.model import GPT2Model
from gpt.data import BatchLoader
from gpt.trainer import GPT2Trainer, LinearWarmupCosineDecay, define_optimizer
from jobs.configure import ExecutableJobs, load_config


class TrainJob(ExecutableJobs):
    """
    This training script can be run both on a single gpu in debug mode,
    and also in a larger training run with distributed data parallel (ddp).
    """

    def __init__(self, job_name="train"):
        super().__init__(job_name)
        self.data_dir = self.temp_dir.joinpath("data")
        self.model_dir = self.temp_dir.joinpath("models")
        self.model_dir.mkdir(exist_ok=True, parents=True)
        self.model_cfg = load_config(self.project_dir, "model")
        self.optimizer_cfg = load_config(self.project_dir, "optimizer")
        self.trainer_cfg = load_config(self.project_dir, "trainer")

    def execute(self):
        torch.manual_seed(1337)
        torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
        torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn

        tokens_per_iter = (
            self.trainer_cfg.gradient_accumulation_steps
            * self.trainer_cfg.batch_size
            * self.model_cfg.block_size
        )
        logging.info(f"tokens per iteration will be: {tokens_per_iter:,}")

        logging.info("Define batch loader")
        batch_loader = BatchLoader(
            data_dir=self.data_dir,
            block_size=self.model_cfg.block_size,
            batch_size=self.trainer_cfg.batch_size,
            device=self.device
        )

        logging.info("Deriving vocab_size from the dataset")
        meta_path = os.path.join(self.data_dir, "meta.pkl")
        if os.path.exists(meta_path):
            with open(meta_path, "rb") as f:
                meta = pickle.load(f)
            meta_vocab_size = meta["vocab_size"]
            print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")
            self.model_cfg.vocab_size = meta_vocab_size

        logging.info("Initialize a new model from scratch")
        model_args = asdict(self.model_cfg)
        model = GPT2Model(self.model_cfg)
        model.to(self.device)

        logging.info("Define optimizer")
        optimizer = define_optimizer(
            model=model,
            weight_decay=self.optimizer_cfg.weight_decay,
            learning_rate=self.optimizer_cfg.learning_rate,
            beta1=self.optimizer_cfg.beta1,
            beta2=self.optimizer_cfg.beta2,
        )
        lr_scheduler = LinearWarmupCosineDecay(
            optimizer=optimizer,
            learning_rate=self.optimizer_cfg.learning_rate,
            warmup_steps=self.trainer_cfg.warmup_iters,
            max_steps=self.trainer_cfg.max_iters,
        )

        logging.info("compiling the model... (takes a ~minute)")
        model = torch.compile(model)  # requires PyTorch 2.0

        logging.info("start training")
        trainer = GPT2Trainer(
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            batch_loader=batch_loader,
            trainer_cfg=self.trainer_cfg,
            model_dir=self.model_dir,
            model_args=model_args,
        )
        trainer.train(model)

    @property
    def device(self) -> str:
        if torch.cuda.is_available():
            ddp_local_rank = os.environ.get("LOCAL_RANK", -1)
            return "cuda" if ddp_local_rank == -1 else f"cuda:{ddp_local_rank}"
        else:
            return "cpu"

    @property
    def device_type(self) -> str:
        # for later use in torch.autocast
        # note: float16 data type will automatically use a GradScaler
        return "cuda" if "cuda" in self.device else "cpu"
