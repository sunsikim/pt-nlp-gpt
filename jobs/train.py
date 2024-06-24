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
from gpt.trainer import GPT2Trainer
from jobs.configure import ExecutableJobs, load_config


class TrainJob(ExecutableJobs):
    """
    This training script can be run both on a single gpu in debug mode,
    and also in a larger training run with distributed data parallel (ddp).
    """

    def __init__(self, job_name="train-shakespeare"):
        super().__init__(job_name)
        self.data_type = job_name.split("-")[1]
        self.data_dir = self.temp_dir.joinpath(f"data/{self.data_type}")
        self.model_dir = self.temp_dir.joinpath(f"models/{self.data_type}")
        self.model_dir.mkdir(exist_ok=True, parents=True)
        self.model_cfg = load_config(self.project_dir, self.data_type, "model")
        self.optimizer_cfg = load_config(self.project_dir, self.data_type, "optimizer")
        self.trainer_cfg = load_config(self.project_dir, self.data_type, "trainer")

    def execute(self):
        # various inits, derived attributes, I/O setup
        ddp = int(os.environ.get("RANK", -1)) != -1  # is this a ddp run?
        if ddp:
            init_process_group(backend="nccl")
            ddp_rank = int(os.environ["RANK"])
            ddp_local_rank = int(os.environ["LOCAL_RANK"])
            ddp_world_size = int(os.environ["WORLD_SIZE"])
            torch.cuda.set_device(self.device)
            seed_offset = ddp_rank  # each process gets a different seed
            # world_size number of processes will be training simultaneously, so we can scale
            # down the desired gradient accumulation iterations per process proportionally
            assert self.trainer_cfg.gradient_accumulation_steps % ddp_world_size == 0
            self.trainer_cfg.gradient_accumulation_steps //= ddp_world_size
        else:
            # if not ddp, we are running on a single gpu, and one process
            seed_offset = 0
            ddp_world_size = 1

        torch.manual_seed(1337 + seed_offset)
        torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
        torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn

        tokens_per_iter = (
            ddp_world_size
            * self.trainer_cfg.gradient_accumulation_steps
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
        optimizer = self.define_optimizer(model)

        logging.info("compiling the model... (takes a ~minute)")
        model = torch.compile(model)  # requires PyTorch 2.0

        # wrap model into DDP container
        if ddp:
            model = DDP(model, device_ids=[ddp_local_rank])

        logging.info("start training")
        trainer = GPT2Trainer(
            optimizer=optimizer,
            batch_loader=batch_loader,
            trainer_cfg=self.trainer_cfg,
            model_dir=self.model_dir,
            model_args=model_args,
        )
        trainer.train(model)

        if ddp:
            destroy_process_group()

    def define_optimizer(self, model: GPT2Model) -> torch.optim.Optimizer:
        # optimizer
        # start with all candidate parameters
        param_dict = {pn: p for pn, p in model.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": self.optimizer_cfg.weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]

        # Create AdamW optimizer and use the fused version if it is available
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = True if fused_available and self.device_type == "cuda" else None
        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=self.optimizer_cfg.learning_rate,
            betas=(self.optimizer_cfg.beta1, self.optimizer_cfg.beta2),
            fused=use_fused,
        )
        return optimizer

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
