import pathlib
import torch
import math
import time
import logging
import inspect
from gpt.data import BatchLoader
from gpt.model import GPT2Model
from jobs.configure import TrainerConfig
from contextlib import nullcontext


class GPT2Trainer:

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        batch_loader: BatchLoader,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
        trainer_cfg: TrainerConfig,
        model_dir: pathlib.Path,
        model_args: dict,
    ):
        self.trainer_cfg = trainer_cfg
        self.batch_loader = batch_loader
        self.model_dir = model_dir
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.max_lr = optimizer.param_groups[0]['lr']
        self.min_lr = self.max_lr / 10
        self.iter_num = 0
        self.best_val_loss = 1e9
        self.model_args = model_args
        self.context = (
            nullcontext()
            if not torch.cuda.is_available()
            else torch.amp.autocast(device_type="cuda", dtype=self.ptdtype)
        )

    def train(self, model):
        # initialize a GradScaler. If enabled=False scaler is a no-op
        scaler = torch.cuda.amp.GradScaler(enabled=(self.dtype == "float16"))
        X, Y = self.batch_loader.get_batch("train")  # fetch the very first batch
        t0 = time.time()
        local_iter_num = 0  # number of iterations in the lifetime of this process
        while True:

            # evaluate the loss on train/val sets and write checkpoints
            if self.iter_num % self.trainer_cfg.eval_interval == 0:
                losses = self.estimate_loss(model)
                msg = f"step {self.iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
                logging.info(msg)
                if losses["val"] < self.best_val_loss:
                    best_val_loss = losses["val"]
                    if self.iter_num > 0:
                        checkpoint = {
                            "model": model.state_dict(),
                            "optimizer": self.optimizer.state_dict(),
                            "model_args": self.model_args,
                            "iter_num": self.iter_num,
                            "best_val_loss": best_val_loss,
                        }
                        logging.info(f"saving checkpoint to {self.model_dir}")
                        torch.save(checkpoint, self.model_dir.joinpath("ckpt.pt"))

            # forward backward update, with optional gradient accumulation to simulate larger batch size
            # and using the GradScaler if data type is float16
            for micro_step in range(self.trainer_cfg.gradient_accumulation_steps):
                with self.context:
                    logits, loss = model(X, Y)
                    # scale the loss to account for gradient accumulation
                    loss /= self.trainer_cfg.gradient_accumulation_steps
                # immediately async prefetch next batch while model is doing the forward pass on the GPU
                X, Y = self.batch_loader.get_batch("train")
                # backward pass, with gradient scaling if training in fp16
                scaler.scale(loss).backward()
            # clip the gradient
            if self.trainer_cfg.grad_clip != 0.0:
                scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.trainer_cfg.grad_clip)
            # step the optimizer and scaler if training in fp16
            scaler.step(self.optimizer)
            scaler.update()
            # flush the gradients as soon as we can, no need for this memory anymore
            self.optimizer.zero_grad(set_to_none=True)
            self.lr_scheduler.step()

            # timing and logging
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            if self.iter_num % self.trainer_cfg.log_interval == 0:
                # get loss as float. note: this is a CPU-GPU sync point
                # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
                lossf = loss.item() * self.trainer_cfg.gradient_accumulation_steps
                logging.info(f"iter {self.iter_num}: loss {lossf:.4f}, time {dt * 1000:.2f}ms")
            self.iter_num += 1
            local_iter_num += 1

            # termination conditions
            if self.iter_num > self.trainer_cfg.max_iters:
                break

    @torch.no_grad()
    def estimate_loss(self, model):
        """
        helps estimate an arbitrarily accurate loss over either split using many batches
        """
        out = {}
        model.eval()
        for split in ["train", "val"]:
            losses = torch.zeros(self.trainer_cfg.eval_iters)
            for k in range(self.trainer_cfg.eval_iters):
                X, Y = self.batch_loader.get_batch(split)
                with self.context:
                    logits, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out

    @property
    def dtype(self) -> str:
        return "bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16"

    @property
    def ptdtype(self):
        return torch.bfloat16 if self.dtype == "bfloat16" else torch.float16


class LinearWarmupCosineDecay(torch.optim.lr_scheduler.LRScheduler):

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        learning_rate: float,
        warmup_steps: int,
        max_steps: int,
    ):
        self.max_lr = learning_rate
        self.min_lr = learning_rate / 10
        self.max_steps = max_steps
        self.warmup_steps = warmup_steps
        super(LinearWarmupCosineDecay, self).__init__(optimizer, last_epoch=-1)

    def get_lr(self) -> list[float]:
        """
        learning rate decay scheduler (cosine with warmup)
        """
        if self._step_count < self.warmup_steps:
            # 1) linear warmup for warmup_iters steps
            lr = self.max_lr * self._step_count / self.warmup_steps
        else:
            # 2) in between, use cosine decay down to min learning rate
            decay_ratio = (self._step_count - self.warmup_steps) / (self.max_steps - self.warmup_steps)
            assert 0 <= decay_ratio <= 1
            coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
            lr = self.min_lr + coeff * (self.max_lr - self.min_lr)
        # return list of learning rates to be applied on every parameter group
        return [lr] * len(self.optimizer.param_groups)


def define_optimizer(
    model: GPT2Model,
    weight_decay: float,
    learning_rate: float,
    beta1: float,
    beta2: float,
) -> torch.optim.Optimizer:
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
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]

    # Create AdamW optimizer and use the fused version if it is available
    fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
    use_fused = True if fused_available and torch.cuda.is_available() else None
    optimizer = torch.optim.AdamW(
        optim_groups,
        lr=learning_rate,
        betas=(beta1, beta2),
        fused=use_fused,
    )
    return optimizer
