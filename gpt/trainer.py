import pathlib
import torch
import math
import time
import logging
import os
import wandb
from gpt.data import BatchLoader
from jobs.configure import TrainerConfig
from contextlib import nullcontext


class GPT2Trainer:

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        batch_loader: BatchLoader,
        trainer_cfg: TrainerConfig,
        model_dir: pathlib.Path,
        model_args: dict,
    ):
        self.trainer_cfg = trainer_cfg
        self.batch_loader = batch_loader
        self.model_dir = model_dir
        self.optimizer = optimizer
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
        ddp_rank = int(os.environ.get("RANK", -1))
        self.ddp = ddp_rank != -1
        self.master_process = (not self.ddp) or (self.ddp and ddp_rank == 0)
        if self.trainer_cfg.wandb_log and self.master_process:
            wandb.init(
                project=self.trainer_cfg.wandb_project,
                name=self.trainer_cfg.wandb_run_name,

            )

    def train(self, model):
        # initialize a GradScaler. If enabled=False scaler is a no-op
        scaler = torch.cuda.amp.GradScaler(enabled=(self.dtype == "float16"))
        checkpoint = None  # free up memory
        X, Y = self.batch_loader.get_batch("train")  # fetch the very first batch
        t0 = time.time()
        local_iter_num = 0  # number of iterations in the lifetime of this process
        raw_model = model.module if self.ddp else model  # unwrap DDP container if needed
        while True:

            # determine and set the learning rate for this iteration
            lr = self.get_lr(self.iter_num)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr

            # evaluate the loss on train/val sets and write checkpoints
            if self.iter_num % self.trainer_cfg.eval_interval == 0 and self.master_process:
                losses = self.estimate_loss(model)
                msg = f"step {self.iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
                logging.info(msg)
                if self.trainer_cfg.wandb_log:
                    wandb.log(
                        {
                            "iter": self.iter_num,
                            "train/loss": losses["train"],
                            "val/loss": losses["val"],
                            "lr": lr,
                        }
                    )
                if losses["val"] < self.best_val_loss:
                    best_val_loss = losses["val"]
                    if self.iter_num > 0:
                        checkpoint = {
                            "model": raw_model.state_dict(),
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
                if self.ddp:
                    # in DDP training we only need to sync gradients at the last micro step.
                    # the official way to do this is with model.no_sync() context manager, but
                    # I really dislike that this bloats the code and forces us to repeat code
                    # looking at the source of that context manager, it just toggles this variable
                    model.require_backward_grad_sync = (
                            micro_step == self.trainer_cfg.gradient_accumulation_steps - 1
                    )
                with self.context:
                    logits, loss = model(X, Y)
                    loss = (
                            loss / self.trainer_cfg.gradient_accumulation_steps
                    )  # scale the loss to account for gradient accumulation
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

            # timing and logging
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            if self.iter_num % self.trainer_cfg.log_interval == 0 and self.master_process:
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

    def get_lr(self, it: int) -> float:
        """
        learning rate decay scheduler (cosine with warmup)
        """
        # 1) linear warmup for warmup_iters steps
        if it < self.trainer_cfg.warmup_iters:
            return self.max_lr * it / self.trainer_cfg.warmup_iters
        # 2) in between, use cosine decay down to min learning rate
        decay_ratio = (it - self.trainer_cfg.warmup_iters) / (
                self.trainer_cfg.max_iters - self.trainer_cfg.warmup_iters
        )
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
        return self.min_lr + coeff * (self.max_lr - self.min_lr)

    @property
    def dtype(self) -> str:
        return "bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16"

    @property
    def ptdtype(self):
        return torch.bfloat16 if self.dtype == "bfloat16" else torch.float16

