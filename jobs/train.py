import os
import pickle
import torch
import accelerate
import time
from gpt.model import GPT2Model
from gpt.data import GPT2Dataset
from accelerate.logging import get_logger
from gpt.trainer import LinearWarmupCosineDecay, define_optimizer
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
        self.log_dir = self.temp_dir.joinpath("logs")
        self.log_dir.mkdir(exist_ok=True, parents=True)
        self.model_cfg = load_config(self.project_dir, "model")
        self.optimizer_cfg = load_config(self.project_dir, "optimizer")
        self.trainer_cfg = load_config(self.project_dir, "trainer")
        self.best_val_loss = 1e9

        ckpt_dir = self.temp_dir.joinpath("checkpoints")
        if ckpt_dir.exists():
            os.system(f"rm -rf {ckpt_dir}")

    def execute(self):
        project_config = accelerate.utils.ProjectConfiguration(
            project_dir=str(self.temp_dir),
            logging_dir=str(self.log_dir),
            automatic_checkpoint_naming=True,
            total_limit=50,
        )
        accelerator = accelerate.Accelerator(
            project_config=project_config,
            log_with="tensorboard",
            gradient_accumulation_steps=self.trainer_cfg.gradient_accumulation_steps,
        )
        logger = get_logger(__name__, log_level="INFO")
        tokens_per_iter = (
            self.trainer_cfg.gradient_accumulation_steps
            * self.trainer_cfg.batch_size
            * self.model_cfg.block_size
        )
        logger.info(f"tokens per iteration will be: {tokens_per_iter:,}")

        torch.manual_seed(1337)
        logger.info("Deriving vocab_size from the dataset")
        meta_path = os.path.join(self.data_dir, "meta.pkl")
        if os.path.exists(meta_path):
            with open(meta_path, "rb") as f:
                meta = pickle.load(f)
            meta_vocab_size = meta["vocab_size"]
            print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")
            self.model_cfg.vocab_size = meta_vocab_size

        logger.info("Define train objects")
        model = GPT2Model(self.model_cfg)
        train_loader = torch.utils.data.DataLoader(
            dataset=GPT2Dataset(
                data_dir=self.data_dir,
                block_size=self.model_cfg.block_size,
                split="train",
            ),
            batch_size=self.trainer_cfg.batch_size,
            shuffle=True,
        )
        validation_loader = torch.utils.data.DataLoader(
            dataset=GPT2Dataset(
                data_dir=self.data_dir,
                block_size=self.model_cfg.block_size,
                split="val",
            ),
            batch_size=self.trainer_cfg.batch_size,
            shuffle=True,
        )
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
        model, train_loader, validation_loader, optimizer, scheduler = accelerator.prepare(
            model, train_loader, validation_loader, optimizer, lr_scheduler
        )

        logger.info("Start training", main_process_only=False, in_order=True)
        accelerator.init_trackers(f"nanoGPT-{int(time.time())}")
        tracker_idx = 0
        for batch_idx, train_batch in enumerate(train_loader, start=1):
            step_start = time.time()
            optimizer.zero_grad()
            inputs, targets = train_batch
            with accelerator.accumulate(model):
                _, loss = model(inputs, targets)
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    max_norm = self.trainer_cfg.grad_clip
                    accelerator.clip_grad_norm_(model.parameters(), max_norm)
                optimizer.step()
                scheduler.step()
            step_elapse = (time.time() - step_start) * 1000

            if batch_idx % self.trainer_cfg.log_interval == 0 and accelerator.is_main_process:
                loss_estimate = loss.item() * self.trainer_cfg.gradient_accumulation_steps
                logger.info(f"step {batch_idx}: loss {loss_estimate:.4f}, time {step_elapse:.2f}ms")

            if batch_idx % self.trainer_cfg.eval_interval == 0 and accelerator.is_main_process:
                model.eval()
                train_loss = self.evaluate_loss(model, train_loader)
                val_loss = self.evaluate_loss(model, validation_loader)
                logger.info(f"step {batch_idx}: train loss {train_loss:.4f}, val loss {val_loss:.4f}")
                accelerator.log({"train_loss": train_loss, "val_loss": val_loss}, step=tracker_idx)
                tracker_idx += 1
                model.train()
                if val_loss < self.best_val_loss:
                    accelerator.save_state()
                    self.best_val_loss = val_loss

            if batch_idx >= self.trainer_cfg.max_iters:
                break

        logger.info("Finished training")
        accelerator.wait_for_everyone()
        model = accelerator.unwrap_model(model)
        accelerator.save_model(model, str(self.model_dir))

    @torch.no_grad()
    def evaluate_loss(self, model, batch_loader):
        losses = torch.zeros(self.trainer_cfg.eval_iters)
        for iter_idx, eval_batch in enumerate(batch_loader):
            if iter_idx == self.trainer_cfg.eval_iters:
                return losses.mean()
            inputs, targets = eval_batch
            _, loss = model(inputs, targets)
            losses[iter_idx] = loss.item()
