import os
import pickle
import logging
import torch
import accelerate
from gpt.model import GPT2Model
from gpt.data import GPT2Dataset, define_batch_loader
from gpt.trainer import CosineWarmupScheduler, define_optimizer
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
        self.best_val_loss = 1e9

    def execute(self):
        torch.manual_seed(1337)

        logging.info("Deriving vocab_size from the dataset")
        meta_path = os.path.join(self.data_dir, "meta.pkl")
        if os.path.exists(meta_path):
            with open(meta_path, "rb") as f:
                meta = pickle.load(f)
            meta_vocab_size = meta["vocab_size"]
            print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")
            self.model_cfg.vocab_size = meta_vocab_size

        logging.info("Define train objects")
        model = GPT2Model(self.model_cfg)
        train_loader = define_batch_loader(
            dataset=GPT2Dataset(
                data_dir=self.data_dir,
                block_size=self.trainer_cfg.batch_size,
                split="train",
            ),
            batch_size=self.trainer_cfg.batch_size,
        )
        validation_loader = define_batch_loader(
            dataset=GPT2Dataset(
                data_dir=self.data_dir,
                block_size=self.trainer_cfg.batch_size,
                split="train",
            ),
            batch_size=self.trainer_cfg.batch_size,
        )
        optimizer = define_optimizer(
            model=model,
            weight_decay=self.optimizer_cfg.weight_decay,
            learning_rate=self.optimizer_cfg.learning_rate,
            beta1=self.optimizer_cfg.beta1,
            beta2=self.optimizer_cfg.beta2,
        )
        scheduler = CosineWarmupScheduler(
            optimizer=optimizer,
            warmup_steps=self.trainer_cfg.warmup_steps,
            learning_rate=self.optimizer_cfg.learning_rate,
            max_steps=self.trainer_cfg.warmup_steps,
        )
        project_config = accelerate.utils.ProjectConfiguration(
            project_dir=str(self.temp_dir),
            automatic_checkpoint_naming=True,
            total_limit=50,
        )
        accelerator = accelerate.Accelerator(
            project_config=project_config,
            gradient_accumulation_steps=self.trainer_cfg.gradient_accumulation_steps,
        )
        model, train_loader, validation_loader, optimizer, scheduler = accelerator.prepare(
            model, train_loader, validation_loader, optimizer, scheduler
        )

        logging.info("Start training")
        for batch_idx, train_batch in enumerate(train_loader, start=1):
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

            if batch_idx % self.trainer_cfg.eval_interval == 0 and accelerator.is_main_process:
                model.eval()
                train_loss = self.estimate_loss(model, train_loader)
                val_loss = self.estimate_loss(model, validation_loader)
                logging.info(f"step {batch_idx}: train loss {train_loss:.4f}, val loss {val_loss:.4f}")
                model.train()

    def estimate_loss(self, model, batch_loader):
        losses = torch.zeros(self.trainer_cfg.eval_iters)
        for iter_idx, eval_batch in enumerate(batch_loader):
            inputs, targets = eval_batch
            _, loss = model(inputs, targets)
            losses[iter_idx] = loss.item()
            if iter_idx == self.trainer_cfg.eval_iters:
                return losses.mean()
