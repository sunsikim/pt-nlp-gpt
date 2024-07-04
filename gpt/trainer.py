import torch
import math
from gpt.model import GPT2Model


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
            decay_ratio = 1.0 if decay_ratio > 1.0 else decay_ratio
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
    optimizer = torch.optim.AdamW(
        optim_groups,
        lr=learning_rate,
        betas=(beta1, beta2),
    )
    return optimizer
