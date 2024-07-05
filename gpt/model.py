import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from jobs.configure import GPT2Config


class GPT2Attention(nn.Module):

    def __init__(self, config: GPT2Config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.max_positions = config.block_size
        self.embed_dim = config.n_embd
        self.num_heads = config.n_head
        self.head_dim = config.n_embd // config.n_head
        # causal mask to ensure that attention is only applied to the left in the input sequence
        causal_mask = torch.tril(torch.ones(self.max_positions, self.max_positions))
        causal_mask = causal_mask.view(1, 1, self.max_positions, self.max_positions)
        self.register_buffer("bias", causal_mask)

    def forward(self, x):
        # batch size, sequence length, embedding dimensionality (n_embd)
        bsz, seq_len, embed_dim = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(embed_dim, dim=2)
        q = q.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :seq_len, :seq_len] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        # re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(bsz, seq_len, embed_dim)

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class GPT2MLP(nn.Module):

    def __init__(self, config: GPT2Config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class GPT2Block(nn.Module):

    def __init__(self, config: GPT2Config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = GPT2Attention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = GPT2MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT2Model(nn.Module):

    def __init__(self, config: GPT2Config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        self.transformer = nn.ModuleDict(
            dict(
                token_embeddings=nn.Embedding(config.vocab_size, config.n_embd),
                position_embeddings=nn.Embedding(config.block_size, config.n_embd),
                dropout=nn.Dropout(config.dropout),
                h=nn.ModuleList([GPT2Block(config) for _ in range(config.n_layer)]),
                ln_f=nn.LayerNorm(config.n_embd, bias=config.bias),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # weight tying(https://paperswithcode.com/method/weight-tying)
        self.transformer.token_embeddings.weight = self.lm_head.weight

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for parameter_name, parameter in self.named_parameters():
            if parameter_name.endswith("c_proj.weight"):
                std = 0.02 / math.sqrt(2 * config.n_layer)
                torch.nn.init.normal_(parameter, mean=0.0, std=std)

    @property
    def num_params(self):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.LongTensor, targets=None):
        device = input_ids.device
        bsz, seq_len = input_ids.size()
        if seq_len > self.config.block_size:
            raise ValueError("Sequence length is larger than block size.")

        pos = torch.arange(0, seq_len, dtype=torch.long, device=device)  # shape (t)
        # token embeddings of shape (b, t, n_embd)
        tok_emb = self.transformer.token_embeddings(input_ids)
        # position embeddings of shape (t, n_embd)
        pos_emb = self.transformer.position_embeddings(pos)
        # forward the GPT model itself
        x = self.transformer.dropout(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            # note: using list [-1] to preserve the time dim
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.position_embeddings.weight = nn.Parameter(
            self.transformer.position_embeddings.weight[:block_size]
        )
        for block in self.transformer.h:
            if hasattr(block.attn, "bias"):
                block.attn.bias = block.attn.bias[:, :, :block_size, :block_size]
