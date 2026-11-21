"""Native 32x32 Pixie with the v3 palette, conditioning and causal slot head.

Raster pixels [B,1024] become 256 patches with slots TL, TR, BL, BR. The
23-token prefix is followed by 256 target-patch positions. Learned position
tables are retained for warm-start continuity alongside partial axial RoPE.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from train import v3_model as M

PALETTE = M.PALETTE
VOCAB = M.VOCAB
NO_PALETTE = M.NO_PALETTE
N_PIX = 1024


@dataclass
class P32Config:
    arch: str = "p32"
    d_model: int = 512
    n_layer: int = 16
    n_head: int = 8
    ffn_mult: int = 4
    dropout: float = 0.1
    head_mult: int = 2
    text_len: int = 18
    n_pal: int = 3
    style: bool = True

    def __post_init__(self):
        if self.arch != "p32":
            raise ValueError("P32 requires arch=p32")
        if (self.text_len, self.n_pal, self.style) != (18, 3, True):
            raise ValueError("P32 requires text_len=18, n_pal=3 and style=True")
        if self.n_head <= 0 or self.d_model % self.n_head or self.d_model // self.n_head < 8:
            raise ValueError("head dimension must be an integer of at least eight")
        if self.n_layer < 1 or self.ffn_mult < 1 or self.head_mult < 1:
            raise ValueError("model depths and multipliers must be positive")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0,1)")

    def to_dict(self):
        return asdict(self)


def to_patches(img):
    """[B,1024] raster palette ids to [B,256,4] patch slots."""
    return img.reshape(img.shape[0], 16, 2, 16, 2).permute(0, 1, 3, 2, 4).reshape(img.shape[0], 256, 4)


def from_patches(patches):
    return patches.reshape(patches.shape[0], 16, 16, 2, 2).permute(0, 1, 3, 2, 4).reshape(patches.shape[0], N_PIX)


def apply_2d_rope(x, coordinates, rotary_dim):
    """Rotate [B,H,S,D] q/k; coordinates [S,2] are zero for the prefix."""
    axis_dim = rotary_dim // 2
    frequency = 10000.0 ** (-torch.arange(0, axis_dim, 2, device=x.device, dtype=torch.float32) / axis_dim)
    parts = []
    for axis in range(2):
        angle = coordinates[:, axis, None].float() * frequency[None]
        cosine, sine = angle.cos().to(x.dtype), angle.sin().to(x.dtype)
        pairs = x[..., axis * axis_dim:(axis + 1) * axis_dim].reshape(*x.shape[:-1], axis_dim // 2, 2)
        a, b = pairs.unbind(-1)
        parts.append(torch.stack((a * cosine - b * sine, a * sine + b * cosine), -1).flatten(-2))
    return torch.cat((*parts, x[..., rotary_dim:]), -1)


class RotaryBlock(M.Block):
    def __init__(self, config):
        super().__init__(config)
        self.rotary_dim = (self.d // self.n_head // 2) // 4 * 4

    def forward(self, x, coordinates):
        b, s, _ = x.shape
        q, k, v = self.qkv(self.ln1(x)).reshape(b, s, 3, self.n_head, self.d // self.n_head).unbind(2)
        q = apply_2d_rope(q.transpose(1, 2), coordinates, self.rotary_dim)
        k = apply_2d_rope(k.transpose(1, 2), coordinates, self.rotary_dim)
        attention = F.scaled_dot_product_attention(q, k, v.transpose(1, 2), is_causal=True,
                                                  dropout_p=self.p if self.training else 0.0)
        x = x + self.drop(self.out(attention.transpose(1, 2).reshape(b, s, self.d)))
        return x + self.drop(self.fc2(F.gelu(self.fc1(self.ln2(x)))))


class P32(M._Base):
    NP = 256
    OUT0 = 23
    SEQ = 279
    cond_emb = M.P50.cond_emb

    def __init__(self, config: P32Config):
        super().__init__()
        self.config = config
        self.TL, self.NC = config.text_len, 4
        self.gradient_checkpointing = False
        d = config.d_model
        self.init_text(d)
        self.pal_type = nn.Parameter(torch.zeros(d))
        self.style_emb = nn.ModuleList(nn.Embedding(n, d) for n in M.STYLE_SIZES)
        self.slot_emb = nn.ModuleList(nn.Embedding(VOCAB, d) for _ in range(4))
        self.pos = nn.Embedding(self.SEQ, d)
        self.row_in, self.col_in = nn.Embedding(16, d), nn.Embedding(16, d)
        self.row_out, self.col_out = nn.Embedding(16, d), nn.Embedding(16, d)
        self.blocks = nn.ModuleList(RotaryBlock(config) for _ in range(config.n_layer))
        self.ln_f = nn.LayerNorm(d)
        self.head = M.SlotCausalHead(d, config.head_mult)
        self.apply(M._init)
        for parameter in (self.bos, self.sep, self.head.q, self.pal_type):
            nn.init.normal_(parameter, std=0.02)
        indices = torch.arange(self.NP)
        self.register_buffer("pr", indices // 16, persistent=False)
        self.register_buffer("pc", indices % 16, persistent=False)
        self.register_buffer("vmask", M.vocab_mask("cpu"), persistent=False)
        coordinates = torch.cat((torch.zeros(self.OUT0, 2, dtype=torch.long),
                                 torch.stack((indices // 16, indices % 16), -1)))
        self.register_buffer("coordinates", coordinates, persistent=False)

    def trunk(self, clip_h, patches, cond=None, *, n_patches=None):
        """Target m sees the prefix and patches < m; optional truncated decoding."""
        count = self.NP if n_patches is None else n_patches
        if clip_h.shape[1:] != (self.TL, M.CLIP_D):
            raise ValueError("CLIP features must have shape [B,18,768]")
        if not 1 <= count <= self.NP:
            raise ValueError("n_patches must be in [1,256]")
        b, d = patches.shape[0], self.config.d_model
        text = self.embed_text(clip_h)
        pin = patches[:, :count - 1]
        image = sum(self.slot_emb[j](pin[..., j]) for j in range(4))
        image = image + self.row_in(self.pr[:count - 1]) + self.col_in(self.pc[:count - 1])
        x = torch.cat((self.bos.expand(b, 1, d).to(text.dtype), text,
                       self.cond_emb(cond, b, patches.device).to(text.dtype),
                       self.sep.expand(b, 1, d).to(text.dtype), image), 1)
        x = x + self.pos.weight[None, :self.OUT0 + count]
        target = self.row_out(self.pr[:count]) + self.col_out(self.pc[:count])
        x = torch.cat((x[:, :self.OUT0], x[:, self.OUT0:] + target[None]), 1)
        coordinates = self.coordinates[:x.shape[1]]
        for block in self.blocks:
            if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                x = checkpoint(block, x, coordinates, use_reentrant=False)
            else:
                x = block(x, coordinates)
        return self.ln_f(x[:, self.OUT0:])

    def forward(self, clip_h, img, cond=None):
        patches = to_patches(img)
        logits = self.head(self.trunk(clip_h, patches, cond), patches) + self.vmask
        return logits.reshape(img.shape[0], 16, 16, 2, 2, VOCAB).permute(0, 1, 3, 2, 4, 5).reshape(img.shape[0], N_PIX, VOCAB)

    def loss_fwd(self, clip_h, img, alpha_weight=0.25, cond=None, chunks=16):
        """Checkpoint CE over flattened patch positions, including batch size one.

        `chunks` splits B*256 patch positions. Each chunk projects four slots
        to the padded vocabulary and reduces immediately, keeping logits out
        of the retained backward graph.
        """
        if chunks < 0:
            raise ValueError("chunks must be nonnegative")
        patches = to_patches(img)
        hidden = self.head.pre(self.trunk(clip_h, patches, cond), patches).reshape(-1, 4, self.config.d_model)
        targets = patches.reshape(-1, 4)

        def part(values, target):
            logits = (self.head.W_out(values) + self.vmask).float()
            lse = torch.logsumexp(logits, -1)
            nll = lse - logits.gather(-1, target.unsqueeze(-1)).squeeze(-1)
            lp0 = (logits[..., 0] - lse).clamp(max=-1e-6)
            alpha = torch.where(target == 0, -lp0, -torch.log(-torch.expm1(lp0)))
            return nll.sum(), alpha.sum()

        nll_sum, alpha_sum = hidden.new_zeros(()), hidden.new_zeros(())
        for values, target in zip(hidden.chunk(max(1, chunks), 0), targets.chunk(max(1, chunks), 0)):
            if chunks and torch.is_grad_enabled():
                nll, alpha = checkpoint(part, values, target, use_reentrant=False)
            else:
                nll, alpha = part(values, target)
            nll_sum, alpha_sum = nll_sum + nll, alpha_sum + alpha
        mean_nll = nll_sum / img.numel()
        return mean_nll + alpha_weight * alpha_sum / img.numel(), mean_nll

    def load_v3_warmstart(self, path):
        """Load a v3 P50 checkpoint strictly, remapping only position tables.

        RoPE has no learned or persistent tensors, so `new` is empty after a
        complete warm start. The report accounts for every state-dict tensor.
        """
        source = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(source, dict) or not isinstance(source.get("config"), dict) or not isinstance(source.get("model"), dict):
            raise ValueError("warm-start requires v3 model and config dictionaries")
        config = source["config"]
        if config.get("arch") != "p50" or source.get("arch", "p50") != "p50":
            raise ValueError("warm-start requires the v3 p50 architecture")
        for key in ("d_model", "n_layer", "n_head", "ffn_mult", "dropout", "head_mult", "text_len", "n_pal", "style"):
            if config.get(key) != getattr(self.config, key):
                raise ValueError(f"warm-start config mismatch for {key}")
        state = source["model"]
        own = self.state_dict()
        if state.keys() != own.keys():
            missing = sorted(own.keys() - state.keys())
            extra = sorted(state.keys() - own.keys())
            raise ValueError(f"warm-start tensor keys differ; missing={missing}, extra={extra}")
        mapped, report = {}, {"copied": [], "remapped": [], "new": []}
        for key, target in own.items():
            value = state[key]
            if key == "pos.weight":
                if tuple(value.shape) != (self.OUT0 + 64, self.config.d_model):
                    raise ValueError("warm-start shape mismatch for pos.weight")
                patch_pos = value[self.OUT0:].reshape(8, 8, -1).repeat_interleave(2, 0).repeat_interleave(2, 1).reshape(self.NP, -1)
                mapped[key] = torch.cat((value[:self.OUT0], patch_pos))
                report["remapped"].append(key)
            elif key in {f"{name}.weight" for name in ("row_in", "col_in", "row_out", "col_out")}:
                if tuple(value.shape) != (8, self.config.d_model):
                    raise ValueError(f"warm-start shape mismatch for {key}")
                mapped[key] = value.repeat_interleave(2, 0)
                report["remapped"].append(key)
            elif value.shape == target.shape:
                mapped[key] = value
                report["copied"].append(key)
            else:
                raise ValueError(f"warm-start shape mismatch for {key}")
        self.load_state_dict(mapped, strict=True)
        return report

    @torch.no_grad()
    def sample(self, clip_h, clip_u=None, *, cfg=1.0, temperature=0.8, top_k=32, generator=None, cond=None):
        if cfg != 1 and (clip_u is None or clip_u.shape != clip_h.shape):
            raise ValueError("CFG requires unconditional CLIP features matching clip_h")
        if not 0 <= top_k <= PALETTE:
            raise ValueError("top_k must be in [0,4097]")
        b, device = clip_h.shape[0], clip_h.device
        amp_dtype = (torch.bfloat16 if torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16) if device.type == "cuda" else torch.float32
        patches = torch.zeros(b, self.NP, 4, dtype=torch.long, device=device)
        text = torch.cat((clip_h, clip_u)) if cfg != 1 else clip_h
        conditions = None if cond is None else (torch.cat((cond, cond)) if cfg != 1 else cond)
        for m in range(self.NP):
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
                previous = torch.cat((patches, patches)) if cfg != 1 else patches
                h = self.trunk(text, previous, conditions, n_patches=m + 1)[:, -1]
                slots = []
                for k in range(4):
                    prefix = [torch.cat((slot, slot)) for slot in slots] if cfg != 1 else slots
                    logits = self.head.step(h, prefix).float()
                    if cfg != 1:
                        logits = logits[b:] + cfg * (logits[:b] - logits[b:])
                    pixel = M._pick(logits, temperature, top_k, generator)
                    patches[:, m, k] = pixel
                    slots.append(pixel)
        return from_patches(patches)
