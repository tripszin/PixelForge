"""Pixie v3 models (self-contained; pixelgen is not imported or modified).

Text: frozen CLIP-L/14 text transformer, penultimate hidden states (= HF
hidden_states[-2], i.e. 11 of 12 layers, no final LayerNorm), run online in
bf16 under no_grad. Input = 26 CLIP ids (BOS, <=24 ids, EOS, EOS-padded).

PixAR  : pilot (a). v2-size trunk (d256 x L12), 256 pixel tokens, seq 283.
P50    : pilot (b). d512 x L16, 2x2 patch tokens (64), seq 91, slot-aware
         causal MLP intra-patch head. Pixel order inside a patch: TL, TR, BL, BR.
Both: palette vocab padded to 4160 (indices >= 4097 masked to -inf), loss =
exact 4097-way NLL + 0.25 x alpha NLL, causal SDPA via is_causal (no attn_mask).
"""
from __future__ import annotations
import math
from dataclasses import asdict, dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

PALETTE = 4097
VOCAB = 4160
TEXT_LEN = 26
N_PIX = 256
CLIP_D = 768
CLIP_LAYERS_USED = 11
CLIP_BOS, CLIP_EOS = 49406, 49407


# ---------------------------------------------------------------- CLIP text (frozen)
class _ClipLayer(nn.Module):
    def __init__(self, d=CLIP_D, h=12):
        super().__init__()
        self.h = h
        self.layer_norm1 = nn.LayerNorm(d); self.layer_norm2 = nn.LayerNorm(d)
        self.q_proj = nn.Linear(d, d); self.k_proj = nn.Linear(d, d); self.v_proj = nn.Linear(d, d)
        self.out_proj = nn.Linear(d, d)
        self.fc1 = nn.Linear(d, 4 * d); self.fc2 = nn.Linear(4 * d, d)

    def forward(self, x):
        b, s, d = x.shape
        y = self.layer_norm1(x)
        q, k, v = (p(y).view(b, s, self.h, d // self.h).transpose(1, 2) for p in (self.q_proj, self.k_proj, self.v_proj))
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.out_proj(a.transpose(1, 2).reshape(b, s, d))
        y = self.fc1(self.layer_norm2(x))
        y = y * torch.sigmoid(1.702 * y)  # quick_gelu
        return x + self.fc2(y)


class ClipTextPenultimate(nn.Module):
    """Frozen; weights from openai/clip-vit-large-patch14 text_model.*"""

    def __init__(self, n_layers=CLIP_LAYERS_USED):
        super().__init__()
        self.token_embedding = nn.Embedding(49408, CLIP_D)
        self.position_embedding = nn.Embedding(77, CLIP_D)
        self.layers = nn.ModuleList(_ClipLayer() for _ in range(n_layers))
        for p in self.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, ids):  # ids [B, S<=77] long
        x = self.token_embedding(ids) + self.position_embedding.weight[: ids.shape[1]][None]
        for l in self.layers:
            x = l(x)
        return x

    def load_hf(self, sd: dict):
        """sd: HF-style names ("text_model.encoder.layers.0.self_attn.q_proj.weight", ...)."""
        m = {"token_embedding.weight": "text_model.embeddings.token_embedding.weight",
             "position_embedding.weight": "text_model.embeddings.position_embedding.weight"}
        for i in range(len(self.layers)):
            p = f"text_model.encoder.layers.{i}."
            for a, b in (("q_proj", "self_attn.q_proj"), ("k_proj", "self_attn.k_proj"), ("v_proj", "self_attn.v_proj"),
                         ("out_proj", "self_attn.out_proj"), ("fc1", "mlp.fc1"), ("fc2", "mlp.fc2"),
                         ("layer_norm1", "layer_norm1"), ("layer_norm2", "layer_norm2")):
                for w in ("weight", "bias"):
                    m[f"layers.{i}.{a}.{w}"] = p + f"{b}.{w}"
        own = {k: sd[v] for k, v in m.items()}
        self.load_state_dict(own, strict=True)


# ---------------------------------------------------------------- shared trunk
@dataclass
class V3Config:
    arch: str = "pix"          # "pix" | "p50"
    d_model: int = 256
    n_layer: int = 12
    n_head: int = 8
    ffn_mult: int = 4
    dropout: float = 0.1
    head_mult: int = 2          # p50 intra-patch MLP hidden = head_mult * d
    text_len: int = TEXT_LEN    # s7: 18 = BOS + 16 ids + EOS (old runs: 26)
    n_pal: int = 0              # s7: palette conditioning slots (E_pix ids, NO_PALETTE = 4097)
    style: bool = False         # s7: one style slot (fill / outline / colour-count tags, 0 = ANY)

    def to_dict(self):
        return asdict(self)


class Block(nn.Module):
    def __init__(self, c: V3Config):
        super().__init__()
        self.n_head, self.d, self.p = c.n_head, c.d_model, c.dropout
        self.ln1 = nn.LayerNorm(c.d_model); self.qkv = nn.Linear(c.d_model, 3 * c.d_model)
        self.out = nn.Linear(c.d_model, c.d_model); self.ln2 = nn.LayerNorm(c.d_model)
        self.fc1 = nn.Linear(c.d_model, c.d_model * c.ffn_mult); self.fc2 = nn.Linear(c.d_model * c.ffn_mult, c.d_model)
        self.drop = nn.Dropout(c.dropout)

    def forward(self, x):
        b, s, _ = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(b, s, 3, self.n_head, self.d // self.n_head).unbind(2)
        a = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                                           is_causal=True, dropout_p=self.p if self.training else 0.0)
        x = x + self.drop(self.out(a.transpose(1, 2).reshape(b, s, self.d)))
        return x + self.drop(self.fc2(F.gelu(self.fc1(self.ln2(x)))))


def _init(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, std=0.02)


NO_PALETTE = PALETTE          # 4097: padded-vocab id, never a pixel target, used as the empty palette slot
STYLE_SIZES = (4, 3, 4)       # fill: any/small/medium/full; outline: any/dark/soft; colours: any/low/mid/high
N_COND_STYLE = 3              # style ids per sample in the cond tensor


def default_cond(c, b, device):
    """cond [B, n_pal + 3]: all NO_PALETTE + ANY style (what dev eval and plain prompts use)."""
    t = torch.zeros(b, c.n_pal + N_COND_STYLE, dtype=torch.long, device=device)
    t[:, :c.n_pal] = NO_PALETTE
    return t


def vocab_mask(device):
    m = torch.zeros(VOCAB, device=device)
    m[PALETTE:] = float("-inf")
    return m


class _Base(nn.Module):
    def init_text(self, d):
        self.clip_ln = nn.LayerNorm(CLIP_D)
        self.text_proj = nn.Linear(CLIP_D, d)
        self.bos = nn.Parameter(torch.zeros(d)); self.sep = nn.Parameter(torch.zeros(d))

    def embed_text(self, clip_h):  # clip_h [B,26,768] (any float dtype)
        return self.text_proj(self.clip_ln(clip_h.float()))

    def n_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------- pilot (a): pixel AR
class PixAR(_Base):
    SEQ = 1 + TEXT_LEN + 1 + N_PIX - 1  # 283
    OUT0 = 1 + TEXT_LEN

    def __init__(self, c: V3Config):
        super().__init__()
        self.config = c; d = c.d_model
        self.init_text(d)
        self.img_emb = nn.Embedding(VOCAB, d)
        self.pos = nn.Embedding(self.SEQ, d)
        self.row_in = nn.Embedding(16, d); self.col_in = nn.Embedding(16, d)
        self.row_out = nn.Embedding(16, d); self.col_out = nn.Embedding(16, d)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.n_layer))
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, VOCAB, bias=False)
        self.apply(_init)
        nn.init.normal_(self.bos, std=0.02); nn.init.normal_(self.sep, std=0.02)
        j = torch.arange(N_PIX)
        self.register_buffer("pr", j // 16, persistent=False); self.register_buffer("pc", j % 16, persistent=False)
        self.register_buffer("vmask", vocab_mask("cpu"), persistent=False)

    def forward(self, clip_h, img):
        """clip_h [B,26,768], img [B,256] raster palette ids -> logits [B,256,VOCAB] (raster order)."""
        b = img.shape[0]; d = self.config.d_model
        t = self.embed_text(clip_h)
        im = self.img_emb(img[:, :N_PIX - 1]) + self.row_in(self.pr[:-1]) + self.col_in(self.pc[:-1])
        x = torch.cat([self.bos.expand(b, 1, d).to(t.dtype), t, self.sep.expand(b, 1, d).to(t.dtype), im], 1)
        x = x + self.pos.weight[None]
        tgt = self.row_out(self.pr) + self.col_out(self.pc)
        x = torch.cat([x[:, :self.OUT0], x[:, self.OUT0:] + tgt[None]], 1)
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.ln_f(x[:, self.OUT0:])) + self.vmask

    @torch.no_grad()
    def sample(self, clip_h, clip_u=None, *, cfg=1.0, temperature=0.8, top_k=32, generator=None):
        b = clip_h.shape[0]; dev = clip_h.device
        img = torch.zeros(b, N_PIX, dtype=torch.long, device=dev)
        hh = torch.cat([clip_h, clip_u], 0) if cfg != 1.0 else clip_h
        for k in range(N_PIX):
            with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                lg = self(hh, torch.cat([img, img], 0) if cfg != 1.0 else img)[:, k].float()
            if cfg != 1.0:
                lg = lg[b:] + cfg * (lg[:b] - lg[b:])
            img[:, k] = _pick(lg, temperature, top_k, generator)
        return img


def _pick(lg, temperature, top_k, generator):
    lg = lg[:, :PALETTE]
    if temperature <= 0:
        return lg.argmax(-1)
    lg = lg / temperature
    if top_k:
        v, _ = lg.topk(top_k, -1)
        lg = lg.masked_fill(lg < v[:, -1:], float("-inf"))
    return torch.multinomial(F.softmax(lg, -1), 1, generator=generator).squeeze(-1)


# ---------------------------------------------------------------- patch helpers
def to_patches(img):
    """[B,256] raster -> [B,64,4] (patch raster order; slots TL,TR,BL,BR)."""
    b = img.shape[0]
    return img.view(b, 8, 2, 8, 2).permute(0, 1, 3, 2, 4).reshape(b, 64, 4)


def from_patches(p):
    b = p.shape[0]
    return p.view(b, 8, 8, 2, 2).permute(0, 1, 3, 2, 4).reshape(b, N_PIX)


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__(); self.w = nn.Parameter(torch.ones(d)); self.eps = eps

    def forward(self, x):
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)).to(x.dtype) * self.w


class SlotCausalHead(nn.Module):
    """a   = W_h h + sum_{j<k} W_j E_pix[p_j]       (one matrix per slot j)
       u_k = h + GELU(a + q_k) W_2                  (q_k slot embedding, hidden = head_mult*d)
       logits_k = W_out RMSNorm(u_k)                (vocab padded to 4160)"""

    def __init__(self, d, mult=2):
        super().__init__()
        hd = mult * d
        self.W_h = nn.Linear(d, hd)
        self.W_j = nn.ModuleList(nn.Linear(d, hd, bias=False) for _ in range(3))
        self.E_pix = nn.Embedding(VOCAB, d)
        self.q = nn.Parameter(torch.zeros(4, hd))
        self.W_2 = nn.Linear(hd, d)
        self.norm = RMSNorm(d)
        self.W_out = nn.Linear(d, VOCAB, bias=False)

    def forward(self, h, patches):
        """Teacher forcing: h [B,P,d], patches [B,P,4] -> logits [B,P,4,VOCAB]. Slot k sees p_j only for j<k."""
        base = self.W_h(h)                                           # [B,P,hd]
        e = [self.W_j[j](self.E_pix(patches[..., j])) for j in range(3)]   # each [B,P,hd]
        a = torch.stack([base, base + e[0], base + e[0] + e[1], base + e[0] + e[1] + e[2]], 2)  # [B,P,4,hd]
        u = h.unsqueeze(2) + self.W_2(F.gelu(a + self.q))
        return self.W_out(self.norm(u))

    def pre(self, h, patches):
        """Same as forward but stops before W_out: RMSNorm(u) [B,P,4,d] (for chunked CE)."""
        base = self.W_h(h)
        e = [self.W_j[j](self.E_pix(patches[..., j])) for j in range(3)]
        a = torch.stack([base, base + e[0], base + e[0] + e[1], base + e[0] + e[1] + e[2]], 2)
        return self.norm(h.unsqueeze(2) + self.W_2(F.gelu(a + self.q)))

    def step(self, h, prev):
        """Sampling one slot: h [B,d], prev list of already-chosen pixel ids ([B] each) -> logits [B,VOCAB]."""
        a = self.W_h(h)
        for j, p in enumerate(prev):
            a = a + self.W_j[j](self.E_pix(p))
        k = len(prev)
        u = h + self.W_2(F.gelu(a + self.q[k]))
        return self.W_out(self.norm(u))


# ---------------------------------------------------------------- pilot (b): P50
class P50(_Base):
    NP = 64
    SEQ = 1 + TEXT_LEN + 1 + NP - 1  # 91
    OUT0 = 1 + TEXT_LEN

    def __init__(self, c: V3Config):
        super().__init__()
        self.config = c; d = c.d_model
        self.TL = c.text_len; self.NC = c.n_pal + (1 if c.style else 0)
        self.OUT0 = 1 + self.TL + self.NC; self.SEQ = self.OUT0 + self.NP   # old config: 27 / 91
        self.init_text(d)
        if c.n_pal:
            self.pal_type = nn.Parameter(torch.zeros(d))
        if c.style:
            self.style_emb = nn.ModuleList(nn.Embedding(n, d) for n in STYLE_SIZES)
        self.slot_emb = nn.ModuleList(nn.Embedding(VOCAB, d) for _ in range(4))
        self.pos = nn.Embedding(self.SEQ, d)
        self.row_in = nn.Embedding(8, d); self.col_in = nn.Embedding(8, d)
        self.row_out = nn.Embedding(8, d); self.col_out = nn.Embedding(8, d)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.n_layer))
        self.ln_f = nn.LayerNorm(d)
        self.head = SlotCausalHead(d, c.head_mult)
        self.apply(_init)
        nn.init.normal_(self.bos, std=0.02); nn.init.normal_(self.sep, std=0.02)
        nn.init.normal_(self.head.q, std=0.02)
        if c.n_pal:
            nn.init.normal_(self.pal_type, std=0.02)
        j = torch.arange(self.NP)
        self.register_buffer("pr", j // 8, persistent=False); self.register_buffer("pc", j % 8, persistent=False)
        self.register_buffer("vmask", vocab_mask("cpu"), persistent=False)

    def cond_emb(self, cond, b, device):
        """cond [B, n_pal+3] long (None = default NO_PALETTE / ANY) -> [B, NC, d] prefix tokens."""
        c = self.config
        if cond is None:
            cond = default_cond(c, b, device)
        out = []
        if c.n_pal:
            out.append(self.head.E_pix(cond[:, :c.n_pal]) + self.pal_type)
        if c.style:
            s = cond[:, c.n_pal:]
            out.append((self.style_emb[0](s[:, 0]) + self.style_emb[1](s[:, 1]) + self.style_emb[2](s[:, 2]))[:, None])
        return torch.cat(out, 1)

    def trunk(self, clip_h, patches, cond=None):
        """patches [B,64,4] -> h [B,64,d]; h[:,m] depends only on text (+cond) and patches < m."""
        b = patches.shape[0]; d = self.config.d_model
        assert clip_h.shape[1] == self.TL, (clip_h.shape, self.TL)
        t = self.embed_text(clip_h)
        pin = patches[:, :self.NP - 1]
        im = sum(self.slot_emb[j](pin[..., j]) for j in range(4)) + self.row_in(self.pr[:-1]) + self.col_in(self.pc[:-1])
        pre = [self.bos.expand(b, 1, d).to(t.dtype), t]
        if self.NC:
            pre.append(self.cond_emb(cond, b, patches.device).to(t.dtype))
        x = torch.cat(pre + [self.sep.expand(b, 1, d).to(t.dtype), im], 1)
        x = x + self.pos.weight[None]
        tgt = self.row_out(self.pr) + self.col_out(self.pc)
        x = torch.cat([x[:, :self.OUT0], x[:, self.OUT0:] + tgt[None]], 1)
        for blk in self.blocks:
            x = blk(x)
        return self.ln_f(x[:, self.OUT0:])

    def forward(self, clip_h, img, cond=None):
        """clip_h [B,TL,768], img [B,256] raster -> logits [B,256,VOCAB] in RASTER order (same layout as PixAR)."""
        p = to_patches(img)
        lg = self.head(self.trunk(clip_h, p, cond), p) + self.vmask      # [B,64,4,V]
        b = img.shape[0]
        return lg.view(b, 8, 8, 2, 2, VOCAB).permute(0, 1, 3, 2, 4, 5).reshape(b, N_PIX, VOCAB)

    def loss_fwd(self, clip_h, img, alpha_weight, cond=None, chunks=0):
        """Training loss without building raster logits: CE in patch order (targets permuted, not logits).
        chunks > 0: W_out + log-softmax per batch chunk under activation checkpointing, so the
        [B,64,4,4160] logits are never all alive at once. Same math as losses(); returns (loss, mean nll)."""
        p = to_patches(img)
        u = self.head.pre(self.trunk(clip_h, p, cond), p)          # [B,64,4,d]

        def part(uu, tt):
            lg = (self.head.W_out(uu) + self.vmask).float()
            lse = torch.logsumexp(lg, -1)
            nll = lse - lg.gather(-1, tt.unsqueeze(-1)).squeeze(-1)
            lp0 = (lg[..., 0] - lse).clamp(max=-1e-6)
            a = torch.where(tt == 0, -lp0, -torch.log(-torch.expm1(lp0)))
            return nll.sum(), a.sum()
        if chunks and chunks > 1:
            from torch.utils.checkpoint import checkpoint
            ns = []; as_ = []
            for uu, tt in zip(u.chunk(chunks, 0), p.chunk(chunks, 0)):
                n_, a_ = checkpoint(part, uu, tt, use_reentrant=False)
                ns.append(n_); as_.append(a_)
            nsum = sum(ns); asum = sum(as_)
        else:
            nsum, asum = part(u, p)
        cnt = img.numel()
        nll = nsum / cnt
        return nll + alpha_weight * (asum / cnt), nll

    @torch.no_grad()
    def sample(self, clip_h, clip_u=None, *, cfg=1.0, temperature=0.8, top_k=32, generator=None, cond=None):
        b = clip_h.shape[0]; dev = clip_h.device
        P = torch.zeros(b, self.NP, 4, dtype=torch.long, device=dev)
        hh = torch.cat([clip_h, clip_u], 0) if cfg != 1.0 else clip_h
        cc = None if cond is None else (torch.cat([cond, cond], 0) if cfg != 1.0 else cond)
        for m in range(self.NP):
            with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                h = self.trunk(hh, torch.cat([P, P], 0) if cfg != 1.0 else P, cc)[:, m]
                prev = []
                for k in range(4):
                    pp = [torch.cat([x, x], 0) for x in prev] if cfg != 1.0 else prev
                    lg = (self.head.step(h, pp) + self.vmask).float()
                    if cfg != 1.0:
                        lg = lg[b:] + cfg * (lg[:b] - lg[b:])
                    nxt = _pick(lg, temperature, top_k, generator)
                    P[:, m, k] = nxt; prev.append(nxt)
        return from_patches(P)


def build(c: V3Config):
    return PixAR(c) if c.arch == "pix" else P50(c)


def losses(logits, img, alpha_weight):
    """Exact 4097-way NLL per pixel ([B,256]) + alpha NLL; logits in raster order, padded vocab masked."""
    logp = F.log_softmax(logits.float(), dim=-1)
    nll = -logp.gather(-1, img.unsqueeze(-1)).squeeze(-1)
    lp0 = logp[..., 0]
    lop = torch.log(-torch.expm1(lp0.clamp(max=-1e-6)))
    a_nll = torch.where(img == 0, -lp0, -lop)
    return nll.mean() + alpha_weight * a_nll.mean(), nll, a_nll, lp0


SLOT_OF_PIXEL = None


def slot_of_pixel(device):
    j = torch.arange(N_PIX, device=device)
    return (j // 16 % 2) * 2 + (j % 16 % 2)  # 0 TL, 1 TR, 2 BL, 3 BR


def boundary_ratio(img):
    """P(neighbour pixels differ | pair crosses a 2x2 patch boundary) / P(differ | pair inside a patch).
    img [B,256] long. Real sprites give ~1; >1 means visible block seams."""
    x = img.view(-1, 16, 16)
    dh = (x[:, :, 1:] != x[:, :, :-1]).float()   # pair between col c and c+1
    dv = (x[:, 1:, :] != x[:, :-1, :]).float()
    c = torch.arange(15, device=img.device)
    cross = (c % 2 == 1)   # (1,2),(3,4)... cross a boundary
    inside = ~cross
    num = torch.cat([dh[:, :, cross].flatten(), dv[:, cross, :].flatten()]).mean()
    den = torch.cat([dh[:, :, inside].flatten(), dv[:, inside, :].flatten()]).mean()
    return float(num / den.clamp(min=1e-9)), float(num), float(den)
