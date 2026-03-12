import torch
import os
import numpy as np
from typing import Dict
from matplotlib import pyplot as plt


@torch.no_grad()
def _save_heatmap64(hm_64: torch.Tensor,
        save_path: str,
        out_size: int = 512,
        p_lo: float = 1.0,  # percentile low clip
        p_hi: float = 99.0,  # percentile high clip
        gamma: float = 0.9,  # <1.0 => highlight bright regions
        do_reduce = True,):

    def reduce(e):
        m_4096 = e.mean(dim=1, keepdim=True)
        m4 = m_4096.view(64, 64, 64, 64)  # [qh,qw,kh,kw]
        hm = m4.mean(dim=(2, 3))  # [64,64]
        return hm

    if do_reduce :
        hm_64 = reduce(hm_64)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    hm = hm_64.detach().float().cpu().numpy()

    hm = np.nan_to_num(hm, nan=0.0, posinf=0.0, neginf=0.0)
    lo = np.percentile(hm, p_lo)
    hi = np.percentile(hm, p_hi)
    hm = np.clip(hm, lo, hi)
    denom = (hm.max() - hm.min())
    if denom < 1e-12:
        hm01 = np.zeros_like(hm, dtype=np.float32)
    else:
        hm01 = (hm - hm.min()) / (denom + 1e-8)
    if gamma is not None and gamma != 1.0:
        hm01 = np.power(hm01, gamma)
    fig = plt.figure(figsize=(out_size / 100, out_size / 100), dpi=100)
    plt.imshow(hm01, cmap="viridis", interpolation="nearest")
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def sum_blocks(blocks):
    if len(blocks) == 0:
        return None
    out = torch.zeros_like(blocks[0])
    for b in blocks:
        out += b
    return out

def split_blocks(attn_weight: torch.Tensor,
                 txt_len: int,
                 tgt_len: int,
                 src_len = 4096) -> Dict[str, torch.Tensor]:
    """
    attn_weight: (B,H,L,L), softmaxed over last dim
    token order assumed: [T, E(target), I(source)]
    """
    t0, t1 = 0, txt_len
    e0, e1 = t1, t1 + tgt_len
    i0, i1 = e1, e1 + src_len

    QT, QE, QI = slice(t0, t1), slice(e0, e1), slice(i0, i1)
    KT, KE, KI = slice(t0, t1), slice(e0, e1), slice(i0, i1)

    return {
        "T2T": attn_weight[:, :, QT, KT],
        "T2E": attn_weight[:, :, QT, KE],
        "T2I": attn_weight[:, :, QT, KI],

        "E2T": attn_weight[:, :, QE, KT],
        "E2E": attn_weight[:, :, QE, KE],
        "E2I": attn_weight[:, :, QE, KI],

        "I2T": attn_weight[:, :, QI, KT],
        "I2E": attn_weight[:, :, QI, KE],
        "I2I": attn_weight[:, :, QI, KI],
    }

