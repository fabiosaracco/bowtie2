"""Visualisation utilities for the bowtie validate() output."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe


# ── Canonical (x, y) positions for the 7 standard bowtie blocks ──────────────
_CANONICAL_POS = {
    'SCC':         ( 0.0,  0.0),
    'IN':          (-4.5,  0.0),
    'OUT':         ( 4.5,  0.0),
    'TUBES':       ( 0.0, -3.5),
    'INTENDRILS':  (-3.5,  3.5),
    'OUTTENDRILS': ( 3.5,  3.5),
    'OTHERS':      ( 0.0,  4.5),
}

# Default colormaps:
#   blocks → RdYlGn : p=0 (significant) = red, p=1 = green
#   fluxes → RdPu_r : p=0 (significant) = dark purple, p=1 = light
_BLOCK_CMAP = 'RdYlGn'
_FLUX_CMAP  = 'RdPu_r'


# ── Helper: layout ────────────────────────────────────────────────────────────

def _positions(blocks):
    """Return {block: (x, y)}, using canonical positions where known."""
    pos, unknown = {}, []
    for b in blocks:
        if b in _CANONICAL_POS:
            pos[b] = _CANONICAL_POS[b]
        else:
            unknown.append(b)
    for i, b in enumerate(unknown):
        angle = 2 * np.pi * i / max(len(unknown), 1)
        pos[b] = (7.0 * np.cos(angle), 7.0 * np.sin(angle))
    return pos


def _radii(block_dict, r_min=0.35, r_max=1.4):
    """Circle radius for each block ∝ log(obs + 1)."""
    names = list(block_dict.keys())
    raw   = np.array([np.log1p(block_dict[b]['obs']) for b in names], dtype=float)
    lo, hi = raw.min(), raw.max()
    s = (raw - lo) / (hi - lo) if hi > lo else np.full(len(raw), 0.5)
    return {b: r_min + s[i] * (r_max - r_min) for i, b in enumerate(names)}


def _linewidths(flux_dict, lw_min=0.8, lw_max=9.0):
    """Arrow linewidth for each flux ∝ log(obs + 1)."""
    if not flux_dict:
        return {}
    keys = list(flux_dict.keys())
    raw  = np.array([np.log1p(flux_dict[k]['obs']) for k in keys], dtype=float)
    lo, hi = raw.min(), raw.max()
    s = (raw - lo) / (hi - lo) if hi > lo else np.full(len(raw), 0.5)
    return {k: lw_min + s[i] * (lw_max - lw_min) for i, k in enumerate(keys)}


# ── Helper: drawing primitives ────────────────────────────────────────────────

def _offset_endpoints(x0, y0, x1, y1, r0, r1):
    """Shift arrow start/end to lie on the circle boundaries."""
    dx, dy = x1 - x0, y1 - y0
    d = np.hypot(dx, dy)
    if d < 1e-9:
        return x0, y0, x1, y1
    ux, uy = dx / d, dy / d
    return x0 + r0 * ux, y0 + r0 * uy, x1 - r1 * ux, y1 - r1 * uy


def _draw_self_loop(ax, x, y, r, color, lw):
    """Small circular loop drawn above a block circle."""
    lr = r * 0.55
    cx, cy = x, y + r + lr
    theta = np.linspace(0.0, 2 * np.pi, 120)
    lx = cx + lr * np.cos(theta)
    ly = cy + lr * np.sin(theta)
    ax.plot(lx, ly, color=color, lw=lw, zorder=4, solid_capstyle='round')
    ax.annotate('', xy=(lx[-1], ly[-1] - 0.01), xytext=(lx[-2], ly[-2]),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                                mutation_scale=8 + lw * 0.5),
                zorder=5)


# ── Core scene renderer ───────────────────────────────────────────────────────

def _draw_scene(ax, block_dict, flux_dict, radii, lws, pos,
                block_cmap, flux_cmap, norm,
                show_block_color, show_block_size,
                show_flux_color, show_flux_size,
                neutral_r=0.55, neutral_lw=2.0):
    """Draw one complete bowtie panel onto *ax*.

    Parameters
    ----------
    show_block_color : bool   – colour blocks by their p-value
    show_block_size  : bool   – size blocks by log(obs)
    show_flux_color  : bool   – colour arrows by their p-value
    show_flux_size   : bool   – size arrows by log(obs)
    neutral_r        : float  – uniform radius when show_block_size=False
    neutral_lw       : float  – uniform linewidth when show_flux_size=False
    """
    # ── arrows (drawn first, behind circles) ─────────────────────────────────
    for fkey, fval in flux_dict.items():
        src, tgt = fkey.split('->')
        if src not in pos or tgt not in pos:
            continue
        lw    = lws.get(fkey, neutral_lw) if show_flux_size else neutral_lw
        color = flux_cmap(norm(fval['p_value'])) if show_flux_color else '0.60'
        r0    = radii.get(src, neutral_r) if show_block_size else neutral_r
        r1    = radii.get(tgt, neutral_r) if show_block_size else neutral_r

        if src == tgt:
            _draw_self_loop(ax, pos[src][0], pos[src][1], r0, color, lw)
        else:
            x0, y0, x1, y1 = _offset_endpoints(
                pos[src][0], pos[src][1], pos[tgt][0], pos[tgt][1], r0, r1)
            ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                        arrowprops=dict(
                            arrowstyle='->', color=color, lw=lw,
                            mutation_scale=10 + lw,
                            connectionstyle='arc3,rad=0.08'),
                        zorder=4)

    # ── block circles ─────────────────────────────────────────────────────────
    for b, bval in block_dict.items():
        x, y  = pos[b]
        r     = radii[b] if show_block_size else neutral_r
        color = block_cmap(norm(bval['p_value'])) if show_block_color else '0.82'
        ax.add_patch(plt.Circle((x, y), r, color=color, ec='black', lw=0.9, zorder=3))
        fs = max(6, int(10 * r))
        ax.text(x, y + r * 0.18, b,
                ha='center', va='center', fontsize=fs, fontweight='bold', zorder=5,
                path_effects=[pe.withStroke(linewidth=2, foreground='white')])
        ax.text(x, y - r * 0.28, f'n={bval["obs"]}',
                ha='center', va='center', fontsize=max(5, fs - 2),
                color='0.15', zorder=5)

    # ── axis limits ───────────────────────────────────────────────────────────
    xs, ys = zip(*pos.values())
    pad = 2.5
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(min(ys) - pad, max(ys) + pad)
    ax.set_aspect('equal')
    ax.axis('off')


# ── Public API ────────────────────────────────────────────────────────────────

def plot_bowtie(block_dict, flux_dict, figsize=(7, 6)):
    """Draw three bowtie diagrams from the output of :func:`sam_bowtie.validate`.

    Parameters
    ----------
    block_dict : dict
        Keys = block labels (``'SCC'``, ``'IN'``, ``'OUT'``, …).
        Values = ``{'obs': int, 'count_sample': list, 'p_value': float}``.
    flux_dict : dict
        Keys = ``"src->tgt"`` strings.
        Values = ``{'obs': float, 'count_sample': list, 'p_value': float}``.
    figsize : tuple
        ``(width, height)`` in inches *per subplot*.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure with three side-by-side subplots:

        1. **Blocks** – circle size ∝ log(obs), circle colour = p-value.
        2. **Fluxes** – arrow width ∝ log(obs), arrow colour = p-value.
        3. **Combined** – both, with two separate colour scales.
    """
    blocks = list(block_dict.keys())
    pos    = _positions(blocks)
    rmap   = _radii(block_dict)
    lwmap  = _linewidths(flux_dict)
    norm   = mcolors.Normalize(vmin=0.0, vmax=1.0)
    bcmap  = plt.get_cmap(_BLOCK_CMAP)
    fcmap  = plt.get_cmap(_FLUX_CMAP)

    fig, axes = plt.subplots(1, 3, figsize=(figsize[0] * 3, figsize[1]))
    fig.subplots_adjust(wspace=0.30)

    # ── Panel 1: block sizes + block p-value colours ─────────────────────────
    _draw_scene(axes[0], block_dict, flux_dict, rmap, lwmap, pos,
                block_cmap=bcmap, flux_cmap=bcmap, norm=norm,
                show_block_color=True,  show_block_size=True,
                show_flux_color=False,  show_flux_size=False,
                neutral_lw=1.5)
    axes[0].set_title('Blocks\n(size ∝ log n, colour = p-value)', fontsize=9)
    plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=bcmap),
                 ax=axes[0], shrink=0.55, pad=0.02,
                 orientation='horizontal', label='p-value (blocks)')

    # ── Panel 2: flux widths + flux p-value colours ──────────────────────────
    _draw_scene(axes[1], block_dict, flux_dict, rmap, lwmap, pos,
                block_cmap=fcmap, flux_cmap=fcmap, norm=norm,
                show_block_color=False, show_block_size=False,
                show_flux_color=True,   show_flux_size=True,
                neutral_r=0.55)
    axes[1].set_title('Fluxes\n(width ∝ log flux, colour = p-value)', fontsize=9)
    plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=fcmap),
                 ax=axes[1], shrink=0.55, pad=0.02,
                 orientation='horizontal', label='p-value (fluxes)')

    # ── Panel 3: combined – two separate colour scales ───────────────────────
    _draw_scene(axes[2], block_dict, flux_dict, rmap, lwmap, pos,
                block_cmap=bcmap, flux_cmap=fcmap, norm=norm,
                show_block_color=True,  show_block_size=True,
                show_flux_color=True,   show_flux_size=True)
    axes[2].set_title('Combined\n(separate colour scales)', fontsize=9)
    plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=bcmap),
                 ax=axes[2], shrink=0.45, pad=0.02,
                 orientation='horizontal', label='p-value (blocks)')
    plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=fcmap),
                 ax=axes[2], shrink=0.45, pad=0.10,
                 orientation='horizontal', label='p-value (fluxes)')

    fig.tight_layout()
    return fig
