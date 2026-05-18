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
    'INTENDRILS':  (-4.5,  3.5),
    'OUTTENDRILS': ( 4.5,  3.5),
    'OTHERS':      ( 0.0,  4.5),
}

# Default colormaps:
#   blocks → RdYlGn : p=0 (significant) = red, p=1 = green
#   fluxes → RdPu_r : p=0 (significant) = dark purple, p=1 = light
_BLOCK_CMAP = 'RdYlGn'
_FLUX_CMAP  = 'RdPu_r'

# Floor for LogNorm (avoids log(0) when a p-value is exactly 0)
_PVAL_FLOOR = 1e-6


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


def _log_norm(value_dicts):
    """Build a LogNorm spanning all p-values found in *value_dicts*.

    The lower bound is rounded down to the nearest power of 10 so that
    colourbar ticks land on clean values (e.g. 1e-4 instead of 3.2e-4).
    """
    pvals = [v['p_value'] for d in value_dicts for v in d.values()]
    pos_pvals = [p for p in pvals if p > 0.0]
    if pos_pvals:
        raw_min = min(pos_pvals)
        vmin = 10 ** np.floor(np.log10(raw_min))   # round down to power of 10
        vmin = max(_PVAL_FLOOR, vmin)
    else:
        vmin = _PVAL_FLOOR
    return mcolors.LogNorm(vmin=vmin, vmax=1.0)


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
    ax.plot(lx, ly, color=color, lw=lw, zorder=2, solid_capstyle='round')
    ax.annotate('', xy=(lx[-1], ly[-1] - 0.01), xytext=(lx[-2], ly[-2]),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                                mutation_scale=8 + lw * 0.5),
                zorder=3)


# ── Core scene renderer ───────────────────────────────────────────────────────

def _draw_scene(ax, block_dict, obs_flux_dict, validated_flux_keys,
                radii, lws, pos,
                block_cmap, flux_cmap, block_norm, flux_norm,
                show_block_color, show_block_size,
                show_flux_color, show_flux_size,
                neutral_r=0.55, neutral_lw=2.0,
                neutral_arrow_color='black',
                unvalidated_color='0.70'):
    """Draw one complete bowtie panel onto *ax*.

    Parameters
    ----------
    obs_flux_dict       : dict              – flux entries with obs > 0 only
    validated_flux_keys : set               – keys whose p-value passes the significance threshold
    radii               : dict[str, float]  – circle radius per block
    lws                 : dict[str, float]  – arrow linewidth per flux key
    pos                 : dict[str, tuple]  – (x, y) position per block
    block_cmap          : Colormap          – colormap applied to block circles
    flux_cmap           : Colormap          – colormap applied to validated arrows
    block_norm          : Normalize         – norm (log-scale) for block p-values
    flux_norm           : Normalize         – norm (log-scale) for flux p-values
    show_block_color    : bool              – colour blocks by their p-value
    show_block_size     : bool              – size blocks by log(obs)
    show_flux_color     : bool              – colour validated arrows by p-value; non-validated gray
    show_flux_size      : bool              – size arrows by log(obs)
    neutral_r           : float             – uniform radius used when show_block_size=False
    neutral_lw          : float             – uniform linewidth used when show_flux_size=False
    neutral_arrow_color : str               – arrow colour when show_flux_color=False
    unvalidated_color   : str               – arrow colour for non-validated observed fluxes
    """
    # ── arrows (drawn first, behind circles) ─────────────────────────────────
    for fkey, fval in obs_flux_dict.items():
        if type(fkey) is not tuple:
            src, tgt = fkey.split('->')
        else:
            src, tgt = fkey
        if src not in pos or tgt not in pos:
            continue
        lw = lws.get(fkey, neutral_lw) if show_flux_size else neutral_lw
        if not show_flux_color:
            color = neutral_arrow_color
        elif fkey in validated_flux_keys:
            color = flux_cmap(flux_norm(max(fval['p_value'], _PVAL_FLOOR)))
        else:
            color = unvalidated_color
        r0 = radii.get(src, neutral_r) if show_block_size else neutral_r
        r1 = radii.get(tgt, neutral_r) if show_block_size else neutral_r

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
                        zorder=2)

    # ── block circles (drawn after arrows so they sit on top) ─────────────────
    for b, bval in block_dict.items():
        x, y  = pos[b]
        r     = radii[b] if show_block_size else neutral_r
        color = block_cmap(block_norm(max(bval['p_value'], _PVAL_FLOOR))) if show_block_color else '0.82'
        ax.add_patch(plt.Circle((x, y), r, color=color, ec='black', lw=0.9, zorder=4))
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


def _add_colorbar(fig, ax, cmap, norm, label, shrink=0.65, pad=0.02):
    """Attach a horizontal colorbar with log-scale ticks."""
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                      ax=ax, shrink=shrink, pad=pad,
                      orientation='horizontal', label=label)
    cb.ax.xaxis.set_major_formatter(plt.FuncFormatter(
        lambda x, _: f'{x:.0e}' if x < 0.01 else f'{x:.2f}'))
    return cb


# ── Public API ────────────────────────────────────────────────────────────────

def plot_bowtie(block_dict, flux_dict, alpha=0.05, figsize=(7, 6)):
    """Draw three bowtie diagrams from the output of :func:`sam_bowtie.validate`.

    Parameters
    ----------
    block_dict : dict
        Keys = block labels (``'SCC'``, ``'IN'``, ``'OUT'``, …).
        Values = ``{'obs': int, 'count_sample': list, 'p_value': float}``.
    flux_dict : dict
        Keys = ``"src->tgt"`` strings.
        Values = ``{'obs': float, 'count_sample': list, 'p_value': float}``.
    alpha : float
        Significance threshold. Fluxes with p-value ≤ alpha are "validated"
        and coloured by their p-value in Figures 2 and 3; others are gray.
    figsize : tuple
        ``(width, height)`` in inches for *each* figure.

    Returns
    -------
    fig1, fig2, fig3 : matplotlib.figure.Figure
        1. **Blocks** – size ∝ log(obs), colour = p-value (log scale).
           All observed fluxes shown in black.
        2. **Fluxes** – width ∝ log(obs), validated fluxes coloured by
           p-value (log scale), non-validated in gray.
        3. **Combined** – both colour scales together.
    """
    blocks     = list(block_dict.keys())
    pos        = _positions(blocks)
    rmap       = _radii(block_dict)
    obs_flux   = {k: v for k, v in flux_dict.items() if v['obs'] > 0}
    lwmap      = _linewidths(obs_flux)
    valid_keys = {k for k, v in obs_flux.items() if v['p_value'] <= alpha}
    bnorm      = _log_norm([block_dict])
    fnorm      = _log_norm([obs_flux]) if obs_flux else bnorm
    bcmap      = plt.get_cmap(_BLOCK_CMAP)
    fcmap      = plt.get_cmap(_FLUX_CMAP)

    # Arguments shared across all three scenes
    common = dict(radii=rmap, lws=lwmap, pos=pos,
                  block_norm=bnorm, flux_norm=fnorm)

    # ── Figure 1: block sizes + block p-value colours; all observed fluxes in black
    fig1, ax1 = plt.subplots(figsize=figsize)
    _draw_scene(ax1, block_dict, obs_flux, validated_flux_keys=set(),
                block_cmap=bcmap, flux_cmap=bcmap,
                show_block_color=True, show_block_size=True,
                show_flux_color=False, show_flux_size=False,
                neutral_r=0.55, neutral_lw=1.5,
                neutral_arrow_color='black', **common)
    ax1.set_title('Blocks\n(size ∝ log n, colour = p-value)', fontsize=10)
    _add_colorbar(fig1, ax1, bcmap, bnorm, 'p-value (blocks)')
    fig1.tight_layout()

    # ── Figure 2: flux widths + p-value colours; non-validated observed fluxes in gray
    fig2, ax2 = plt.subplots(figsize=figsize)
    _draw_scene(ax2, block_dict, obs_flux, validated_flux_keys=valid_keys,
                block_cmap=fcmap, flux_cmap=fcmap,
                show_block_color=False, show_block_size=False,
                show_flux_color=True,  show_flux_size=True,
                neutral_r=0.55, neutral_lw=2.0, **common)
    ax2.set_title(f'Fluxes\n(width ∝ log flux, validated coloured [α={alpha}], non-validated gray)',
                  fontsize=10)
    _add_colorbar(fig2, ax2, fcmap, fnorm, 'p-value (fluxes)')
    fig2.tight_layout()

    # ── Figure 3: combined – block colours (RdYlGn) + flux colours (RdPu_r)
    fig3, ax3 = plt.subplots(figsize=figsize)
    _draw_scene(ax3, block_dict, obs_flux, validated_flux_keys=valid_keys,
                block_cmap=bcmap, flux_cmap=fcmap,
                show_block_color=True, show_block_size=True,
                show_flux_color=True,  show_flux_size=True,
                neutral_r=0.55, neutral_lw=2.0, **common)
    ax3.set_title('Combined\n(separate colour scales)', fontsize=10)
    _add_colorbar(fig3, ax3, bcmap, bnorm, 'p-value (blocks)', shrink=0.50, pad=0.02)
    _add_colorbar(fig3, ax3, fcmap, fnorm, 'p-value (fluxes)', shrink=0.50, pad=0.12)
    fig3.tight_layout()

    return fig1, fig2, fig3
