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
_BLOCK_CMAP = 'cool'
_FLUX_CMAP  = 'cool'

# Floor for LogNorm (avoids log(0) when a p-value is exactly 0)
_PVAL_FLOOR = 1e-6


def _fdr(p_vals_list, alpha):
    """Benjamini-Hochberg FDR threshold for *p_vals_list* at level *alpha*.

    Returns 0.0 when no hypothesis can be rejected.
    """
    p = np.sort(np.asarray(p_vals_list, dtype=float))
    m = len(p)
    if m == 0:
        return 0.0
    thresholds = np.arange(1, m + 1) * alpha / m
    below = p <= thresholds
    if not np.any(below):
        return 0.0
    return float(thresholds[np.max(np.where(below))])


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


def _log_norm(value_dicts, validated_keys=None, fdr_th=None):
    """Build a LogNorm for p-values in *value_dicts*.

    If *validated_keys* is provided, only those entries are used to set vmin.
    If all such p-values are 0 and *fdr_th* > 0, vmin falls back to the
    power of 10 just below *fdr_th*, so the colorbar has a meaningful range.
    The lower bound is always rounded down to the nearest power of 10.
    """
    if validated_keys is not None:
        pvals = [v['p_value']
                 for d in value_dicts
                 for k, v in d.items()
                 if k in validated_keys]
    else:
        pvals = [v['p_value'] for d in value_dicts for v in d.values()]
    pos_pvals = [p for p in pvals if p > 0.0]
    if pos_pvals:
        vmin = 10 ** np.floor(np.log10(min(pos_pvals)))
        vmin = max(_PVAL_FLOOR, vmin)
    elif fdr_th is not None and fdr_th > 0:
        # All validated p-values are 0: anchor range just below the FDR threshold
        vmin = 10 ** np.floor(np.log10(fdr_th))
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


def _draw_self_loop(ax, x, y, r, color, lw, arrow_border=False):
    """Small circular loop drawn above a block circle."""
    lr = r * 0.55
    cx, cy = x, y + r + lr
    theta = np.linspace(0.0, 2 * np.pi, 120)
    lx = cx + lr * np.cos(theta)
    ly = cy + lr * np.sin(theta)
    loop_line, = ax.plot(lx, ly, color=color, lw=lw, zorder=2, solid_capstyle='round')
    if arrow_border:
        loop_line.set_path_effects([pe.withStroke(linewidth=lw + 2.0, foreground='black')])
    tail_w = max(0.12, lw / 10.0)
    head_w = max(0.40, tail_w * 3.0)
    head_l = max(0.30, head_w * 0.65)
    astyle  = (f'simple,tail_width={tail_w:.3f},'
               f'head_width={head_w:.3f},head_length={head_l:.3f}')
    ann = ax.annotate('', xy=(lx[-1], ly[-1] - 0.01), xytext=(lx[-2], ly[-2]),
                arrowprops=dict(arrowstyle=astyle, color=color, lw=0,
                                mutation_scale=10),
                zorder=3)
    if arrow_border:
        arrow_patch = getattr(ann, 'arrow_patch', None)
        if arrow_patch is not None:
            arrow_patch.set_path_effects(
                [pe.withStroke(linewidth=2.0, foreground='black')])


# ── Core scene renderer ───────────────────────────────────────────────────────

def _draw_scene(ax, block_dict, obs_flux_dict, validated_flux_keys,
                radii, lws, pos,
                block_cmap, flux_cmap, block_norm, flux_norm,
                show_block_color, show_block_size,
                show_flux_color, show_flux_size,
                neutral_r=0.55, neutral_lw=2.0,
                neutral_block_color='0.82',
                neutral_arrow_color='black',
                unvalidated_color='0.70',
                validated_block_keys=None,
                arrow_border=False):
    """Draw one complete bowtie panel onto *ax*.

    Parameters
    ----------
    obs_flux_dict        : dict              – flux entries with obs > 0 only
    validated_flux_keys  : set               – flux keys whose p-value passes FDR threshold
    radii                : dict[str, float]  – circle radius per block
    lws                  : dict[str, float]  – arrow linewidth per flux key
    pos                  : dict[str, tuple]  – (x, y) position per block
    block_cmap           : Colormap          – colormap applied to validated block circles
    flux_cmap            : Colormap          – colormap applied to validated arrows
    block_norm           : Normalize         – norm (log-scale) for block p-values
    flux_norm            : Normalize         – norm (log-scale) for flux p-values
    show_block_color     : bool              – colour validated blocks by their p-value
    show_block_size      : bool              – size blocks by log(obs)
    show_flux_color      : bool              – colour validated arrows by p-value; others gray+behind
    show_flux_size       : bool              – size arrows by log(obs)
    neutral_r            : float             – uniform radius used when show_block_size=False
    neutral_lw           : float             – uniform linewidth used when show_flux_size=False
    neutral_block_color  : str               – fill colour for non-validated / uncoloured blocks
    neutral_arrow_color  : str               – arrow colour when show_flux_color=False
    unvalidated_color    : str               – arrow colour for non-validated observed fluxes
    validated_block_keys : set | None        – block keys to colour; None = colour all
    arrow_border         : bool              – if True, draw a thin black stroke around validated arrows
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
            _draw_self_loop(ax, pos[src][0], pos[src][1], r0, color, lw,
                            arrow_border=arrow_border)
        else:
            # Non-validated arrows go behind validated ones
            arrow_zorder = 1 if (show_flux_color and fkey not in validated_flux_keys) else 2
            x0, y0, x1, y1 = _offset_endpoints(
                pos[src][0], pos[src][1], pos[tgt][0], pos[tgt][1], r0, r1)
            tail_w = max(0.12, lw / 10.0)
            head_w = max(0.40, tail_w * 3.0)
            head_l = max(0.30, head_w * 0.65)
            astyle  = (f'simple,tail_width={tail_w:.3f},'
                       f'head_width={head_w:.3f},head_length={head_l:.3f}')
            ann = ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                              arrowprops=dict(
                                  arrowstyle=astyle, color=color, lw=0,
                                  mutation_scale=10,
                                  connectionstyle='arc3,rad=0.08'),
                              zorder=arrow_zorder)
            if arrow_border and arrow_zorder == 2:
                arrow_patch = getattr(ann, 'arrow_patch', None)
                if arrow_patch is not None:
                    arrow_patch.set_path_effects(
                        [pe.withStroke(linewidth=2.0, foreground='black')])

    # ── block circles (drawn after arrows so they sit on top) ─────────────────
    for b, bval in block_dict.items():
        x, y  = pos[b]
        r     = radii[b] if show_block_size else neutral_r
        is_validated = (validated_block_keys is None or b in validated_block_keys)
        if show_block_color and is_validated:
            color = block_cmap(block_norm(max(bval['p_value'], _PVAL_FLOOR)))
        else:
            color = neutral_block_color
        ax.add_patch(plt.Circle((x, y), r, color=color, ec='black', lw=0.9, zorder=4))
        fs = max(6, int(10 * r))
        ax.text(x, y + r * 0.18, b,
                ha='center', va='center', fontsize=fs, fontweight='bold', zorder=5,
                path_effects=[pe.withStroke(linewidth=2, foreground='white')])
        if bval.get('obs', 0) > 0:
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


def _add_colorbar(fig, ax, cmap, norm, label, fdr_th=None, shrink=0.65, pad=0.02):
    """Attach a horizontal colorbar with log-scale ticks and an optional FDR marker."""
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                      ax=ax, shrink=shrink, pad=pad,
                      orientation='horizontal', label=label)
    cb.ax.xaxis.set_major_formatter(plt.FuncFormatter(
        lambda x, _: f'{x:.0e}' if x < 0.01 else f'{x:.2f}'))
    if fdr_th is not None and fdr_th > 0:
        cb.ax.axvline(x=fdr_th, color='red', linestyle='--', linewidth=1.5)
    return cb


# ── Public API ────────────────────────────────────────────────────────────────

def plot_bowtie_blocks(block_dict, alpha=0.05, figsize=(7, 6)):
    """Draw a bowtie diagram coloured by block-level statistical validation.

    Parameters
    ----------
    block_dict : dict
        Keys = block labels (``'SCC'``, ``'IN'``, ``'OUT'``, …).
        Values = ``{'obs': int, 'p_value': float, ...}``.
    alpha : float
        Nominal FDR level. Only blocks that pass Benjamini-Hochberg FDR
        correction are coloured; the rest are shown in neutral gray.
    figsize : tuple
        ``(width, height)`` in inches.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    # ── FDR threshold
    p_vals_list = [v['p_value'] for v in block_dict.values()]
    fdr_th = _fdr(p_vals_list, alpha)
    validated_keys = {k for k, v in block_dict.items()
                      if fdr_th > 0 and v['p_value'] <= fdr_th}

    # ── Layout
    blocks = list(block_dict.keys())
    pos    = _positions(blocks)
    rmap   = _radii(block_dict)

    # ── Norm (from validated p-values only; fallback to fdr_th if all are 0)
    bnorm = _log_norm([block_dict], validated_keys=validated_keys, fdr_th=fdr_th)
    bcmap = plt.get_cmap(_BLOCK_CMAP)

    fig, ax = plt.subplots(figsize=figsize)
    _draw_scene(ax, block_dict, obs_flux_dict={}, validated_flux_keys=set(),
                radii=rmap, lws={}, pos=pos,
                block_cmap=bcmap, flux_cmap=bcmap, block_norm=bnorm, flux_norm=bnorm,
                show_block_color=True, show_block_size=True,
                show_flux_color=False, show_flux_size=False,
                neutral_r=0.55, neutral_lw=2.0,
                neutral_block_color='0.82',
                validated_block_keys=validated_keys,
                arrow_border=False)
    ax.set_title(f'Blocks  (size ∝ log n, colour = p-value, FDR α={alpha})', fontsize=10)
    _add_colorbar(fig, ax, bcmap, bnorm, 'p-value (blocks)',
                  fdr_th=fdr_th if fdr_th > 0 else None)
    fig.tight_layout()
    return fig


def plot_bowtie_fluxes(flux_dict, alpha=0.05, figsize=(7, 6)):
    """Draw a bowtie diagram coloured by flux-level statistical validation.

    Parameters
    ----------
    flux_dict : dict
        Keys = ``"src->tgt"`` strings.
        Values = ``{'obs': float, 'p_value': float, ...}``.
    alpha : float
        Nominal FDR level. Only fluxes that pass Benjamini-Hochberg FDR
        correction are coloured and drawn in front; others are gray and behind.
    figsize : tuple
        ``(width, height)`` in inches.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    obs_flux = {k: v for k, v in flux_dict.items() if v['obs'] > 0}

    # ── FDR threshold (computed over observed fluxes only)
    p_vals_list = [v['p_value'] for v in obs_flux.values()]
    fdr_th = _fdr(p_vals_list, alpha) if p_vals_list else 0.0
    validated_keys = {k for k, v in obs_flux.items()
                      if fdr_th > 0 and v['p_value'] <= fdr_th}

    # ── Extract blocks from flux keys; dummy block_dict for positioning only
    all_blocks = set()
    for key in obs_flux:
        if type(key) is tuple:
            src, tgt = key
        else:
            src, tgt = key.split('->')
        all_blocks.update([src, tgt])
    dummy_block_dict = {b: {'obs': 0, 'p_value': 1.0} for b in all_blocks}

    # ── Layout
    neutral_r = 0.55
    pos   = _positions(list(all_blocks))
    lwmap = _linewidths(obs_flux)
    rmap  = {b: neutral_r for b in all_blocks}

    # ── Norm (from validated flux p-values only; fallback to fdr_th if all are 0)
    fnorm = _log_norm([obs_flux], validated_keys=validated_keys, fdr_th=fdr_th)
    fcmap = plt.get_cmap(_FLUX_CMAP)

    fig, ax = plt.subplots(figsize=figsize)
    _draw_scene(ax, dummy_block_dict, obs_flux_dict=obs_flux,
                validated_flux_keys=validated_keys,
                radii=rmap, lws=lwmap, pos=pos,
                block_cmap=fcmap, flux_cmap=fcmap, block_norm=fnorm, flux_norm=fnorm,
                show_block_color=False, show_block_size=False,
                show_flux_color=True, show_flux_size=True,
                neutral_r=neutral_r, neutral_lw=2.0,
                neutral_block_color='white',
                validated_block_keys=None,
                arrow_border=True)
    ax.set_title(f'Fluxes  (width ∝ log flux, FDR α={alpha})', fontsize=10)
    _add_colorbar(fig, ax, fcmap, fnorm, 'p-value (fluxes)',
                  fdr_th=fdr_th if fdr_th > 0 else None)
    fig.tight_layout()
    return fig
