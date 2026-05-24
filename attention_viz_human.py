"""Interactive side-by-side attention visualisation (Bokeh backend)."""

import numpy as np


def to_rgba_uint32(image: np.ndarray) -> np.ndarray:
    """Return (H, W) uint32 RGBA array that Bokeh's image_rgba expects."""
    if image.dtype != np.uint8:
        raise ValueError(f"Expected uint8 image; got dtype {image.dtype}")

    if image.ndim == 2:
        h, w = image.shape
        rgba8 = np.empty((h, w, 4), dtype=np.uint8)
        rgba8[..., 0] = image
        rgba8[..., 1] = image
        rgba8[..., 2] = image
        rgba8[..., 3] = 255
    elif image.ndim == 3 and image.shape[2] == 3:
        h, w, _ = image.shape
        rgba8 = np.empty((h, w, 4), dtype=np.uint8)
        rgba8[..., :3] = image
        rgba8[..., 3] = 255
    elif image.ndim == 3 and image.shape[2] == 4:
        rgba8 = np.ascontiguousarray(image)
    else:
        raise ValueError(f"Unsupported image shape {image.shape}")

    return rgba8.view(dtype=np.uint32).reshape(rgba8.shape[:2])


def _grid_sources(grid_h, grid_w, x_off, y_off, cell_w, cell_h):
    total_w = grid_w * cell_w
    total_h = grid_h * cell_h

    vert_x0, vert_x1, vert_y0, vert_y1 = [], [], [], []
    for j in range(grid_w + 1):
        x = x_off + j * cell_w
        vert_x0.append(x); vert_x1.append(x)
        vert_y0.append(y_off); vert_y1.append(y_off + total_h)

    horz_x0, horz_x1, horz_y0, horz_y1 = [], [], [], []
    for i in range(grid_h + 1):
        y = y_off + i * cell_h
        horz_x0.append(x_off); horz_x1.append(x_off + total_w)
        horz_y0.append(y);     horz_y1.append(y)

    return (
        dict(x0=vert_x0, x1=vert_x1, y0=vert_y0, y1=vert_y1),
        dict(x0=horz_x0, x1=horz_x1, y0=horz_y0, y1=horz_y1),
    )


def _quad_source(grid_h, grid_w, x_off, y_off, cell_w, cell_h, img_label):
    x, y, w, h, labels = [], [], [], [], []
    for i in range(grid_h):
        for j in range(grid_w):
            x.append(x_off + (j + 0.5) * cell_w)
            y.append(y_off + (i + 0.5) * cell_h)
            w.append(cell_w)
            h.append(cell_h)
            labels.append(f"{img_label} [{i},{j}]")
    return dict(x=x, y=y, width=w, height=h, label=labels)


def attention_lol(
    attention: np.ndarray,
    img1: np.ndarray,
    img2: np.ndarray,
    *,
    threshold: float = 0.3,
    gap_px: int = 30,
    title: str = "Attention Relevance",
    frame_height: int = 500,
    curve_bend: float = 0.18,
    max_lines: int = 20000,
    heatmap_height: int = 100,
    scale: float = 1.0,
    logistic_steepness: float = 1.0,
):
    """Side-by-side attention viz. Images scaled to common display height."""
    frame_height = int(round(frame_height * scale))
    heatmap_height = int(round(heatmap_height * scale))
    if attention.ndim != 4:
        raise ValueError(
            f"attention must be 4-D (H1,W1,H2,W2); got shape {attention.shape}"
        )
    H1, W1, H2, W2 = attention.shape

    img1_rgba = to_rgba_uint32(img1)
    img2_rgba = to_rgba_uint32(img2)

    img1_h, img1_w = img1_rgba.shape
    img2_h, img2_w = img2_rgba.shape

    # Scale taller image down so both share min height.
    common_h = float(min(img1_h, img2_h))
    scale1 = common_h / img1_h
    scale2 = common_h / img2_h

    disp_w1 = img1_w * scale1
    disp_w2 = img2_w * scale2
    disp_h  = common_h

    x_off1 = 0.0
    x_off2 = disp_w1 + float(gap_px)
    total_w = disp_w1 + float(gap_px) + disp_w2
    total_h = disp_h

    cell_w1, cell_h1 = disp_w1 / W1, disp_h / H1
    cell_w2, cell_h2 = disp_w2 / W2, disp_h / H2

    from bokeh.plotting import figure
    from bokeh.models import (ColumnDataSource, CustomJS, Div, Slider,
                              CheckboxGroup, Button, Range1d, TapTool,
                              RadioGroup, LinearColorMapper, ColorBar)
    from bokeh.layouts import column, row, gridplot
    from bokeh.io import show, output_notebook, state
    if not state.curstate().notebook:
        output_notebook()

    # Frame width derived from data aspect so 1 X-unit == 1 Y-unit on screen.
    frame_width = int(round(frame_height * total_w / total_h))

    fig = figure(
        title=title,
        x_range=Range1d(start=0, end=total_w),
        y_range=Range1d(start=0, end=total_h),
        tools="reset",
        toolbar_location="above",
        frame_width=frame_width,
        frame_height=frame_height,
        match_aspect=True,
        output_backend="webgl",
    )
    fig.xaxis.visible = False
    fig.yaxis.visible = False
    fig.xgrid.visible = False
    fig.ygrid.visible = False
    fig.toolbar.logo = None

    fig.image_rgba(
        image=[img1_rgba], x=x_off1, y=0,
        dw=disp_w1, dh=disp_h,
    )
    fig.image_rgba(
        image=[img2_rgba], x=x_off2, y=0,
        dw=disp_w2, dh=disp_h,
    )

    v1, h1 = _grid_sources(H1, W1, x_off1, 0, cell_w1, cell_h1)
    v2, h2 = _grid_sources(H2, W2, x_off2, 0, cell_w2, cell_h2)

    def _merge(a, b):
        return {k: a[k] + b[k] for k in a}

    grid_vert = fig.segment(
        x0="x0", y0="y0", x1="x1", y1="y1",
        source=ColumnDataSource(_merge(v1, v2)),
        color="black", line_dash="dashed", line_width=1.5, alpha=0.85,
    )
    grid_horz = fig.segment(
        x0="x0", y0="y0", x1="x1", y1="y1",
        source=ColumnDataSource(_merge(h1, h2)),
        color="black", line_dash="dashed", line_width=1.5, alpha=0.85,
    )
    grid_vert.visible = False
    grid_horz.visible = False

    # -- attention bezier curves --------------------------------------------
    # Modified logistic: tanh maps (-inf, +inf) -> (-1, +1) smoothly.
    mapped = np.tanh(logistic_steepness * np.asarray(attention, dtype=np.float64))

    # Cell centres
    ccx1 = x_off1 + (np.arange(W1) + 0.5) * cell_w1
    ccy1 = (np.arange(H1) + 0.5) * cell_h1
    ccx2 = x_off2 + (np.arange(W2) + 0.5) * cell_w2
    ccy2 = (np.arange(H2) + 0.5) * cell_h2

    ii1, jj1, ii2, jj2 = np.meshgrid(
        np.arange(H1), np.arange(W1), np.arange(H2), np.arange(W2),
        indexing="ij",
    )
    ii1_f = ii1.ravel(); jj1_f = jj1.ravel()
    ii2_f = ii2.ravel(); jj2_f = jj2.ravel()
    x0 = ccx1[jj1_f]
    y0 = ccy1[ii1_f]
    x1 = ccx2[jj2_f]
    y1 = ccy2[ii2_f]
    m  = mapped[ii1_f, jj1_f, ii2_f, jj2_f]
    am = np.abs(m)

    if am.size > max_lines:
        keep = np.argpartition(-am, max_lines - 1)[:max_lines]
        x0, y0, x1, y1 = x0[keep], y0[keep], x1[keep], y1[keep]
        m, am = m[keep], am[keep]
        ii1_f, jj1_f = ii1_f[keep], jj1_f[keep]
        ii2_f, jj2_f = ii2_f[keep], jj2_f[keep]

    # Bezier control points: offset perpendicular to chord for slight bend.
    dx, dy = (x1 - x0), (y1 - y0)
    length = np.hypot(dx, dy)
    length[length == 0] = 1.0
    px, py = -dy / length, dx / length
    bend = curve_bend * length
    mx, my = (x0 + x1) * 0.5, (y0 + y1) * 0.5
    cxA = mx + px * bend * 0.5 + dx * -0.15
    cyA = my + py * bend * 0.5 + dy * -0.15
    cxB = mx + px * bend * 0.5 + dx *  0.15
    cyB = my + py * bend * 0.5 + dy *  0.15

    init_alpha = np.where(am >= threshold, am, 0.0)
    # Width: 1..10 px scaled by |mapped|. Outline: +2.5 px halo.
    line_w    = 1.0 + am * 9.0
    outline_w = line_w + 2.5
    pos_mask = m >= 0
    neg_mask = ~pos_mask

    def _make_src(mask):
        f32 = np.float32
        return ColumnDataSource(dict(
            x0=x0[mask].astype(f32), y0=y0[mask].astype(f32),
            x1=x1[mask].astype(f32), y1=y1[mask].astype(f32),
            cx0=cxA[mask].astype(f32), cy0=cyA[mask].astype(f32),
            cx1=cxB[mask].astype(f32), cy1=cyB[mask].astype(f32),
            abs_mapped=am[mask].astype(f32),
            line_alpha=init_alpha[mask].astype(f32),
            line_width=line_w[mask].astype(f32),
            outline_width=outline_w[mask].astype(f32),
            i1=ii1_f[mask].astype(np.int32),
            j1=jj1_f[mask].astype(np.int32),
            i2=ii2_f[mask].astype(np.int32),
            j2=jj2_f[mask].astype(np.int32),
        ))

    pos_src = _make_src(pos_mask)
    neg_src = _make_src(neg_mask)

    # Black halo first (drawn beneath), colored core on top.
    pos_outline = fig.bezier(
        x0="x0", y0="y0", x1="x1", y1="y1",
        cx0="cx0", cy0="cy0", cx1="cx1", cy1="cy1",
        source=pos_src,
        line_color="black", line_alpha="line_alpha",
        line_width="outline_width", line_cap="round",
    )
    neg_outline = fig.bezier(
        x0="x0", y0="y0", x1="x1", y1="y1",
        cx0="cx0", cy0="cy0", cx1="cx1", cy1="cy1",
        source=neg_src,
        line_color="black", line_alpha="line_alpha",
        line_width="outline_width", line_cap="round",
    )
    pos_glyph = fig.bezier(
        x0="x0", y0="y0", x1="x1", y1="y1",
        cx0="cx0", cy0="cy0", cx1="cx1", cy1="cy1",
        source=pos_src,
        line_color="#ff2020", line_alpha="line_alpha",
        line_width="line_width", line_cap="round",
    )
    neg_glyph = fig.bezier(
        x0="x0", y0="y0", x1="x1", y1="y1",
        cx0="cx0", cy0="cy0", cx1="cx1", cy1="cy1",
        source=neg_src,
        line_color="#1f77ff", line_alpha="line_alpha",
        line_width="line_width", line_cap="round",
    )
    # Default: only positives shown.
    neg_glyph.visible = False
    neg_outline.visible = False

    # -- tap (click) cells: two sources (one per image), multi-select toggle.
    def _tap_src(H, W, x_off, y_off, cw, ch, label):
        xs, ys, ws_, hs_, labs, iis, jjs = [], [], [], [], [], [], []
        for i in range(H):
            for j in range(W):
                xs.append(x_off + (j + 0.5) * cw)
                ys.append(y_off + (i + 0.5) * ch)
                ws_.append(cw); hs_.append(ch)
                labs.append(f"{label} [{i},{j}]")
                iis.append(i); jjs.append(j)
        return ColumnDataSource(dict(x=xs, y=ys, width=ws_, height=hs_,
                                     label=labs, i=iis, j=jjs))

    tap_src1 = _tap_src(H1, W1, x_off1, 0, cell_w1, cell_h1, "img1")
    tap_src2 = _tap_src(H2, W2, x_off2, 0, cell_w2, cell_h2, "img2")

    rect_kwargs = dict(
        x="x", y="y", width="width", height="height",
        fill_color="yellow", line_color="yellow",
        fill_alpha=0.0, line_alpha=0.0,
        selection_fill_alpha=0.35, selection_line_alpha=1.0,
        selection_line_color="yellow",
        nonselection_fill_alpha=0.0, nonselection_line_alpha=0.0,
        width_units="data", height_units="data",
    )
    fig.rect(source=tap_src1, **rect_kwargs)
    fig.rect(source=tap_src2, **rect_kwargs)

    fig.add_tools(TapTool(mode="toggle"))

    # -- heatmaps under each image -----------------------------------------
    f32 = np.float32
    # Flat per-cell value arrays. Rect glyph keyed by `value` field.
    mean_h1_f = mapped.mean(axis=(2, 3)).astype(f32).ravel()
    mean_h2_f = mapped.mean(axis=(0, 1)).astype(f32).ravel()
    max_h1_f  = mapped.max(axis=(2, 3)).astype(f32).ravel()
    max_h2_f  = mapped.max(axis=(0, 1)).astype(f32).ravel()

    # Cell centers in heatmap data coords (same x as main fig grid cells).
    h1_xs, h1_ys = [], []
    for i in range(H1):
        for j in range(W1):
            h1_xs.append(x_off1 + (j + 0.5) * cell_w1)
            h1_ys.append((i + 0.5) * (disp_h / H1))
    h2_xs, h2_ys = [], []
    for i in range(H2):
        for j in range(W2):
            h2_xs.append(x_off2 + (j + 0.5) * cell_w2)
            h2_ys.append((i + 0.5) * (disp_h / H2))

    heat1_src = ColumnDataSource(dict(x=h1_xs, y=h1_ys,
                                       value=mean_h1_f.tolist()))
    heat2_src = ColumnDataSource(dict(x=h2_xs, y=h2_ys,
                                       value=mean_h2_f.tolist()))
    # Precomputed full aggregations as flat lists.
    precomp_src = ColumnDataSource(dict(
        mean_h1=[mean_h1_f.tolist()], max_h1=[max_h1_f.tolist()],
        mean_h2=[mean_h2_f.tolist()], max_h2=[max_h2_f.tolist()],
    ))
    # Flat mapped tensor for JS partial aggregation.
    att_src = ColumnDataSource(dict(att=[mapped.astype(f32).ravel()]))

    # Custom diverging palette: full blue -> white -> full red.
    def _lerp_hex(c0, c1, t):
        r = round(c0[0] + (c1[0] - c0[0]) * t)
        g = round(c0[1] + (c1[1] - c0[1]) * t)
        b = round(c0[2] + (c1[2] - c0[2]) * t)
        return f"#{r:02x}{g:02x}{b:02x}"
    BLUE  = (0, 0, 255)
    WHITE = (255, 255, 255)
    RED   = (255, 0, 0)
    N = 11
    half = N // 2
    palette = []
    for k in range(half):
        palette.append(_lerp_hex(BLUE, WHITE, k / half))
    palette.append(_lerp_hex(WHITE, WHITE, 0))
    for k in range(1, half + 1):
        palette.append(_lerp_hex(WHITE, RED, k / half))
    heat_mapper = LinearColorMapper(palette=palette, low=-1.0, high=1.0)

    # Heatmap aspect ratio == image aspect ratio. Each heatmap has the same
    # display size as its image above → cells size = main-fig grid cells.
    dh1 = disp_h  # disp_w1 / (disp_w1/disp_h) == disp_h
    dh2 = disp_h
    y_span = disp_h
    heatmap_height = int(round(y_span * frame_height / total_h))
    y_off_h1 = 0
    y_off_h2 = 0

    fig_heat = figure(
        frame_width=frame_width, frame_height=heatmap_height,
        x_range=fig.x_range,
        y_range=Range1d(start=0, end=y_span),
        toolbar_location=None, tools="",
    )
    fig_heat.xaxis.visible = False; fig_heat.yaxis.visible = False
    fig_heat.xgrid.visible = False; fig_heat.ygrid.visible = False
    fig_heat.min_border_top = 0
    fig_heat.min_border_bottom = 0

    cell_h1_heat = disp_h / H1
    cell_h2_heat = disp_h / H2
    fig_heat.rect(
        x="x", y="y", width=cell_w1, height=cell_h1_heat,
        source=heat1_src,
        fill_color={"field": "value", "transform": heat_mapper},
        line_color=None, width_units="data", height_units="data",
    )
    fig_heat.rect(
        x="x", y="y", width=cell_w2, height=cell_h2_heat,
        source=heat2_src,
        fill_color={"field": "value", "transform": heat_mapper},
        line_color=None, width_units="data", height_units="data",
    )

    # ColorBar inside frame, centered in gap between heatmaps.
    # `location` tuple = pixel offset from frame's bottom-left (NOT data).
    gap_px_in_frame = gap_px * frame_width / total_w
    gap_center_x_px = int(round((x_off1 + disp_w1 + gap_px / 2) * frame_width / total_w))
    gap_start_x_px = int(round((x_off1 + disp_w1) * frame_width / total_w))
    bar_w_px = max(8, int(gap_px_in_frame * 0.5))
    bar_h_px = max(30, heatmap_height - 20)
    color_bar = ColorBar(
        color_mapper=heat_mapper,
        location=(gap_start_x_px,
                  (heatmap_height - bar_h_px) // 2),
        width=bar_w_px,
        height=bar_h_px,
        orientation="vertical",
        label_standoff=4,
    )
    fig_heat.add_layout(color_bar)

    # -- widgets -----------------------------------------------------------
    debug_div = Div(text="Tap cells (multi-select). Both images selectable.",
                    width=450, height=25,
                    styles={"font-family": "monospace"})

    thr_slider = Slider(start=0.0, end=1.0, step=0.01, value=threshold,
                        title="Threshold |tanh(attention)|", width=300)
    method_radio = RadioGroup(labels=["mean", "max"], active=0, inline=True,
                              width=140)
    sign_chk = CheckboxGroup(labels=["positive (red)", "negative (blue)"],
                             active=[0], width=180)
    grid_chk = CheckboxGroup(labels=["Show grid"], active=[], width=100)
    reset_btn = Button(label="Clear selection", width=140, button_type="warning")

    # -- shared JS: build masks, update lines, (optionally) heatmaps ------
    masks_code = """
        const sel1 = tap_src1.selected.indices;
        const sel2 = tap_src2.selected.indices;
        const has1 = sel1.length > 0;
        const has2 = sel2.length > 0;
        const m1 = new Uint8Array(H1 * W1);
        const m2 = new Uint8Array(H2 * W2);
        for (const k of sel1) m1[tap_src1.data.i[k] * W1 + tap_src1.data.j[k]] = 1;
        for (const k of sel2) m2[tap_src2.data.i[k] * W2 + tap_src2.data.j[k]] = 1;
    """

    # `thr` must be set by caller before this block runs.
    lines_code = """
        for (const src of [pos_src, neg_src]) {
            const am  = src.data.abs_mapped;
            const la  = src.data.line_alpha;
            const i1a = src.data.i1, j1a = src.data.j1;
            const i2a = src.data.i2, j2a = src.data.j2;
            for (let k = 0; k < am.length; k++) {
                let show = am[k] >= thr;
                if (show && has1) show = m1[i1a[k] * W1 + j1a[k]] === 1;
                if (show && has2) show = m2[i2a[k] * W2 + j2a[k]] === 1;
                la[k] = show ? am[k] : 0.0;
            }
            src.change.emit();
        }
    """

    heatmaps_code = """
        const useMax = method_radio.active === 1;
        const att = att_src.data.att[0];
        const H2W2 = H2 * W2;

        // heatmap1: per (i1,j1) aggregate over (i2,j2). No S2 -> precomp full.
        const v1 = heat1_src.data.value;
        if (!has2) {
            const full = useMax ? precomp_src.data.max_h1[0]
                                : precomp_src.data.mean_h1[0];
            for (let k = 0; k < v1.length; k++) v1[k] = full[k];
        } else {
            const sel2_lin = [];
            for (let kk = 0; kk < m2.length; kk++) {
                if (m2[kk] === 1) sel2_lin.push(kk);
            }
            const N2 = sel2_lin.length;
            for (let i1 = 0; i1 < H1; i1++) {
                for (let j1 = 0; j1 < W1; j1++) {
                    const base = (i1 * W1 + j1) * H2W2;
                    let v;
                    if (useMax) {
                        v = -Infinity;
                        for (let n = 0; n < N2; n++) {
                            const a = att[base + sel2_lin[n]];
                            if (a > v) v = a;
                        }
                    } else {
                        let s = 0.0;
                        for (let n = 0; n < N2; n++) s += att[base + sel2_lin[n]];
                        v = s / N2;
                    }
                    v1[i1 * W1 + j1] = v;
                }
            }
        }
        heat1_src.change.emit();

        // heatmap2: per (i2,j2) aggregate over (i1,j1). No S1 -> precomp full.
        const v2 = heat2_src.data.value;
        if (!has1) {
            const full = useMax ? precomp_src.data.max_h2[0]
                                : precomp_src.data.mean_h2[0];
            for (let k = 0; k < v2.length; k++) v2[k] = full[k];
        } else {
            const sel1_lin = [];
            for (let kk = 0; kk < m1.length; kk++) {
                if (m1[kk] === 1) sel1_lin.push(kk);
            }
            const N1 = sel1_lin.length;
            for (let i2 = 0; i2 < H2; i2++) {
                for (let j2 = 0; j2 < W2; j2++) {
                    const off22 = i2 * W2 + j2;
                    let v;
                    if (useMax) {
                        v = -Infinity;
                        for (let n = 0; n < N1; n++) {
                            const a = att[sel1_lin[n] * H2W2 + off22];
                            if (a > v) v = a;
                        }
                    } else {
                        let s = 0.0;
                        for (let n = 0; n < N1; n++) {
                            s += att[sel1_lin[n] * H2W2 + off22];
                        }
                        v = s / N1;
                    }
                    v2[i2 * W2 + j2] = v;
                }
            }
        }
        heat2_src.change.emit();
    """

    common_args = dict(
        thr_slider=thr_slider, method_radio=method_radio,
        tap_src1=tap_src1, tap_src2=tap_src2,
        pos_src=pos_src, neg_src=neg_src,
        att_src=att_src, precomp_src=precomp_src,
        heat1_src=heat1_src, heat2_src=heat2_src,
        H1=H1, W1=W1, H2=H2, W2=W2,
    )

    # Threshold: coarse update during drag (10 bins, skipped if unchanged),
    # exact update on release.
    thr_drag_code = """
        const exact = cb_obj.value;
        const coarse = Math.round(exact * 10) / 10;
        if (cb_obj._last_thr === coarse) return;
        cb_obj._last_thr = coarse;
        const thr = coarse;
    """ + masks_code + lines_code

    thr_release_code = """
        const thr = cb_obj.value;
        cb_obj._last_thr = thr;
    """ + masks_code + lines_code

    thr_slider.js_on_change("value", CustomJS(args=common_args, code=thr_drag_code))
    thr_slider.js_on_change("value_throttled",
                            CustomJS(args=common_args, code=thr_release_code))

    # Method radio affects only heatmaps.
    method_radio.js_on_change("active", CustomJS(
        args=common_args, code=masks_code + heatmaps_code))

    sel_summary = """
        const n1 = tap_src1.selected.indices.length;
        const n2 = tap_src2.selected.indices.length;
        div.text = "Selected: img1=" + n1 + " cell(s), img2=" + n2 + " cell(s).";
        const thr = thr_slider.value;
    """
    # Tap selections affect both lines and heatmaps.
    tap_src1.selected.js_on_change("indices", CustomJS(
        args=dict(div=debug_div, **common_args),
        code=sel_summary + masks_code + lines_code + heatmaps_code,
    ))
    tap_src2.selected.js_on_change("indices", CustomJS(
        args=dict(div=debug_div, **common_args),
        code=sel_summary + masks_code + lines_code + heatmaps_code,
    ))

    sign_chk.js_on_change("active", CustomJS(
        args=dict(pos=pos_glyph, neg=neg_glyph,
                  pos_o=pos_outline, neg_o=neg_outline),
        code="""
            const a = cb_obj.active;
            const showPos = a.indexOf(0) !== -1;
            const showNeg = a.indexOf(1) !== -1;
            pos.visible = showPos;   pos_o.visible = showPos;
            neg.visible = showNeg;   neg_o.visible = showNeg;
        """,
    ))

    grid_chk.js_on_change("active", CustomJS(
        args=dict(v=grid_vert, h=grid_horz),
        code="""
            const on = (cb_obj.active.length > 0);
            v.visible = on;
            h.visible = on;
        """,
    ))

    reset_btn.js_on_click(CustomJS(
        args=dict(div=debug_div, **common_args),
        code="""
            tap_src1.selected.indices = [];
            tap_src2.selected.indices = [];
            div.text = "Selection cleared.";
            const thr = thr_slider.value;
        """ + masks_code + lines_code + heatmaps_code,
    ))

    # gridplot aligns frames across rows regardless of border/colorbar widths.
    plots = gridplot([[fig], [fig_heat]],
                     toolbar_location="above", merge_tools=True)
    layout = column(
        plots,
        row(thr_slider, method_radio, sign_chk, grid_chk, reset_btn, debug_div),
    )
    show(layout)
    return layout
