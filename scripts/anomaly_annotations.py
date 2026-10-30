"""Consistent anomaly markers; interval times always use the original CSV clock."""
from io import BytesIO
from pathlib import Path
import matplotlib.patheffects as pe
import numpy as np


FAULT_COLOR = "#C51B7D"
RECOVERY_COLOR = "#008571"


def save_figure(fig, path, **kwargs):
    # Render first, then write once; avoids Pillow's w+b overwrite on Windows.
    buffer = BytesIO()
    fig.savefig(buffer, format="png", **kwargs)
    Path(path).write_bytes(buffer.getvalue())


def mark_time_interval(axes, interval, origin_s=0.0):
    """Shade [first bad sample, first recovered sample) on the requested clock."""
    start = interval["start_s"] - origin_s
    end = interval["recovery_s"] - origin_s
    for ax in np.asarray(axes, dtype=object).ravel():
        ax.axvspan(start, end, color=FAULT_COLOR, alpha=.17, zorder=0)
        ax.axvline(start, color=FAULT_COLOR, lw=1.6, zorder=4)
        ax.axvline(end, color=RECOVERY_COLOR, lw=1.6, ls="--", zorder=4)
        for value, marker, color in ((start, "^", FAULT_COLOR), (end, "s", RECOVERY_COLOR)):
            ax.plot(value, .975, marker=marker, color=color, ms=7,
                    markeredgecolor="white", transform=ax.get_xaxis_transform(), zorder=6)


def interval_trajectory(df, interval, columns):
    """Include interpolated boundary positions, independent of plot downsampling."""
    t = df["time_s"].to_numpy(dtype=float)
    start, end = interval["start_s"], interval["recovery_s"]
    if start < t[0] or end > t[-1]:
        raise ValueError("The marked interval must lie inside the trajectory time range")
    sample_times = np.r_[start, t[(t > start) & (t < end)], end]
    points = np.column_stack([np.interp(sample_times, t, df[c]) for c in columns])
    return sample_times, points


def mark_trajectory_interval(ax, df, interval, columns, labels=True):
    _, points = interval_trajectory(df, interval, columns)
    is_3d = len(columns) == 3
    if is_3d:
        ax.computed_zorder = False
    line, = ax.plot(*points.T, color=FAULT_COLOR, lw=3.4, zorder=10,
                    label="实飞外部姿态异常段")
    line.set_path_effects([pe.Stroke(linewidth=5.6, foreground="white"), pe.Normal()])
    for point, marker, color, label in (
        (points[0], "^", FAULT_COLOR, "异常开始"),
        (points[-1], "s", RECOVERY_COLOR, "外部姿态恢复"),
    ):
        ax.scatter(*point, marker=marker, c=color, s=95, edgecolors="white",
                   linewidths=1.2, zorder=12, label=label,
                   **({"depthshade": False} if is_3d else {}))
    if labels:
        if is_3d:
            from mpl_toolkits.mplot3d import proj3d
        for point, text, offset, color in (
            (points[0], f"异常开始\n{interval['start_s']:.6f} s", (20, 28), FAULT_COLOR),
            (points[-1], f"外部姿态恢复\n{interval['recovery_s']:.6f} s", (-18, -42), RECOVERY_COLOR),
        ):
            if is_3d:
                point = proj3d.proj_transform(*point, ax.get_proj())[:2]
                offset = (28, -40) if color == FAULT_COLOR else (-28, 40)
            ax.annotate(text, point, xytext=offset, textcoords="offset points",
                        ha="left" if offset[0] > 0 else "right", va="center",
                        color=color, fontsize=8.5, fontweight="bold", zorder=15,
                        bbox=dict(boxstyle="round,pad=.3", fc="white", ec=color, alpha=.95),
                        arrowprops=dict(arrowstyle="->", color=color, lw=1.1))


def finish_annotated_layout(fig, interval=None, origin_s=0.0, clock="原始日志时间", aggregate=False):
    """Reserve a header so interval labels do not cover curves or legends."""
    if interval is None:
        if not fig.get_constrained_layout():
            fig.tight_layout()
        return
    start, end = interval["start_s"], interval["recovery_s"]
    caption = (
        f"洋红色：实飞外部姿态异常 {end-start:.6f} s；▲ 异常开始；■ 外部姿态恢复（不代表 EKF 已恢复）\n"
        f"原始日志：{start:.6f}–{end:.6f} s"
    )
    if aggregate:
        caption = (f"本次实飞异常时间参考：{start:.6f}–{end:.6f} s；持续 {end-start:.6f} s\n"
                   "本图为整段统计；异常区间的变化请结合时序图查看")
    if origin_s:
        caption += f"  |  当前时间轴（{clock}）：{start-origin_s:.6f}–{end-origin_s:.6f} s"
    fig._suptitle.set_in_layout(False)
    fig._suptitle.set_y(.985)
    fig.text(.5, .942, caption, ha="center", va="top", fontsize=10, linespacing=1.6,
             bbox=dict(boxstyle="round,pad=.4", fc="#FCEAF4", ec=FAULT_COLOR, lw=.8))
    top = min(.875, 1 - 1.2 / fig.get_figheight())
    if fig.get_constrained_layout():
        fig.get_layout_engine().set(rect=(0, 0, 1, top))
    else:
        fig.tight_layout(rect=(0, 0, 1, top))
