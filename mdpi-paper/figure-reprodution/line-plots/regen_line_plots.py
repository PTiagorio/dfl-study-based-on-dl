#!/usr/bin/env python3
"""
Regenerates the 16 result figures from the tables in template.tex.
- 600 dpi (minimum recommended by MDPI)
- colorblind-safe palette (all tests PASS)
- distinct markers and line styles -> also readable in black and white
- all figures are exported with exactly the same dimensions
"""

import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter


# ---------------------------------------------------------------------
# GEOMETRY
# ---------------------------------------------------------------------

# Original reference width:
TEXTWIDTH_IN = 394.35522 / 72.27
W = 0.8 * TEXTWIDTH_IN          # 4.366 in
H = W / 1.618                   # original aspect ratio

# Fixed final canvas for ALL figures.
#
# It is deliberately wider than W to provide enough space
# for longer legends, such as:
#
# Unpoisoned | 1 Poisoned | 2 Poisoned | 3 Poisoned | 4 Poisoned
#
# All 16 figures will have exactly these dimensions.
FIG_W = 5.6
FIG_H = 3.25

DPI = 660


# ---------------------------------------------------------------------
# SCALE COMPENSATION
# ---------------------------------------------------------------------
#
# The canvas has a width of FIG_W, but \includegraphics places the figure
# at 0.8\textwidth (= W). In other words, everything is scaled down by W/FIG_W = 0.78 when
# inserted into the paper: text authored at 8 pt would appear at 6.2 pt, smaller
# than in the figures from the original submission. Since Reviewer 3's complaint was
# specifically about the readability of the figure text, this would be undesirable.
#
# Solution: author everything multiplied by SCALE so that, after scaling down,
# it remains at the intended size. The file size does not change (it remains
# FIG_W x FIG_H); only the content is drawn larger.
#
# The subplots_adjust margins are fractions of the canvas, so they are
# invariant to scaling and do not need to be changed.

SCALE = FIG_W / W               # 1.282

FS_LABEL = 8.0 * SCALE          # axis labels        -> 8.0 pt in the paper
FS_TICK = 7.5 * SCALE           # axis tick labels   -> 7.5 pt in the paper
FS_LEGEND = 7.5 * SCALE         # legend             -> 7.5 pt in the paper

LW_SERIES = 1.4 * SCALE         # line width
MS_MARKER = 3.6 * SCALE         # marker size
LW_AXES = 0.6 * SCALE           # axes width
LW_GRID = 0.5 * SCALE           # grid width
LW_MARKER_EDGE = 0.5 * SCALE    # white marker edge


# ---------------------------------------------------------------------
# PALETTE
# ---------------------------------------------------------------------

PALETTE = [
    "#009E73",
    "#0072B2",
    "#8C6D00",
    "#A64D79",
    "#D55E00"
]

MARKERS = [
    "o",
    "s",
    "^",
    "D",
    "v"
]

LINESTYLES = [
    (0, ()),
    (0, (5, 1.5)),
    (0, (1, 1.2)),
    (0, (6, 1.5, 1, 1.5)),
    (0, (3, 1, 1, 1, 1, 1))
]


INK = "#1a1a1a"
MUTED = "#4d4d4d"
GRID = "#d9d9d9"


plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": FS_LABEL,
    "axes.labelsize": FS_LABEL,
    "xtick.labelsize": FS_TICK,
    "ytick.labelsize": FS_TICK,
    "legend.fontsize": FS_LEGEND,
    "axes.edgecolor": MUTED,
    "axes.linewidth": LW_AXES,
    "text.color": INK,
    "axes.labelcolor": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "figure.dpi": DPI,
    "savefig.dpi": DPI,
})


def num(s):
    s = s.strip()

    try:
        return float(s)

    except ValueError:
        return None


def make_chart(
    path,
    x,
    xlabel,
    series,
    ylim=(0.200, 1.000),
    flat_first=False
):
    """
    series:
        list of (name, [values aligned with x])

    All figures use exactly:
        - the same canvas
        - the same axes position
        - the same plotting area
        - the same space reserved for the legend
    """

    # --------------------------------------------------------------
    # FIXED CANVAS
    # --------------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(FIG_W, FIG_H)
    )

    # --------------------------------------------------------------
    # FIXED LAYOUT
    # --------------------------------------------------------------
    #
    # We do not use tight_layout().
    #
    # This ensures that a larger legend DOES NOT change the size
    # of the chart.
    #
    # left / right / bottom / top are identical in all figures.
    #
    # The region above top=0.76 is reserved for the legend.
    #

    # left/bottom were increased relative to the original values (0.12 / 0.17) because
    # scale compensation increased the font sizes and the y-axis label started to
    # touch the left edge. They are identical across all 16 figures, so the
    # geometry remains identical across them.
    fig.subplots_adjust(
        left=0.155,
        right=0.98,
        bottom=0.20,
        top=0.84
    )


    # --------------------------------------------------------------
    # GRID / SPINES
    # --------------------------------------------------------------

    ax.set_axisbelow(True)

    ax.yaxis.grid(
        True,
        color=GRID,
        linewidth=LW_GRID
    )

    ax.xaxis.grid(False)

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


    # --------------------------------------------------------------
    # SERIES
    # --------------------------------------------------------------

    for i, (name, ys) in enumerate(series):

        c = PALETTE[i % len(PALETTE)]

        pts = [
            (xx, yy)
            for xx, yy in zip(x, ys)
            if yy is not None
        ]

        if not pts:
            continue


        # ----------------------------------------------------------
        # Horizontal baseline
        # ----------------------------------------------------------

        if flat_first and i == 0 and len(pts) == 1:

            ax.axhline(
                pts[0][1],
                color=c,
                linewidth=LW_SERIES,
                linestyle=LINESTYLES[i % len(LINESTYLES)],
                zorder=3
            )

            # Dummy plot for the legend
            ax.plot(
                [],
                [],
                color=c,
                linewidth=LW_SERIES,
                linestyle=LINESTYLES[i % len(LINESTYLES)],
                label=name
            )

            continue


        # ----------------------------------------------------------
        # Normal series
        # ----------------------------------------------------------

        xs, ys2 = zip(*pts)

        ax.plot(
            xs,
            ys2,
            color=c,
            linewidth=LW_SERIES,
            linestyle=LINESTYLES[i % len(LINESTYLES)],
            marker=MARKERS[i % len(MARKERS)],
            markersize=MS_MARKER,
            markeredgecolor="white",
            markeredgewidth=LW_MARKER_EDGE,
            label=name,
            zorder=3,
            clip_on=False
        )


    # --------------------------------------------------------------
    # AXES
    # --------------------------------------------------------------

    ax.set_xlabel(xlabel)

    ax.set_ylabel(
        "Weighted F1-score"
    )

    ax.set_ylim(*ylim)

    ax.yaxis.set_major_formatter(
        FormatStrFormatter("%.3f")
    )


    # --------------------------------------------------------------
    # X TICKS
    # --------------------------------------------------------------

    ax.set_xticks(x)

    if all(
        abs(v - round(v)) < 1e-9
        for v in x
    ):

        ax.set_xticklabels([
            "%d" % round(v)
            for v in x
        ])

    else:

        ax.set_xticklabels([
            "%.2f" % v
            for v in x
        ])


    # --------------------------------------------------------------
    # LEGEND
    # --------------------------------------------------------------
    #
    # The legend always uses the fixed space above the chart.
    #
    # Since FIG_W is sufficiently wide, both:
    #
    # 1 Node | 3 Nodes | ...
    #
    # and:
    #
    # Unpoisoned | 1 Poisoned | ...
    #
    # should fit on a single line.
    #
    # NOTE: with scale compensation the fonts became larger, and the
    # 5-entry legend no longer fit with the previous spacing
    # (it required 6.35 in on a 5.6 in canvas). The spacing was
    # reduced to fit within 5.47 in, while keeping a single line and the
    # file dimensions unchanged.
    #

    # The legend is centered on the FIGURE, not on the axes: since the
    # left (0.12) and right (0.02) margins are different, the center of the axes
    # does not coincide with the center of the canvas, causing the legend to be clipped on the right.
    # y=0.90 in figure coordinates corresponds to the same position as the
    # previous 1.09 in axes coordinates.
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 0.90),
        bbox_transform=fig.transFigure,
        ncol=len(series),
        frameon=False,
        handlelength=1.5,
        columnspacing=0.8,
        handletextpad=0.35,
        borderaxespad=0.0
    )


    # --------------------------------------------------------------
    # SAVE
    # --------------------------------------------------------------

    os.makedirs(
        os.path.dirname(path),
        exist_ok=True
    )

    # IMPORTANT:
    #
    # DO NOT use:
    #
    #     bbox_inches="tight"
    #
    # because this would once again make the final size depend
    # on the legend content.
    #
    # We also DO NOT use tight_layout().
    #
    # Consequently, all images have exactly
    # FIG_W x FIG_H dimensions.

    fig.savefig(
        path,
        dpi=DPI,
        facecolor="white"
    )

    plt.close(fig)


# ---------------------------------------------------------------------
# INPUT
# ---------------------------------------------------------------------

T = json.load(
    open("tabs.json")
)

PAIRS = json.load(
    open("pairs.json")
)

OUT = "Figures_new"

made = []


# ---------------------------------------------------------------------
# GENERATE FIGURES
# ---------------------------------------------------------------------

for label, imgpath in PAIRS:

    rows = T[label]

    name = os.path.basename(
        imgpath
    )

    dest = os.path.join(
        OUT,
        os.path.dirname(imgpath).split("/", 1)[-1],
        name
    )


    # -----------------------------------------------------------------
    # NOISE
    # -----------------------------------------------------------------

    if "through-noise" in label:

        hdr = rows[0][1:]

        x = [
            num(r[0])
            for r in rows[1:]
        ]

        series = [
            (
                hdr[j].strip(),
                [
                    num(r[j + 1])
                    for r in rows[1:]
                ]
            )
            for j in range(len(hdr))
        ]

        make_chart(
            dest,
            x,
            "Noise Percentage Over Data",
            series
        )


    # -----------------------------------------------------------------
    # UNBALANCE
    # -----------------------------------------------------------------

    elif "through-unbalance" in label:

        hdr = rows[0][1:]

        x = [
            num(r[0])
            for r in rows[1:]
        ]

        series = [
            (
                hdr[j].strip(),
                [
                    num(r[j + 1])
                    for r in rows[1:]
                ]
            )
            for j in range(len(hdr))
        ]

        make_chart(
            dest,
            x,
            "Unbalance Level",
            series,
            flat_first=True
        )


    # -----------------------------------------------------------------
    # MINORITY POISONED
    # -----------------------------------------------------------------

    elif "minority-poisoned" in label:

        # Baseline:
        # 9 Nodes Without Poison
        base = num(
            rows[0][1]
        )

        # 1 Poisoned ... 4 Poisoned
        hdr = rows[1][1:]

        # F1 results
        body = rows[2:7]


        # -------------------------------------------------------------
        # IMPORTANT CHANGE:
        #
        # Previously:
        #
        # x = [0.00] + [...]
        #
        # Now we use ONLY the poisoning levels actually shown
        # in the experiment:
        #
        # 0.20, 0.40, 0.60, 0.80, 1.00
        # -------------------------------------------------------------

        x = [
            num(r[0])
            for r in body
        ]


        # -------------------------------------------------------------
        # Unpoisoned baseline
        # -------------------------------------------------------------

        series = [
            (
                "Unpoisoned",
                [base]
            )
        ]


        # -------------------------------------------------------------
        # 1 ... 4 Poisoned
        # -------------------------------------------------------------

        series += [
            (
                hdr[j].strip(),
                [
                    num(r[j + 1])
                    for r in body
                ]
            )
            for j in range(len(hdr))
        ]


        make_chart(
            dest,
            x,
            "Noise Percentage Over Poisoned Nodes",
            series,
            flat_first=True
        )


    else:
        continue


    made.append(dest)


print(
    f"{len(made)} figures generated in {OUT}/"
)