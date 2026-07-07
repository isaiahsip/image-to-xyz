"""
svg_image_to_3d_points.py

Converts SVG or raster image files (PNG, JPG, etc.) into a set of
(x, y, z) coordinate points lying on a flat plane (z = 0 by default)
in 3D space, and exports them as a standard .xyz file for import into
molecular design / visualization software (e.g. Rowan).

Extraction modes
-----------------
  "outline"   Trace only the edges/boundary of shapes.
              - SVG:    samples points along the path geometry itself (exact).
              - raster: Canny edge detection or contour tracing.

  "fill"      Fill the interior of shapes with a point grid.
              - SVG:    rasterized internally, then filled (see below).
              - raster: thresholds the image, then lays down a grid of
                        points at an adjustable spacing and keeps only
                        the ones that land inside the shape.
              Density is controlled by --spacing: smaller spacing = denser
              fill, larger spacing = sparser fill.

Color matching (--color-mode match, the default)
--------------------------------------------------
The actual color at each point's location in the source image/SVG is
sampled, then matched to the closest element in the standard "Jmol"
periodic table color scheme (the same scheme used by Rowan and most
other molecular viewers). Each point is written to the .xyz file as
that element. Since Rowan colors atoms by element automatically, the
resulting point cloud will visually reproduce the original image's
colors when opened there -- no manual coloring needed.

Use --color-mode single --element X to instead force every point to
the same element (the old, simpler behavior).

Output formats
---------------
  --format csv   Plain x,y,z CSV (no element/color info).
  --format xyz   Standard chemistry .xyz file (element symbol + x y z per
                 line). This is what you want for Rowan.
  --format both  Writes both files.

Usage examples
--------------
    # Filled shape, colors matched to nearest Jmol elements, ready for Rowan
    python svg_image_to_3d_points.py logo.svg --mode fill --spacing 2 \
        --format xyz --scale 0.1 --out molecule_template

    # Force every point to be the same element (e.g. all carbon)
    python svg_image_to_3d_points.py logo.svg --mode fill --color-mode single --element C

    # Just the outline of a raster image, colors matched per point
    python svg_image_to_3d_points.py photo.png --mode outline --raster-mode edges

Dependencies
------------
    pip install svgpathtools opencv-python-headless numpy matplotlib cairosvg
"""

import argparse
import csv
import os
import re
import tempfile

import numpy as np


# Standard "Jmol" element color scheme -- this is the same scheme used by
# Rowan Labs, Jmol, and many other molecular viewers. Verified pixel-by-pixel
# against a Rowan Labs periodic table screenshot (all 118 elements matched
# the reference Jmol values exactly). Elements past Hs with no assigned
# color in that scheme default to white.
ELEMENT_COLORS = {
    "H": "#ffffff", "He": "#d9ffff", "Li": "#cc80ff", "Be": "#c2ff00", "B": "#ffb5b5",
    "C": "#909090", "N": "#3050f8", "O": "#ff0d0d", "F": "#90e050", "Ne": "#b3e3f5",
    "Na": "#ab5cf2", "Mg": "#8aff00", "Al": "#bfa6a6", "Si": "#f0c8a0", "P": "#ff8000",
    "S": "#ffff30", "Cl": "#1ff01f", "Ar": "#80d1e3", "K": "#8f40d4", "Ca": "#3dff00",
    "Sc": "#e6e6e6", "Ti": "#bfc2c7", "V": "#a6a6ab", "Cr": "#8a99c7", "Mn": "#9c7ac7",
    "Fe": "#e06633", "Co": "#f090a0", "Ni": "#50d050", "Cu": "#c88033", "Zn": "#7d80b0",
    "Ga": "#c28f8f", "Ge": "#668f8f", "As": "#bd80e3", "Se": "#ffa100", "Br": "#a62929",
    "Kr": "#5cb8d1", "Rb": "#702eb0", "Sr": "#00ff00", "Y": "#94ffff", "Zr": "#94e0e0",
    "Nb": "#73c2c9", "Mo": "#54b5b5", "Tc": "#3b9e9e", "Ru": "#248f8f", "Rh": "#0a7d8c",
    "Pd": "#006985", "Ag": "#c0c0c0", "Cd": "#ffd98f", "In": "#a67573", "Sn": "#668080",
    "Sb": "#9e63b5", "Te": "#d47a00", "I": "#940094", "Xe": "#429eb0", "Cs": "#57178f",
    "Ba": "#00c900", "Hf": "#4dc2ff", "Ta": "#4da6ff", "W": "#2194d6", "Re": "#267dab",
    "Os": "#266696", "Ir": "#175487", "Pt": "#d0d0e0", "Au": "#ffd123", "Hg": "#b8b8d0",
    "Tl": "#a6544d", "Pb": "#575961", "Bi": "#9e4fb5", "Po": "#ab5c00", "At": "#754f45",
    "Rn": "#428296", "Fr": "#420066", "Ra": "#007d00", "Rf": "#cc0059", "Db": "#d1004f",
    "Sg": "#d90045", "Bh": "#e00038", "Hs": "#e6002e", "Mt": "#eb0026", "Ds": "#ffffff",
    "Rg": "#ffffff", "Cn": "#ffffff", "Nh": "#ffffff", "Fl": "#ffffff", "Mc": "#ffffff",
    "Lv": "#ffffff", "Ts": "#ffffff", "Og": "#ffffff", "La": "#70d4ff", "Ce": "#ffffc7",
    "Pr": "#d9ffc7", "Nd": "#c7ffc7", "Pm": "#a3ffc7", "Sm": "#8fffc7", "Eu": "#61ffc7",
    "Gd": "#45ffc7", "Tb": "#30ffc7", "Dy": "#1fffc7", "Ho": "#00ff9c", "Er": "#00e675",
    "Tm": "#00d452", "Yb": "#00bf38", "Lu": "#00ab24", "Ac": "#70abfa", "Th": "#00baff",
    "Pa": "#00a1ff", "U": "#008fff", "Np": "#0080ff", "Pu": "#006bff", "Am": "#545cf2",
    "Cm": "#785ce3", "Bk": "#8a4fe3", "Cf": "#a136d4", "Es": "#b31fd4", "Fm": "#b31fba",
    "Md": "#b30da6", "No": "#bd0d87", "Lr": "#c70066",
}

_ELEMENT_SYMBOLS = list(ELEMENT_COLORS.keys())
_ELEMENT_RGB = np.array([
    [int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16)] for h in ELEMENT_COLORS.values()
], dtype=float)


def get_element_color(symbol):
    """Look up the Jmol-scheme hex color for an element symbol, defaulting
    to a neutral gray if unrecognized."""
    return ELEMENT_COLORS.get(symbol, "#ff1493")


_RESTRICTED_ELEMENT_MAX_DIST = {
    # These elements have colors that are numerically close (in plain RGB
    # distance) to a much wider range of colors than they visually
    # resemble, so they tend to "steal" nearest-neighbor matches that
    # don't actually look like them. Only allow a match if the color is
    # genuinely close to the element's real color; otherwise fall through
    # to whatever the true next-best match is (found by re-running
    # nearest-neighbor with that element excluded).
    "Fr": 20,   # dark purple (#420066) -- was stealing generic dark grays
    "Ra": 20,   # green (#007d00) -- was stealing very dark, desaturated browns
}


def nearest_elements_for_colors(colors_rgb, black_threshold=15):
    """
    Vectorized nearest-neighbor match: for each (r, g, b) color (0-255),
    find the closest color in the Jmol element palette and return its
    element symbol.

    black_threshold : colors with all channels <= this value are treated
        as pure black and forced to Carbon ("C"), rather than nearest-
        matched.

    A few elements (see _RESTRICTED_ELEMENT_MAX_DIST) are only allowed to
    match colors genuinely close to their real color, since their color
    happens to be numerically close (in plain RGB distance) to a much
    wider range of colors than they visually resemble. If the best
    candidate is one of those but too far from its real color, it's
    excluded and the search repeats -- possibly excluding several
    restricted elements in a row -- until a valid match is found.

    colors_rgb : array-like of shape (N, 3)
    Returns: list of N element symbols.
    """
    colors_rgb = np.asarray(colors_rgb, dtype=float)
    if colors_rgb.ndim == 1:
        colors_rgb = colors_rgb.reshape(1, 3)

    diffs = colors_rgb[:, None, :] - _ELEMENT_RGB[None, :, :]
    dists = np.sum(diffs ** 2, axis=2)  # (N, num_elements)

    restricted_idx = {sym: _ELEMENT_SYMBOLS.index(sym) for sym in _RESTRICTED_ELEMENT_MAX_DIST}

    elements = []
    for i in range(colors_rgb.shape[0]):
        row = dists[i].copy()
        while True:
            best = int(np.argmin(row))
            symbol = _ELEMENT_SYMBOLS[best]
            if symbol in _RESTRICTED_ELEMENT_MAX_DIST:
                actual_dist = float(np.sqrt(dists[i, restricted_idx[symbol]]))
                if actual_dist > _RESTRICTED_ELEMENT_MAX_DIST[symbol]:
                    row[best] = np.inf
                    continue
            elements.append(symbol)
            break

    is_black = np.all(colors_rgb <= black_threshold, axis=1)
    for i in np.nonzero(is_black)[0]:
        elements[i] = "C"

    return elements


_NAMED_COLOR_FALLBACK = {"black": (0, 0, 0), "white": (255, 255, 255), "red": (255, 0, 0),
                          "green": (0, 128, 0), "blue": (0, 0, 255), "none": None}


def _parse_svg_color(color_str):
    """
    Parse an SVG color string (hex, named CSS color, or rgb(...)/rgba(...)
    functional notation) into an (r, g, b) tuple of 0-255 ints, or None if
    the color is 'none' / unparseable.
    """
    if not color_str or color_str.strip().lower() == "none":
        return None
    color_str = color_str.strip()

    m = re.match(r"rgba?\(\s*([\d.]+)%?\s*,\s*([\d.]+)%?\s*,\s*([\d.]+)%?", color_str)
    if m:
        return tuple(int(float(v)) for v in m.groups())

    try:
        import matplotlib.colors as mcolors
        r, g, b = mcolors.to_rgb(color_str)
        return (int(round(r * 255)), int(round(g * 255)), int(round(b * 255)))
    except Exception:
        return _NAMED_COLOR_FALLBACK.get(color_str.lower())


def _path_color(attributes):
    """
    Determine the effective color of an SVG path from its attributes,
    preferring 'fill' (and falling back to 'stroke' if fill is absent or
    'none'). Also checks a 'style' attribute for fill/stroke if present.
    Defaults to black, matching SVG's own default fill.
    """
    style = attributes.get("style", "")
    style_props = dict(
        p.split(":", 1) for p in style.split(";") if ":" in p
    ) if style else {}

    fill = style_props.get("fill", attributes.get("fill"))
    stroke = style_props.get("stroke", attributes.get("stroke"))

    rgb = _parse_svg_color(fill) if fill else None
    if rgb is None:
        rgb = _parse_svg_color(stroke) if stroke else None
    if rgb is None and fill is None and stroke is None:
        rgb = (0, 0, 0)  # SVG default fill is black
    return rgb


def _detect_background_color(color_img):
    """Assume the four corners of the image are background and average them."""
    h, w, _ = color_img.shape
    corners = np.array([color_img[0, 0], color_img[0, w - 1],
                         color_img[h - 1, 0], color_img[h - 1, w - 1]], dtype=float)
    return corners.mean(axis=0)


def _foreground_mask(color_img, color_threshold=30, invert=None):
    """
    Build a foreground mask using color distance from the detected
    background color, rather than plain grayscale brightness. This
    correctly separates multi-colored shapes from the background even
    when a shape's color happens to have similar brightness to the
    background (which a pure grayscale threshold would miss).

    invert : if True, flips the mask (selects the background-colored
    region instead). Leave as None for the normal case.
    """
    bg_color = _detect_background_color(color_img)
    diff = color_img.astype(float) - bg_color
    dist = np.sqrt((diff ** 2).sum(axis=2))
    mask = dist > color_threshold
    if invert:
        mask = ~mask
    return mask


def _resolve_invert(img, invert):
    """
    Decide whether the shape is the dark pixels or the light pixels.

    invert=None -> auto-detect by sampling the four corners of the image
    and assuming the corners are background. If the background is light,
    the shape is the dark pixels (so we need to invert the threshold to
    select them). If invert is explicitly True/False, that choice is
    respected as-is.
    """
    if invert is not None:
        return invert
    h, w = img.shape
    corners = [img[0, 0], img[0, w - 1], img[h - 1, 0], img[h - 1, w - 1]]
    background_is_light = (sum(int(c) for c in corners) / 4.0) > 127
    return background_is_light


# --------------------------------------------------------------------------
# SVG -> points (outline)
# --------------------------------------------------------------------------
def svg_outline_to_points(svg_path, samples_per_path=50, z=0.0):
    """
    Sample points along every path's geometry (exact vector outline).
    Returns (points, colors) where colors[i] is the (r, g, b) fill/stroke
    color of the path that point i came from.
    """
    from svgpathtools import svg2paths

    paths, attributes = svg2paths(svg_path)
    points = []
    colors = []
    for path, attrs in zip(paths, attributes):
        color = _path_color(attrs) or (0, 0, 0)

        if path.length() == 0:
            pt = path[0].start
            points.append((pt.real, pt.imag, z))
            colors.append(color)
            continue

        n = samples_per_path
        for i in range(n):
            t = i / (n - 1) if n > 1 else 0.0
            c = path.point(t)
            points.append((c.real, c.imag, z))
            colors.append(color)
    return points, colors


def _rasterize_svg(svg_path, out_png, resolution_scale=4.0):
    """Render an SVG to a PNG at a higher resolution so fill-sampling has
    enough pixel detail to work with."""
    import cairosvg

    # background_color is required: cairosvg defaults to a transparent
    # background, which collapses to black once flattened, making the
    # shape indistinguishable from the background.
    cairosvg.svg2png(url=svg_path, write_to=out_png, scale=resolution_scale,
                      background_color="white")


# --------------------------------------------------------------------------
# Raster image -> points
# --------------------------------------------------------------------------
def image_outline_to_points(image_path, raster_mode="edges", threshold1=50, threshold2=150,
                             color_threshold=30, invert=None, max_points=20000, z=0.0):
    """Extract boundary/edge points from a raster image. Returns (points, colors)."""
    import cv2

    color_img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if color_img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    height = color_img.shape[0]

    if raster_mode == "edges":
        gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, threshold1, threshold2)
        ys, xs = np.nonzero(edges)

    elif raster_mode == "contours":
        mask = _foreground_mask(color_img, color_threshold=color_threshold, invert=invert)
        binary = (mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        xs, ys = [], []
        for c in contours:
            for pt in c.reshape(-1, 2):
                xs.append(pt[0])
                ys.append(pt[1])
        xs, ys = np.array(xs), np.array(ys)

    else:
        raise ValueError(f"Unknown raster_mode: {raster_mode!r}. Use 'edges' or 'contours'.")

    n = len(xs)
    if n == 0:
        return [], []
    if n > max_points:
        idx = np.linspace(0, n - 1, max_points).astype(int)
        xs, ys = xs[idx], ys[idx]

    points = [(float(x), float(height - y), z) for x, y in zip(xs, ys)]
    bgr = color_img[ys, xs]  # Nx3 in BGR order
    colors = [(int(px[2]), int(px[1]), int(px[0])) for px in bgr]
    return points, colors


def _fill_points_from_mask(mask, color_img, spacing, z=0.0):
    """Sample a grid of points at the given pixel spacing, keeping only
    those that fall inside the foreground mask. Returns (points, colors)."""
    height, width = mask.shape
    spacing = max(1, int(spacing))
    ys_grid, xs_grid = np.mgrid[0:height:spacing, 0:width:spacing]

    keep = mask[ys_grid, xs_grid]
    xs = xs_grid[keep]
    ys = ys_grid[keep]

    points = [(float(x), float(height - y), z) for x, y in zip(xs, ys)]
    bgr = color_img[ys, xs]  # Nx3 in BGR order
    colors = [(int(px[2]), int(px[1]), int(px[0])) for px in bgr]
    return points, colors


def _auto_spacing_for_target(mask, target_atoms=10000, hard_cap=12000):
    """
    Pick a pixel spacing for fill sampling so the resulting point count
    lands near target_atoms, based on the actual foreground area (not
    total image size) -- so a small, mostly-empty image and a large,
    mostly-filled image both land near the same atom count.

    Then verify against the actual foreground pixels at that spacing and
    nudge upward if needed, guaranteeing the result never exceeds
    hard_cap regardless of how irregular the shape's coverage is.
    """
    fg_count = int(mask.sum())
    if fg_count == 0:
        return 1

    spacing = max(1, int(round((fg_count / target_atoms) ** 0.5)))

    def count_at(sp):
        h, w = mask.shape
        yg, xg = np.mgrid[0:h:sp, 0:w:sp]
        return int(mask[yg, xg].sum())

    while count_at(spacing) > hard_cap:
        spacing += 1

    return spacing


def image_fill_to_points(image_path, spacing=4, color_threshold=30, invert=None, z=0.0):
    """
    Fill the interior of shapes in a raster image with a grid of points.
    Returns (points, colors).

    spacing : int
        Distance in pixels between candidate grid points, BEFORE
        keeping only the ones that land inside the shape.
        Smaller spacing -> denser fill. Larger spacing -> sparser fill.
    """
    import cv2

    color_img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if color_img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    mask = _foreground_mask(color_img, color_threshold=color_threshold, invert=invert)
    return _fill_points_from_mask(mask, color_img, spacing, z=z)


# --------------------------------------------------------------------------
# Unified entry point
# --------------------------------------------------------------------------
def extract_points(input_path, mode="fill", raster_mode="contours", spacing=4,
                    samples_per_path=50, color_threshold=30, invert=None,
                    max_points=20000, z=0.0, svg_render_scale=4.0):
    """
    Top-level dispatcher. Auto-detects SVG vs raster from the file
    extension and routes to the right extraction function.

    Returns (points, colors) -- colors[i] is the (r, g, b) 0-255 color
    sampled at points[i]'s location in the source image/SVG.
    """
    is_svg = input_path.lower().endswith(".svg")

    if mode == "outline":
        if is_svg:
            return svg_outline_to_points(input_path, samples_per_path=samples_per_path, z=z)
        else:
            return image_outline_to_points(input_path, raster_mode=raster_mode,
                                            color_threshold=color_threshold, invert=invert,
                                            max_points=max_points, z=z)

    elif mode == "fill":
        if is_svg:
            # Rasterize the SVG first, then fill-sample it like any other image.
            with tempfile.TemporaryDirectory() as tmp:
                tmp_png = os.path.join(tmp, "rendered.png")
                _rasterize_svg(input_path, tmp_png, resolution_scale=svg_render_scale)
                return image_fill_to_points(tmp_png, spacing=spacing,
                                             color_threshold=color_threshold, invert=invert, z=z)
        else:
            return image_fill_to_points(input_path, spacing=spacing,
                                         color_threshold=color_threshold, invert=invert, z=z)

    else:
        raise ValueError(f"Unknown mode: {mode!r}. Use 'outline' or 'fill'.")


# --------------------------------------------------------------------------
# Output helpers
# --------------------------------------------------------------------------
def save_points_csv(points, out_path):
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", "z"])
        writer.writerows(points)


def save_points_xyz(points, out_path, elements, scale=1.0, comment=None):
    """
    Write points as a standard .xyz file:

        <number of atoms>
        <comment line>
        <element> <x> <y> <z>
        ...

    elements : str or list of str
        Either a single element symbol applied to every point, or a
        list of element symbols (one per point, e.g. from color matching).

    scale : float
        Multiplies every coordinate. Use this to convert from raw
        pixel units into whatever units your molecular design app
        expects (e.g. Angstroms).
    """
    if isinstance(elements, str):
        elements = [elements] * len(points)
    if len(elements) != len(points):
        raise ValueError("elements list must be the same length as points")

    if comment is None:
        comment = f"Generated point cloud, {len(points)} points, scale={scale}"

    with open(out_path, "w") as f:
        f.write(f"{len(points)}\n")
        f.write(f"{comment}\n")
        for (x, y, z), elem in zip(points, elements):
            f.write(f"{elem} {x * scale:.6f} {y * scale:.6f} {z * scale:.6f}\n")


def plot_points(points, elements, title="Extracted points"):
    import matplotlib.pyplot as plt

    if not points:
        print("No points to plot.")
        return

    xs, ys, zs = zip(*points)
    if isinstance(elements, str):
        elements = [elements] * len(points)
    point_colors = [get_element_color(e) for e in elements]

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(xs, ys, zs, s=4, c=point_colors, edgecolors="none")
    ax.set_facecolor("#1e1e1e")
    fig.patch.set_facecolor("#1e1e1e")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title, color="white")

    z_range = max(zs) - min(zs) if max(zs) != min(zs) else 1
    ax.set_box_aspect((max(xs) - min(xs) or 1, max(ys) - min(ys) or 1, z_range))

    out_file = title + "_plot.png"
    plt.savefig(out_file, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Saved plot to {out_file}")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Convert an SVG or image into flat 3D coordinate points / .xyz.")
    parser.add_argument("input", help="Path to input .svg / .png / .jpg / etc.")
    parser.add_argument("--mode", choices=["outline", "fill"], default="fill",
                         help="'outline' traces edges only, 'fill' fills the interior (default: fill).")
    parser.add_argument("--raster-mode", choices=["edges", "contours"], default="contours",
                         help="Outline extraction method for raster images (default: contours).")
    parser.add_argument("--spacing", type=int, default=None,
                         help="Grid spacing in pixels for fill mode -- lower = denser, higher = sparser. "
                              "Default: auto-computed from image size to hit --target-atoms.")
    parser.add_argument("--samples", type=int, default=50,
                         help="Points sampled per SVG path in outline mode (default: 50).")
    parser.add_argument("--svg-render-scale", type=float, default=4.0,
                         help="Upscaling factor when rasterizing an SVG for fill mode (default: 4.0).")
    parser.add_argument("--color-threshold", type=int, default=30,
                         help="Color distance from the detected background needed to count as foreground "
                              "(default: 30). Lower = more sensitive to subtle color shape differences.")
    parser.add_argument("--invert", dest="invert", action="store_true", default=None,
                         help="Force treating dark pixels as background (light shape on dark background).")
    parser.add_argument("--no-invert", dest="invert", action="store_false",
                         help="Force treating light pixels as background (dark shape on light background).")
    parser.add_argument("--z", type=float, default=0.0, help="Constant z value for all points (default: 0).")
    parser.add_argument("--max-points", type=int, default=None,
                         help="Cap on outline points. Default: auto-set to --target-atoms.")
    parser.add_argument("--target-atoms", type=int, default=10000,
                         help="Aim for roughly this many atoms when auto-sizing spacing (default: 10000).")
    parser.add_argument("--max-atoms", type=int, default=12000,
                         help="Hard cap on atom count -- auto-sizing will never exceed this (default: 12000).")
    parser.add_argument("--target-spacing", type=float, default=1.1,
                         help="Desired distance between neighboring atoms in output coordinate units, used "
                              "when auto-computing --scale (default: 1.1).")
    parser.add_argument("--format", choices=["csv", "xyz", "both"], default="xyz",
                         help="Output file format (default: xyz).")
    parser.add_argument("--color-mode", choices=["match", "single"], default="match",
                         help="'match' assigns each point the nearest Jmol element to its sampled image color "
                              "(default -- reproduces the image's colors in Rowan). "
                              "'single' forces every point to --element.")
    parser.add_argument("--element", default="C",
                         help="Element symbol used for every point when --color-mode single (default: C).")
    parser.add_argument("--scale", type=float, default=None,
                         help="Multiply all coordinates by this factor, e.g. to convert pixels to Angstroms. "
                              "Default: auto-computed so neighboring atoms end up ~--target-spacing apart.")
    parser.add_argument("--out", default="points", help="Output file path/basename, without extension (default: points).")
    parser.add_argument("--plot", action="store_true", help="Also render a 3D scatter plot PNG.")
    args = parser.parse_args()

    is_svg = args.input.lower().endswith(".svg")
    log = []  # collected log lines, printed as a summary at the end of the run

    def log_and_print(msg):
        print(msg)
        log.append(msg)

    if args.mode == "fill":
        import cv2

        tmp_ctx = tempfile.TemporaryDirectory() if is_svg else None
        try:
            if is_svg:
                raster_path = os.path.join(tmp_ctx.name, "rendered.png")
                _rasterize_svg(args.input, raster_path, resolution_scale=args.svg_render_scale)
            else:
                raster_path = args.input

            color_img = cv2.imread(raster_path, cv2.IMREAD_COLOR)
            if color_img is None:
                raise FileNotFoundError(f"Could not read image: {raster_path}")
            mask = _foreground_mask(color_img, color_threshold=args.color_threshold, invert=args.invert)

            if args.spacing is not None:
                spacing = max(1, args.spacing)
            else:
                spacing = _auto_spacing_for_target(mask, target_atoms=args.target_atoms, hard_cap=args.max_atoms)

            points, colors = _fill_points_from_mask(mask, color_img, spacing, z=args.z)
        finally:
            if tmp_ctx is not None:
                tmp_ctx.cleanup()

        scale = args.scale if args.scale is not None else (args.target_spacing / spacing)

        if args.spacing is None or args.scale is None:
            log_and_print(f"Auto-sized: spacing={spacing}px, scale={scale:.4f} "
                           f"(target ~{args.target_atoms} atoms, hard cap {args.max_atoms})")

    else:  # outline
        max_points = args.max_points if args.max_points is not None else args.target_atoms
        points, colors = extract_points(
            args.input, mode="outline", raster_mode=args.raster_mode,
            samples_per_path=args.samples, color_threshold=args.color_threshold,
            invert=args.invert, max_points=max_points, z=args.z,
        )
        scale = args.scale if args.scale is not None else args.target_spacing
        if args.scale is None:
            log_and_print(f"Auto-sized: scale={scale:.4f}")

    log_and_print(f"Extracted {len(points)} points from {args.input} (mode={args.mode}).")

    if args.color_mode == "match":
        elements = nearest_elements_for_colors(colors) if points else []
        unique = sorted(set(elements))
        log_and_print(f"Matched colors to {len(unique)} distinct element(s): {', '.join(unique)}")
    else:
        elements = args.element

    base = args.out
    if args.format in ("csv", "both"):
        csv_path = base + ".csv"
        save_points_csv(points, csv_path)
        log_and_print(f"Saved CSV to {csv_path}")
    if args.format in ("xyz", "both"):
        xyz_path = base + ".xyz"
        save_points_xyz(points, xyz_path, elements=elements, scale=scale)
        log_and_print(f"Saved XYZ to {xyz_path}")

    if args.plot:
        plot_points(points, elements, title=os.path.splitext(os.path.basename(args.input))[0])

    # Post-run summary log
    print()
    print("=" * 50)
    print("RUN SUMMARY")
    print("=" * 50)
    print(f"Input:       {args.input}")
    print(f"Mode:        {args.mode}")
    print(f"Atom count:  {len(points)}")
    if args.color_mode == "match":
        counts = {}
        for e in elements:
            counts[e] = counts.get(e, 0) + 1
        breakdown = ", ".join(f"{e}={n}" for e, n in sorted(counts.items(), key=lambda kv: -kv[1]))
        print(f"Elements:    {breakdown}")
    else:
        print(f"Element:     {args.element} (single, forced)")
    print(f"Scale:       {scale:.4f}")
    if args.mode == "fill":
        print(f"Spacing:     {spacing}px")
    print(f"Output:      {base}.{args.format if args.format != 'both' else '{csv,xyz}'}")
    print("=" * 50)


if __name__ == "__main__":
    main()
