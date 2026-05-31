import os
import uuid
import datetime

import colour
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
import rawpy
import yaml

LINEAR_FORMATS = {'cr2', 'nef', 'arw', 'raf', 'rw2', 'dng', 'exr', 'hdr'}
RAW_FORMATS    = {'cr2', 'nef', 'arw', 'raf', 'rw2', 'dng'}

_ext = lambda path: path.lower().rsplit('.', 1)[-1]

def is_linear(path): return _ext(path) in LINEAR_FORMATS
def is_raw(path):    return _ext(path) in RAW_FORMATS


def read_image_linear(path):
    """Read any supported image as a linear float32 RGB array (H, W, 3)."""
    if is_raw(path):
        with rawpy.imread(path) as raw:
            rgb = raw.postprocess(
                use_camera_wb=False, use_auto_wb=False,
                no_auto_bright=True, output_bps=16, gamma=(1, 1),
            )
        return rgb.astype(np.float32) / 65535.0
    return colour.cctf_decoding(colour.io.read_image(path))


def capture_swatches(image_file):
    """Interactively capture 24 ColorChecker patch centres.

    Left-click to add, right-click to undo. Closes at 24 clicks.
    Returns linear RGB array (24, 3).
    """
    image = read_image_linear(image_file)
    fig, ax = plt.subplots()
    ax.imshow(colour.cctf_encoding(image))

    clicks, markers = [], []

    def redraw_title():
        n = len(clicks)
        ax.set_title(
            f"Click patch centre ({n}/24) — right-click to undo" if n < 24
            else "24 patches captured — closing..."
        )
        fig.canvas.draw()

    redraw_title()

    def on_click(event):
        if event.inaxes is not ax or event.xdata is None:
            return
        if event.button == 1 and len(clicks) < 24:
            x, y = int(event.xdata), int(event.ydata)
            clicks.append((x, y))
            p = 10
            rect  = mpatches.Rectangle((x - p, y - p), p * 2, p * 2,
                                       linewidth=1, edgecolor="red", facecolor="none")
            ax.add_patch(rect)
            label = ax.annotate(str(len(clicks)), (x - p, y - p), color="white",
                                fontsize=7, fontweight="bold",
                                xytext=(2, 2), textcoords="offset points")
            markers.append((rect, label))
            redraw_title()
            if len(clicks) == 24:
                plt.close(fig)
        elif event.button == 3 and clicks:
            clicks.pop()
            for artist in markers.pop():
                artist.remove()
            redraw_title()

    fig.canvas.mpl_connect("button_press_event", on_click)
    plt.show(block=True)

    if len(clicks) != 24:
        raise RuntimeError(f"Capture incomplete: {len(clicks)}/24 clicks recorded.")

    p = 10
    measured = [image[y-p:y+p, x-p:x+p].mean(axis=(0, 1)) for x, y in clicks]
    return np.array(measured)


def capture_references():
    """Return reference linear sRGB values for the ColorChecker24 (post-2014)."""
    ref = colour.CCS_COLOURCHECKERS["ColorChecker24 - After November 2014"]
    return colour.XYZ_to_RGB(
        colour.xyY_to_XYZ(np.array(list(ref.data.values()))),
        colour.RGB_COLOURSPACES["sRGB"],
        illuminant=ref.illuminant,
        chromatic_adaptation_transform="Bradford",
        apply_cctf_encoding=False,
    )


def correct_colour(measured, reference):
    """Compute CCM and return (corrected_swatches, ccm)."""
    ccm = colour.matrix_colour_correction(measured, reference, method="Finlayson 2015")
    return colour.apply_matrix_colour_correction(measured, ccm), ccm


def calc_deltaE(corrected, reference):
    """Print per-patch ΔE2000 and summary. Returns deltaE array."""
    srgb = colour.RGB_COLOURSPACES["sRGB"]
    def to_lab(rgb):
        return colour.XYZ_to_Lab(
            colour.RGB_to_XYZ(rgb, srgb, illuminant=srgb.whitepoint),
            illuminant=srgb.whitepoint,
        )
    dE = colour.delta_E(to_lab(corrected), to_lab(reference))
    print(f"Mean ΔE: {dE.mean():.3f}  Max ΔE: {dE.max():.3f}")
    for i, de in enumerate(dE):
        print(f"  Patch {i+1:2d}: {de:.2f}")
    return dE


def corrected_path(image_file):
    base, ext = os.path.splitext(image_file)
    return f"{base}_corrected{'.tif' if is_raw(image_file) else ext}"


def correct_image(image_file, ccm):
    """Apply CCM, save corrected file, return output path."""
    corrected = np.clip(colour.cctf_encoding(
        colour.apply_matrix_colour_correction(read_image_linear(image_file), ccm)
    ), 0, 1)
    out = corrected_path(image_file)
    if is_raw(image_file):
        Image.fromarray((corrected * 65535).astype(np.uint16), mode="I;16").save(out)
    else:
        Image.fromarray((corrected * 255).astype(np.uint8)).save(out)
    return out


def save_swatches(measured, ccm, image_file, config_dir="config"):
    """Save swatches and CCM to a timestamped YAML. Returns file path."""
    os.makedirs(config_dir, exist_ok=True)
    session_id = str(uuid.uuid4())
    timestamp  = datetime.datetime.now(datetime.timezone.utc).isoformat()
    out = os.path.join(config_dir, f"swatches_{timestamp[:10]}_{session_id[:8]}.yaml")
    with open(out, "w") as f:
        yaml.dump({
            "session_id":   session_id,
            "timestamp":    timestamp,
            "source_image": os.path.abspath(image_file),
            "swatches":     measured.tolist(),
            "ccm":          ccm.tolist(),
        }, f, default_flow_style=False, sort_keys=False)
    print(f"Swatches saved → {out}")
    return out


def load_swatches(yaml_file):
    """Load swatch YAML. Returns (measured, ccm, meta)."""
    with open(yaml_file) as f:
        data = yaml.safe_load(f)
    return (
        np.array(data["swatches"]),
        np.array(data["ccm"]),
        {k: data.get(k) for k in ("session_id", "timestamp", "source_image")},
    )


def run_correction_session(image_file, config_dir="config"):
    """Capture → CCM → save. Returns (ccm, swatch_yaml_path)."""
    measured   = capture_swatches(image_file)
    reference  = capture_references()
    corrected, ccm = correct_colour(measured, reference)
    calc_deltaE(corrected, reference)
    return ccm, save_swatches(measured, ccm, image_file, config_dir)