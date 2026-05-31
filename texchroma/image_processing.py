import os

import numpy as np
import yaml
import matplotlib.colors as mcolors
from PIL import Image
from sklearn.cluster import KMeans
import skimage.color as sc

# ---------------------------------------------------------------------------
# Background removal
# ---------------------------------------------------------------------------

def removebg(image_file):
    """Remove background from image_file; save and return the _remove.png path."""
    try:
        from rembg import remove
    except ImportError:
        raise ImportError("rembg is required for background removal: pip install rembg")
    if not os.path.isfile(image_file):
        raise FileNotFoundError(f"Image file not found: {image_file}")
    filename = os.path.splitext(image_file)[0]
    input_image = Image.open(image_file).convert("RGBA")
    output_image = remove(input_image)
    output_name = f"{filename}_remove.png"
    output_image.save(output_name)
    return output_name


# ---------------------------------------------------------------------------
# Colour correction (optional pass-through)
# ---------------------------------------------------------------------------

def apply_ccm_to_image(image_file, ccm):
    """Apply colour correction matrix to image_file, return corrected PIL Image.

    This is an in-memory operation — it does not write to disk. The CCM is
    applied in linear light and the result is returned as an sRGB-encoded
    RGBA PIL Image ready for downstream processing.

    If ccm is None this function is a no-op and returns the original image
    unchanged, so callers do not need to branch.
    """
    from PIL import Image as _Image
    img = _Image.open(image_file).convert("RGBA")
    if ccm is None:
        return img

    try:
        import colour
    except ImportError:
        raise ImportError(
            "colour-science is required for CCM correction. "
            "Install it with: pip install colour-science"
        )

    rgb = np.array(img)[..., :3].astype(np.float32) / 255.0
    alpha = np.array(img)[..., 3]

    # Linear light for CCM application
    rgb_linear = colour.cctf_decoding(rgb)
    corrected_linear = colour.apply_matrix_colour_correction(rgb_linear, ccm)
    corrected_srgb = colour.cctf_encoding(corrected_linear)
    corrected_srgb = np.clip(corrected_srgb, 0, 1)
    corrected_uint8 = (corrected_srgb * 255).astype(np.uint8)

    rgba = np.dstack([corrected_uint8, alpha])
    return _Image.fromarray(rgba, mode="RGBA")


# ---------------------------------------------------------------------------
# Image preparation and clustering
# ---------------------------------------------------------------------------

def image_prep(ind_image_or_pil, mask_threshold):
    """Convert an RGBA image to (ab_pixels, mask) in CIELAB space.

    Accepts either a file path (str) or a PIL Image object so that a
    CCM-corrected in-memory image can be passed directly without re-saving.
    """
    if isinstance(ind_image_or_pil, str):
        if not ind_image_or_pil.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
            raise ValueError(f"Unsupported file type: {ind_image_or_pil}")
        image_file = Image.open(ind_image_or_pil).convert("RGBA")
    else:
        image_file = ind_image_or_pil.convert("RGBA")   # already a PIL Image

    img_arr = np.array(image_file).astype(np.uint8)   # (H, W, 4)
    rgb     = img_arr[..., :3]
    alpha   = img_arr[..., 3]
    mask    = alpha > mask_threshold

    rgb_float = rgb.astype(np.float32) / 255.0
    lab       = rgb2lab(rgb_float)
    ab_pixels = lab[..., 1:3][mask]   # (N, 2)

    if ab_pixels.shape[0] == 0:
        raise ValueError("No foreground pixels found. Check the alpha mask.")

    return ab_pixels, mask


def sample_pixels(ab_pixels, rand_int):
    sample_size = 50000
    n_samples = min(sample_size, ab_pixels.shape[0])
    np.random.seed(rand_int)
    idx = np.random.choice(ab_pixels.shape[0], n_samples, replace=False)
    return ab_pixels[idx]


def kmeans_clustering(sample, ab_pixels, n_clusters, rand_int):
    if n_clusters < 1:
        raise ValueError(f"n_clusters must be at least 1, got {n_clusters}")
    if ab_pixels.shape[0] < n_clusters:
        raise ValueError(
            f"Fewer foreground pixels ({ab_pixels.shape[0]}) than "
            f"n_clusters ({n_clusters})"
        )
    kmeans = KMeans(n_clusters=n_clusters, random_state=rand_int)
    kmeans.fit(sample)
    labels_fg = kmeans.predict(ab_pixels)
    return kmeans, labels_fg


def convert_to_palette(hex_colors):
    try:
        palette = (
            np.array([mcolors.to_rgb(c) for c in hex_colors]) * 255
        ).astype(np.uint8)
    except ValueError as e:
        raise ValueError(f"Invalid hex colour in palette: {e}")
    return palette


def output_builder(palette, labels_fg, mask, filename, output_dir):
    if not os.path.isdir(output_dir):
        raise FileNotFoundError(f"Output directory does not exist: {output_dir}")
    if len(palette) != (labels_fg.max() + 1):
        raise ValueError(
            f"Palette length {len(palette)} does not match cluster count "
            f"{labels_fg.max() + 1}"
        )

    h, w = mask.shape
    segmented_rgba = np.zeros((h, w, 4), dtype=np.uint8)
    segmented_rgba[..., 3] = 0
    segmented_rgba[mask, :3] = palette[labels_fg]
    segmented_rgba[mask, 3]  = 255

    segmented_pil = Image.fromarray(segmented_rgba, mode="RGBA")
    save_filename  = os.path.splitext(os.path.basename(filename))[0]
    segmented_pil.save(
        os.path.join(output_dir, f"segmented_multicolor_mask_{save_filename}.png")
    )


# ---------------------------------------------------------------------------
# Pipeline entry points
# ---------------------------------------------------------------------------

def process_images(image_dir):
    """Run removebg on every non-removed image in image_dir; return _remove.png paths."""
    input_list = []
    for filename in os.listdir(image_dir):
        if filename.endswith("_remove.png"):
            continue
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff')):
            continue
        filepath   = os.path.join(image_dir, filename)
        output_name = removebg(filepath)
        input_list.append(output_name)
    if not input_list:
        raise ValueError(f"No supported image files found in {image_dir}")
    return input_list


def process_single_image(image_file, config, output_dir, ccm=None):
    """Process one image through the full pipeline.

    Parameters
    ----------
    image_file : str
        Path to the _remove.png (background-removed) image.
    config : dict
        Loaded config YAML. Must contain rand_int, mask_threshold,
        n_clusters, hex_colors.
    output_dir : str
        Directory for segmented output images.
    ccm : np.ndarray or None
        3×3 colour correction matrix loaded from a swatch YAML.
        Pass None to skip colour correction entirely.
    """
    rand_int = config["random_state"]

    # CCM is applied in-memory; returns unchanged PIL Image when ccm is None
    corrected_image = apply_ccm_to_image(image_file, ccm)

    ab_pixels, mask = image_prep(corrected_image, config["mask_threshold"])
    sample          = sample_pixels(ab_pixels, rand_int)
    kmeans, labels_fg = kmeans_clustering(
        sample, ab_pixels, config["n_clusters"], rand_int
    )
    palette = convert_to_palette(config["hex_colors"][: config["n_clusters"]])
    output_builder(palette, labels_fg, mask, image_file, output_dir)


def process_all_images(image_dir, output_dir, config, ccm=None):
    """Process every image in image_dir.

    config must already be loaded (dict). ccm is the optional correction matrix.
    """
    image_list = process_images(image_dir)
    for img_path in image_list:
        process_single_image(img_path, config, output_dir, ccm=ccm)


def purge_images(image_dir, output_dir, purge_all=False):
    for directory in (image_dir, output_dir):
        for filename in os.listdir(directory):
            if purge_all or filename.endswith("_remove.png"):
                filepath = os.path.join(directory, filename)
                try:
                    os.remove(filepath)
                    print(f"Deleted: {filename}")
                except Exception as e:
                    print(f"Failed to delete '{filename}': {e}")

def centroid_to_patch(lab_centroid, size=200):
    # CIELAB → sRGB
    lab = np.array([[lab_centroid]], dtype=np.float64)  # shape (1,1,3)
    rgb = sc.lab2rgb(lab)  # 0–1 float
    rgb_uint8 = (rgb[0, 0] * 255).astype(np.uint8)
    img = Image.fromarray(
        np.full((size, size, 3), rgb_uint8, dtype=np.uint8), 'RGB'
    )
    return img