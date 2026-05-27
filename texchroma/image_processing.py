from rembg import remove
from PIL import Image
import numpy as np
import matplotlib.colors as mcolors
import yaml
import os

from sklearn.cluster import MiniBatchKMeans
from skimage.color import rgb2lab

def removebg(image_file):
    # Iterable, removes backfround from target file and resaves as png.
    filename = os.path.splitext(image_file)[0]
    input_image = Image.open(image_file).convert("RGBA")
    output_image = remove(input_image)
    output_name = f'{filename}_remove.png'
    output_image.save(output_name)
    return output_name

def image_prep(ind_image, mask_threshold):
    # Load RGBA image with PIL
    # Converts RGBA PNG into color and transparency array
    image_file = Image.open(ind_image).convert("RGBA")
    img_arr = np.array(image_file).astype(np.uint8)   # shape: (H, W, 4)



    rgb = img_arr[..., :3]
    alpha = img_arr[..., 3]
    mask = alpha > mask_threshold

    h, w = alpha.shape # Targets only the content after removal of background

    # Prepare RGB for LAB
    # rgb2lab expects float in [0,1]
    rgb_float = rgb.astype(np.float32) / 255.0

    # Only convert full image once
    lab = rgb2lab(rgb_float)

    # Extract a,b channels for foreground pixels only
    ab_pixels = lab[..., 1:3][mask]   # shape: (N, 2)

    # Safety check
    if ab_pixels.shape[0] == 0:
        raise ValueError("No foreground pixels found. Check the alpha mask.")
    
    return ab_pixels, mask

def sample_pixels(ab_pixels):
    # Sampling for speed
    sample_size = 50000
    n_samples = min(sample_size, ab_pixels.shape[0])

    idx = np.random.choice(ab_pixels.shape[0], n_samples, replace=False)
    sample = ab_pixels[idx]
    return sample

def kmeans_clustering(sample, ab_pixels, n_clusters):
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=0,
        batch_size=4096,
        n_init=(n_clusters * 2)
    )
    kmeans.fit(sample)
    labels_fg = kmeans.predict(ab_pixels)  # predict on full set, not sample
    return labels_fg

def convert_to_palette(hex_colors):
    palette = (np.array([mcolors.to_rgb(c) for c in hex_colors]) * 255).astype(np.uint8)  # (6,3)
    return palette

def output_builder(palette, labels_fg, mask, filename, output_dir):
    h, w = mask.shape
    segmented_rgba = np.zeros((h, w, 4), dtype=np.uint8)

    # Fill RGB only on garment pixels
    segmented_rgba[..., 3] = 0  # transparent background by default
    segmented_rgba[mask, :3] = palette[labels_fg]
    segmented_rgba[mask, 3] = 255

    # Save output
    segmented_pil = Image.fromarray(segmented_rgba, mode="RGBA")
    save_filename = os.path.splitext(os.path.basename(filename))[0]  # replaces split("\\")
    segmented_pil.save(os.path.join(output_dir, f"segmented_multicolor_mask_{save_filename}.png"))

def process_images(image_dir):
    input_list = []
    for filename in os.listdir(image_dir):
        if filename.endswith("_remove.png"):
            continue
        else:
            filepath = os.path.join(image_dir, filename)
            output_name = removebg(filepath)
            input_list.append(output_name)
    return input_list

def process_single_image(image_file, config, output_dir):
    ab_pixels, mask = image_prep(image_file, config['mask_threshold'])
    sample = sample_pixels(ab_pixels)
    labels_fg = kmeans_clustering(sample, ab_pixels, config['n_clusters'])  # pass full ab_pixels
    palette = convert_to_palette(config['hex_colors'])
    output_builder(palette, labels_fg, mask, image_file, output_dir)

def process_all_images(image_dir, output_dir):
    with open('config/config.yaml') as f:
        config = yaml.safe_load(f)

    image_list = process_images(image_dir)
    for img in image_list:
        process_single_image(img, config, output_dir)

def purge_images(image_dir, output_dir, purge_all=False):
    for directory in (image_dir, output_dir):
        for filename in os.listdir(directory):
            if purge_all or filename.endswith("_remove.png"):
                filepath = os.path.join(directory, filename)
                try:
                    os.remove(filepath)
                    print(f"File '{filename}' deleted successfully.")
                except Exception as e:
                    print(f"Failed to delete '{filename}': {e}")