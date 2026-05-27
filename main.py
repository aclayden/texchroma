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
    filename = image_file.split(".")[0]
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
    
    return ab_pixels

def sample_pixels(ab_pixels):
    # Sampling for speed
    sample_size = 50000
    n_samples = min(sample_size, ab_pixels.shape[0])

    idx = np.random.choice(ab_pixels.shape[0], n_samples, replace=False)
    sample = ab_pixels[idx]
    return sample

def kmeans_clustering(sample, n_samples):
    kmeans = MiniBatchKMeans(
        n_clusters=n_samples,
        random_state=0,
        batch_size=4096,
        n_init=(n_samples * 2)
    )

    kmeans.fit(sample)

    # Predict cluster labels for all foreground pixels
    labels_fg = kmeans.predict(ab_pixels)   # shape: (N,)
    return labels_fg

def convert_to_palette(labels_fg, hex_colours):
    palette = (np.array([mcolors.to_rgb(c) for c in hex_colors]) * 255).astype(np.uint8)  # (6,3)
    return palette

def output_builder(palette, mask, filename):
    segmented_rgba = np.zeros((h, w, 4), dtype=np.uint8)

    # Fill RGB only on garment pixels
    segmented_rgba[..., 3] = 0  # transparent background by default
    segmented_rgba[mask, :3] = palette[labels_fg]
    segmented_rgba[mask, 3] = 255

    # Save output
    segmented_pil = Image.fromarray(segmented_rgba, mode="RGBA")
    segmented_pil.save(f"segmented_multicolor_mask_{filename}.png")

def process_images():
    image_dir = 'images'  # or os.path.join(script_dir, '..', 'config')

    input_list = []
    for filename in os.listdir(image_dir):
        if filename.endswith("_remove.png"):
            continue
        else:
            filepath = os.path.join(image_dir, filename)
            output_name = removebg(filepath)
            input_list.append(output_name)

    # input_list = [x.split("\\")[-1] for x in input_list]
    return input_list

with open('config/config.yaml') as f:
    config = yaml.safe_load(f)

image_list = process_images()

ab_pixels = image_prep(image_list[0],config['mask_threshold'])
sample = sample_pixels(ab_pixels)
labels_fg = kmeans_clustering(sample, config['n_clusters'])
palette = convert_to_palette(labels_fg, config['hex_colours'])
output_builder(palette, config['mask'], f'{image_list[0].split(".")[0]}.png')

