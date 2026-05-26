from rembg import remove
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from sklearn.cluster import MiniBatchKMeans
from skimage.color import rgb2lab

# -----------------------------
# 1) Remove background
# -----------------------------
input_image = Image.open("IMG_8990.JPG").convert("RGBA")
output_image = remove(input_image)
output_image.save("output.png")

# -----------------------------
# 2) Load RGBA image with PIL
# -----------------------------
img_arr = np.array(output_image).astype(np.uint8)   # shape: (H, W, 4)

rgb = img_arr[..., :3]
alpha = img_arr[..., 3]

h, w = alpha.shape

# -----------------------------
# 3) Foreground mask
# -----------------------------
mask = alpha > 0   # use >0 for uint8 alpha
# If your mask is noisy, try alpha > 10 or alpha > 20

# -----------------------------
# 4) Prepare RGB for LAB
# rgb2lab expects float in [0,1]
# -----------------------------
rgb_float = rgb.astype(np.float32) / 255.0

# Only convert full image once
lab = rgb2lab(rgb_float)

# Extract a,b channels for foreground pixels only
ab_pixels = lab[..., 1:3][mask]   # shape: (N, 2)

# Safety check
if ab_pixels.shape[0] == 0:
    raise ValueError("No foreground pixels found. Check the alpha mask.")

# -----------------------------
# 5) Sampling for speed
# -----------------------------
sample_size = 50000
n_samples = min(sample_size, ab_pixels.shape[0])

idx = np.random.choice(ab_pixels.shape[0], n_samples, replace=False)
sample = ab_pixels[idx]

# -----------------------------
# 6) KMeans clustering
# -----------------------------
n_clusters = 6

kmeans = MiniBatchKMeans(
    n_clusters=n_clusters,
    random_state=0,
    batch_size=4096,
    n_init=10
)

kmeans.fit(sample)

# Predict cluster labels for all foreground pixels
labels_fg = kmeans.predict(ab_pixels)   # shape: (N,)

# -----------------------------
# 7) Fixed palette
# -----------------------------
hex_colors = [
    "#FF0000",  # red
    "#0000FF",  # blue
    "#00FF00",  # green
    "#FFFF00",  # yellow
    "#000000",  # black
    "#FFFFFF",  # white
]

palette = (np.array([mcolors.to_rgb(c) for c in hex_colors]) * 255).astype(np.uint8)  # (6,3)

# -----------------------------
# 8) Build segmented RGBA image
# -----------------------------
segmented_rgba = np.zeros((h, w, 4), dtype=np.uint8)

# Fill RGB only on garment pixels
segmented_rgba[..., 3] = 0  # transparent background by default
segmented_rgba[mask, :3] = palette[labels_fg]
segmented_rgba[mask, 3] = 255

# Save output
segmented_pil = Image.fromarray(segmented_rgba, mode="RGBA")
segmented_pil.save("segmented_multicolor_mask.png")

# -----------------------------
# 9) Preview
# -----------------------------
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(img_arr)
plt.title("Background Removed")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(segmented_rgba)
plt.title("Combined Multicolour Mask")
plt.axis("off")

plt.tight_layout()
plt.show()