import colour
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import yaml

LINEAR_FORMATS = {'cr2', 'nef', 'arw', 'raf', 'rw2', 'dng', 'exr', 'hdr'}

def is_linear(path):
    return path.lower().rsplit('.', 1)[-1] in LINEAR_FORMATS

def capture_swatches(image_file):
    image = colour.io.read_image(image_file)  # always linear after this

    if not is_linear(image_file):
        image = colour.cctf_decoding(image)  # undo sRGB gamma -> linear

    fig, ax = plt.subplots()
    ax.imshow(colour.cctf_encoding(image))  # encode only for display
    ax.set_title("Click centre of each patch: left to right, top to bottom (24 clicks)")
    clicks = plt.ginput(24, timeout=0)
    plt.close()

    patch_px = 10
    measured = []
    for x, y in clicks:
        x, y = int(x), int(y)
        patch = image[y-patch_px:y+patch_px, x-patch_px:x+patch_px]
        measured.append(patch.mean(axis=(0, 1)))

    return np.array(measured)  # linear

def capture_references():
    reference = colour.CCS_COLOURCHECKERS["ColorChecker24 - After November 2014"]
    reference_xyY = np.array(list(reference.data.values()))
    reference_XYZ = colour.xyY_to_XYZ(reference_xyY)

    # Keep linear — match the domain of measured swatches
    reference_swatches = colour.XYZ_to_RGB(
        reference_XYZ,
        colour.RGB_COLOURSPACES["sRGB"],
        illuminant=reference.illuminant,
        chromatic_adaptation_transform="Bradford",
        apply_cctf_encoding=False  # linear to match measured
    )
    return reference_swatches  # linear, may have out-of-gamut values

def correct_colour(measured, reference_swatches):
    ccm = colour.matrix_colour_correction(
        measured,
        reference_swatches,
        method="Finlayson 2015"
    )
    print(ccm)
    corrected_swatches = colour.apply_matrix_colour_correction(measured, ccm)
    return corrected_swatches, ccm  # return ccm for use in correct_image

def calc_deltaE(corrected_swatches, reference_swatches):
    srgb = colour.RGB_COLOURSPACES["sRGB"]

    def to_lab(rgb):
        XYZ = colour.RGB_to_XYZ(
            rgb,
            srgb,
            illuminant=srgb.whitepoint  # explicit
        )
        return colour.XYZ_to_Lab(XYZ, illuminant=srgb.whitepoint)

    deltaE = colour.delta_E(to_lab(corrected_swatches), to_lab(reference_swatches))
    print(f"Mean ΔE: {deltaE.mean():.3f}, Max ΔE: {deltaE.max():.3f}")
    for i, de in enumerate(deltaE):
        print(f"Patch {i+1:2d}: ΔE={de:.2f}")

def corrected_path(image_file):
    base, ext = os.path.splitext(image_file)
    return f"{base}_corrected{ext}"

def correct_image(image_file, ccm):
    image = colour.io.read_image(image_file)

    if not is_linear(image_file):
        image = colour.cctf_decoding(image)  # -> linear

    corrected = colour.apply_matrix_colour_correction(image, ccm)
    corrected = colour.cctf_encoding(corrected)  # -> sRGB gamma for export
    corrected = np.clip(corrected, 0, 1)
    corrected = (corrected * 255).astype(np.uint8)

    Image.fromarray(corrected).save(corrected_path(image_file))

def save_swatches(measured, image_file):
    data = {
        'source_image': image_file,
        'swatches': measured.tolist()  # numpy -> plain lists for yaml
    }
    out = ".\\config\\swatches.yaml")
    with open(out, 'w') as f:
        yaml.dump(data, f)
    return out

def load_swatches(yaml_file):
    with open(yaml_file) as f:
        data = yaml.safe_load(f)
    return np.array(data['swatches']), data['source_image']

# Main
image_file = ".\\images\\cc2.jpeg"

measured = capture_swatches(image_file)
reference_swatches = capture_references()
corrected_swatches, ccm = correct_colour(measured, reference_swatches)
calc_deltaE(corrected_swatches, reference_swatches)
correct_image(image_file, ccm)