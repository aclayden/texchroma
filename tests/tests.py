"""
tests.py — texchroma unit tests

Run with:  python -m pytest tests.py -v
       or: python tests.py          (stdlib unittest runner)

All tests use synthetic numpy/PIL data. No real images, no display,
no rembg neural net, no colour-science or rawpy at import time.
Heavy external dependencies (colour, rawpy, rembg) are mocked where
they appear so the suite runs without those packages installed.
"""

import io
import os
import sys
import math
import types
import tempfile
import unittest
import unittest.mock as mock

import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Helpers for building synthetic test data
# ---------------------------------------------------------------------------

def make_rgba_image(h=40, w=40, rgb=(180, 100, 60), alpha=255):
    """Return a solid-colour PIL RGBA Image."""
    arr = np.zeros((h, w, 4), dtype=np.uint8)
    arr[..., :3] = rgb
    arr[..., 3]  = alpha
    return Image.fromarray(arr, mode="RGBA")


def make_rgba_with_transparent_border(h=40, w=40, border=5):
    """Return a PIL RGBA Image with an opaque centre and transparent border."""
    arr = np.zeros((h, w, 4), dtype=np.uint8)
    arr[..., :3] = (120, 80, 200)
    arr[..., 3]  = 255
    arr[:border, :, 3]  = 0
    arr[-border:, :, 3] = 0
    arr[:, :border, 3]  = 0
    arr[:, -border:, 3] = 0
    return Image.fromarray(arr, mode="RGBA")


def make_swatches(n=24):
    """Return a (n, 3) float32 array of synthetic linear RGB swatch values."""
    rng = np.random.default_rng(42)
    return rng.random((n, 3)).astype(np.float32)


def make_ccm():
    """Return a plausible 3×3 CCM (identity with small perturbation)."""
    return np.eye(3, dtype=np.float64) + np.random.default_rng(7).uniform(
        -0.05, 0.05, (3, 3)
    )


# ---------------------------------------------------------------------------
# Stub the heavy imports so modules load without them installed
# ---------------------------------------------------------------------------

def _stub_colour():
    """Return a minimal mock of the colour-science package."""
    m = types.ModuleType("colour")

    def cctf_decoding(x):
        # approximate sRGB -> linear (gamma 2.2 stand-in)
        return np.power(np.clip(x, 0, 1), 2.2).astype(np.float32)

    def cctf_encoding(x):
        return np.power(np.clip(x, 0, 1), 1 / 2.2).astype(np.float32)

    def apply_matrix_colour_correction(rgb, ccm):
        return (rgb @ ccm.T).astype(np.float32)

    m.cctf_decoding  = cctf_decoding
    m.cctf_encoding  = cctf_encoding
    m.apply_matrix_colour_correction = apply_matrix_colour_correction

    # colour.io sub-module
    io_mod = types.ModuleType("colour.io")
    def read_image(path):
        arr = np.array(Image.open(path).convert("RGB")).astype(np.float32) / 255.0
        return arr
    io_mod.read_image = read_image
    m.io = io_mod
    sys.modules["colour.io"] = io_mod

    return m


def _install_stubs():
    if "colour" not in sys.modules:
        sys.modules["colour"] = _stub_colour()
    if "rawpy" not in sys.modules:
        rp = types.ModuleType("rawpy")
        sys.modules["rawpy"] = rp
    if "rembg" not in sys.modules:
        rb = types.ModuleType("rembg")
        rb.remove = lambda img: img   # no-op: return image unchanged
        sys.modules["rembg"] = rb

_install_stubs()


# ---------------------------------------------------------------------------
# Import the modules under test (after stubs are in place)
# ---------------------------------------------------------------------------

sys.path.insert(0, os.path.dirname(__file__))
import colour_correction as cc
import image_processing  as ip


# ===========================================================================
# colour_correction tests
# ===========================================================================

class TestFormatClassifiers(unittest.TestCase):

    def test_is_linear_raw_formats(self):
        for ext in ("cr2", "CR2", "nef", "ARW", "dng", "rw2", "raf"):
            with self.subTest(ext=ext):
                self.assertTrue(cc.is_linear(f"shot.{ext}"))

    def test_is_linear_non_raw(self):
        for ext in ("jpg", "jpeg", "png", "tif", "tiff"):
            with self.subTest(ext=ext):
                self.assertFalse(cc.is_linear(f"photo.{ext}"))

    def test_is_raw_cr2(self):
        self.assertTrue(cc.is_raw("DSC_0001.CR2"))
        self.assertTrue(cc.is_raw("DSC_0001.cr2"))

    def test_is_raw_excludes_exr(self):
        # exr is in LINEAR_FORMATS but not RAW_FORMATS
        self.assertFalse(cc.is_raw("render.exr"))

    def test_is_raw_excludes_jpg(self):
        self.assertFalse(cc.is_raw("photo.jpg"))


class TestCorrectedPath(unittest.TestCase):

    def test_raw_input_becomes_tif(self):
        result = cc.corrected_path("DSC_0001.CR2")
        self.assertEqual(result, "DSC_0001_corrected.tif")

    def test_raw_lowercase(self):
        result = cc.corrected_path("shot.cr2")
        self.assertEqual(result, "shot_corrected.tif")

    def test_jpeg_keeps_extension(self):
        result = cc.corrected_path("photo.jpg")
        self.assertEqual(result, "photo_corrected.jpg")

    def test_tif_keeps_extension(self):
        result = cc.corrected_path("scan.tif")
        self.assertEqual(result, "scan_corrected.tif")

    def test_preserves_directory(self):
        result = cc.corrected_path("/some/path/image.cr2")
        self.assertEqual(result, "/some/path/image_corrected.tif")


class TestCorrectColour(unittest.TestCase):
    """correct_colour wraps colour.matrix_colour_correction — test the contract."""

    def test_returns_tuple(self):
        measured   = make_swatches()
        references = make_swatches()
        with mock.patch("colour_correction.colour") as mock_colour:
            mock_colour.matrix_colour_correction.return_value = make_ccm()
            mock_colour.apply_matrix_colour_correction.side_effect = (
                lambda m, ccm: m @ ccm.T
            )
            result = cc.correct_colour(measured, references)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    def test_corrected_swatches_shape(self):
        measured   = make_swatches()
        references = make_swatches()
        with mock.patch("colour_correction.colour") as mock_colour:
            mock_colour.matrix_colour_correction.return_value = make_ccm()
            mock_colour.apply_matrix_colour_correction.side_effect = (
                lambda m, ccm: m @ ccm.T
            )
            corrected, _ = cc.correct_colour(measured, references)
        self.assertEqual(corrected.shape, (24, 3))

    def test_identity_ccm_leaves_swatches_unchanged(self):
        """If CCM is identity, corrected == measured (within float tolerance)."""
        measured = make_swatches()
        identity = np.eye(3)
        with mock.patch("colour_correction.colour") as mock_colour:
            mock_colour.matrix_colour_correction.return_value = identity
            mock_colour.apply_matrix_colour_correction.side_effect = (
                lambda m, ccm: m @ ccm.T
            )
            corrected, _ = cc.correct_colour(measured, measured)
        np.testing.assert_allclose(corrected, measured, atol=1e-6)


class TestSaveLoadSwatches(unittest.TestCase):

    def test_round_trip(self):
        measured = make_swatches()
        ccm      = make_ccm()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = cc.save_swatches(measured, ccm, "checker.cr2", config_dir=tmpdir)
            self.assertTrue(os.path.isfile(path))
            loaded_m, loaded_ccm, meta = cc.load_swatches(path)
        np.testing.assert_allclose(loaded_m,   measured, atol=1e-6)
        np.testing.assert_allclose(loaded_ccm, ccm,      atol=1e-6)

    def test_yaml_has_required_keys(self):
        import yaml
        measured = make_swatches()
        ccm      = make_ccm()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = cc.save_swatches(measured, ccm, "checker.cr2", config_dir=tmpdir)
            with open(path) as f:
                data = yaml.safe_load(f)
        for key in ("session_id", "timestamp", "source_image", "swatches", "ccm"):
            self.assertIn(key, data, msg=f"Missing key: {key}")

    def test_filename_contains_date_and_uuid_prefix(self):
        measured = make_swatches()
        ccm      = make_ccm()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = cc.save_swatches(measured, ccm, "checker.cr2", config_dir=tmpdir)
            fname = os.path.basename(path)
        # expect: swatches_YYYY-MM-DD_<8hex>.yaml
        import re
        self.assertRegex(fname, r"^swatches_\d{4}-\d{2}-\d{2}_[0-9a-f]{8}\.yaml$")

    def test_each_save_gets_unique_session_id(self):
        import yaml
        measured = make_swatches()
        ccm      = make_ccm()
        ids = []
        with tempfile.TemporaryDirectory() as tmpdir:
            for _ in range(3):
                path = cc.save_swatches(measured, ccm, "checker.cr2", config_dir=tmpdir)
                with open(path) as f:
                    ids.append(yaml.safe_load(f)["session_id"])
        self.assertEqual(len(set(ids)), 3, "session_ids should be unique")

    def test_load_meta_fields(self):
        measured = make_swatches()
        ccm      = make_ccm()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = cc.save_swatches(measured, ccm, "checker.cr2", config_dir=tmpdir)
            _, _, meta = cc.load_swatches(path)
        self.assertIn("session_id",   meta)
        self.assertIn("timestamp",    meta)
        self.assertIn("source_image", meta)
        self.assertIsNotNone(meta["session_id"])

    def test_config_dir_created_if_absent(self):
        measured = make_swatches()
        ccm      = make_ccm()
        with tempfile.TemporaryDirectory() as tmpdir:
            new_dir = os.path.join(tmpdir, "does_not_exist_yet")
            self.assertFalse(os.path.isdir(new_dir))
            cc.save_swatches(measured, ccm, "checker.cr2", config_dir=new_dir)
            self.assertTrue(os.path.isdir(new_dir))


class TestCorrectImageNonRaw(unittest.TestCase):
    """correct_image on a JPEG/PNG — uses the colour stub, saves 8-bit."""

    def test_saves_corrected_file(self):
        ccm = make_ccm()
        with tempfile.TemporaryDirectory() as tmpdir:
            # write a small synthetic PNG
            src = os.path.join(tmpdir, "photo.png")
            make_rgba_image(h=20, w=20).convert("RGB").save(src)

            out = cc.correct_image(src, ccm)
            self.assertTrue(os.path.isfile(out))

    def test_output_path_is_corrected_variant(self):
        ccm = make_ccm()
        with tempfile.TemporaryDirectory() as tmpdir:
            src = os.path.join(tmpdir, "photo.png")
            make_rgba_image(h=20, w=20).convert("RGB").save(src)
            out = cc.correct_image(src, ccm)
        self.assertTrue(out.endswith("_corrected.png"))

    def test_output_is_valid_image(self):
        ccm = make_ccm()
        with tempfile.TemporaryDirectory() as tmpdir:
            src = os.path.join(tmpdir, "photo.png")
            make_rgba_image(h=20, w=20).convert("RGB").save(src)
            out = cc.correct_image(src, ccm)
            img = Image.open(out)
            self.assertEqual(img.size, (20, 20))


# ===========================================================================
# image_processing tests
# ===========================================================================

class TestImagePrep(unittest.TestCase):

    def test_returns_ab_pixels_and_mask(self):
        pil_img = make_rgba_image(h=30, w=30)
        ab_pixels, mask = ip.image_prep(pil_img, mask_threshold=128)
        self.assertEqual(ab_pixels.ndim, 2)
        self.assertEqual(ab_pixels.shape[1], 2)
        self.assertEqual(mask.shape, (30, 30))

    def test_mask_excludes_transparent_pixels(self):
        pil_img = make_rgba_with_transparent_border(h=40, w=40, border=5)
        ab_pixels, mask = ip.image_prep(pil_img, mask_threshold=128)
        # border pixels should be masked out
        self.assertFalse(mask[0, 0])
        self.assertTrue(mask[20, 20])

    def test_foreground_pixel_count_matches_mask(self):
        pil_img = make_rgba_image(h=20, w=20, alpha=255)
        ab_pixels, mask = ip.image_prep(pil_img, mask_threshold=128)
        self.assertEqual(ab_pixels.shape[0], mask.sum())

    def test_fully_transparent_image_raises(self):
        pil_img = make_rgba_image(h=20, w=20, alpha=0)
        with self.assertRaises(ValueError):
            ip.image_prep(pil_img, mask_threshold=128)

    def test_accepts_file_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.png")
            make_rgba_image(h=20, w=20).save(path)
            ab_pixels, mask = ip.image_prep(path, mask_threshold=128)
        self.assertGreater(ab_pixels.shape[0], 0)

    def test_rejects_unsupported_extension(self):
        with self.assertRaises(ValueError):
            ip.image_prep("image.bmp", mask_threshold=128)


class TestSamplePixels(unittest.TestCase):

    def test_returns_at_most_50000(self):
        ab = np.random.rand(100_000, 2).astype(np.float32)
        sample = ip.sample_pixels(ab, rand_int=0)
        self.assertLessEqual(sample.shape[0], 50_000)

    def test_returns_all_if_fewer_than_50000(self):
        ab = np.random.rand(1_000, 2).astype(np.float32)
        sample = ip.sample_pixels(ab, rand_int=0)
        self.assertEqual(sample.shape[0], 1_000)

    def test_deterministic_with_same_seed(self):
        ab = np.random.rand(10_000, 2).astype(np.float32)
        s1 = ip.sample_pixels(ab, rand_int=99)
        s2 = ip.sample_pixels(ab, rand_int=99)
        np.testing.assert_array_equal(s1, s2)

    def test_different_seeds_give_different_samples(self):
        ab = np.random.rand(10_000, 2).astype(np.float32)
        s1 = ip.sample_pixels(ab, rand_int=1)
        s2 = ip.sample_pixels(ab, rand_int=2)
        self.assertFalse(np.array_equal(s1, s2))


class TestKMeansClustering(unittest.TestCase):

    def _make_ab(self, n=500):
        rng = np.random.default_rng(0)
        return rng.uniform(-50, 50, (n, 2)).astype(np.float32)

    def test_returns_kmeans_and_labels(self):
        ab = self._make_ab()
        kmeans, labels = ip.kmeans_clustering(ab, ab, n_clusters=3, rand_int=0)
        self.assertEqual(labels.shape[0], ab.shape[0])

    def test_label_count_matches_n_clusters(self):
        ab = self._make_ab()
        _, labels = ip.kmeans_clustering(ab, ab, n_clusters=5, rand_int=0)
        self.assertEqual(len(np.unique(labels)), 5)

    def test_raises_on_zero_clusters(self):
        ab = self._make_ab()
        with self.assertRaises(ValueError):
            ip.kmeans_clustering(ab, ab, n_clusters=0, rand_int=0)

    def test_raises_when_pixels_fewer_than_clusters(self):
        ab = self._make_ab(n=3)
        with self.assertRaises(ValueError):
            ip.kmeans_clustering(ab, ab, n_clusters=5, rand_int=0)

    def test_deterministic_with_fixed_random_state(self):
        ab = self._make_ab(n=200)
        _, l1 = ip.kmeans_clustering(ab, ab, n_clusters=3, rand_int=42)
        _, l2 = ip.kmeans_clustering(ab, ab, n_clusters=3, rand_int=42)
        np.testing.assert_array_equal(l1, l2)


class TestConvertToPalette(unittest.TestCase):

    def test_valid_hex_returns_uint8_array(self):
        palette = ip.convert_to_palette(["#ff0000", "#00ff00", "#0000ff"])
        self.assertEqual(palette.dtype, np.uint8)
        self.assertEqual(palette.shape, (3, 3))

    def test_red_maps_correctly(self):
        palette = ip.convert_to_palette(["#ff0000"])
        np.testing.assert_array_equal(palette[0], [255, 0, 0])

    def test_invalid_hex_raises(self):
        with self.assertRaises(ValueError):
            ip.convert_to_palette(["#gggggg"])

    def test_empty_list(self):
        palette = ip.convert_to_palette([])
        self.assertEqual(palette.shape[0], 0)


class TestOutputBuilder(unittest.TestCase):

    def _setup(self, tmpdir, h=30, w=30, n_clusters=3):
        mask = np.ones((h, w), dtype=bool)
        labels_fg = np.zeros(h * w, dtype=int)
        # assign each cluster label to a stripe of pixels
        stride = (h * w) // n_clusters
        for i in range(n_clusters):
            labels_fg[i * stride:(i + 1) * stride] = i
        palette = ip.convert_to_palette(
            ["#ff0000", "#00ff00", "#0000ff"][:n_clusters]
        )
        return mask, labels_fg, palette

    def test_creates_output_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mask, labels_fg, palette = self._setup(tmpdir)
            ip.output_builder(palette, labels_fg, mask, "garment.png", tmpdir)
            expected = os.path.join(tmpdir, "segmented_multicolor_mask_garment.png")
            self.assertTrue(os.path.isfile(expected))

    def test_output_is_rgba(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mask, labels_fg, palette = self._setup(tmpdir)
            ip.output_builder(palette, labels_fg, mask, "garment.png", tmpdir)
            out = Image.open(
                os.path.join(tmpdir, "segmented_multicolor_mask_garment.png")
            )
            self.assertEqual(out.mode, "RGBA")

    def test_raises_on_missing_output_dir(self):
        mask   = np.ones((10, 10), dtype=bool)
        labels = np.zeros(100, dtype=int)
        palette = ip.convert_to_palette(["#ff0000"])
        with self.assertRaises(FileNotFoundError):
            ip.output_builder(palette, labels, mask, "g.png", "/nonexistent/dir")

    def test_raises_on_palette_label_mismatch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mask = np.ones((10, 10), dtype=bool)
            # 3-cluster labels but only 2-colour palette
            labels  = np.array([0, 1, 2] * 33 + [0], dtype=int)
            palette = ip.convert_to_palette(["#ff0000", "#00ff00"])
            with self.assertRaises(ValueError):
                ip.output_builder(palette, labels, mask, "g.png", tmpdir)


class TestApplyCcmToImage(unittest.TestCase):

    def _write_png(self, tmpdir, name="img.png"):
        path = os.path.join(tmpdir, name)
        make_rgba_image(h=20, w=20).save(path)
        return path

    def test_none_ccm_returns_image_unchanged(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._write_png(tmpdir)
            original = np.array(Image.open(path).convert("RGBA"))
            result   = ip.apply_ccm_to_image(path, ccm=None)
            np.testing.assert_array_equal(np.array(result), original)

    def test_returns_pil_rgba(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._write_png(tmpdir)
            result = ip.apply_ccm_to_image(path, ccm=make_ccm())
            self.assertIsInstance(result, Image.Image)
            self.assertEqual(result.mode, "RGBA")

    def test_output_size_unchanged(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._write_png(tmpdir)
            result = ip.apply_ccm_to_image(path, ccm=make_ccm())
            self.assertEqual(result.size, (20, 20))

    def test_identity_ccm_preserves_values(self):
        """Identity CCM (in linear space) should produce negligible change."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._write_png(tmpdir)
            original = np.array(Image.open(path).convert("RGBA"))[..., :3]
            result   = np.array(ip.apply_ccm_to_image(path, np.eye(3)))[..., :3]
            # sRGB -> linear -> identity -> sRGB round-trip: allow 2 DN tolerance
            np.testing.assert_allclose(result.astype(float), original.astype(float),
                                       atol=2)

    def test_alpha_channel_preserved(self):
        """CCM correction must not alter the alpha channel."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._write_png(tmpdir)
            original_alpha = np.array(Image.open(path).convert("RGBA"))[..., 3]
            result_alpha   = np.array(ip.apply_ccm_to_image(path, make_ccm()))[..., 3]
            np.testing.assert_array_equal(result_alpha, original_alpha)


class TestRemoveBg(unittest.TestCase):
    """removebg — mock rembg.remove so no neural net is invoked."""

    def test_saves_remove_png(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            src = os.path.join(tmpdir, "garment.jpg")
            make_rgba_image(h=20, w=20).convert("RGB").save(src)
            with mock.patch("image_processing.remove", side_effect=lambda x: x):
                out = ip.removebg(src)
            self.assertTrue(out.endswith("_remove.png"))
            self.assertTrue(os.path.isfile(out))

    def test_raises_on_missing_file(self):
        with self.assertRaises(FileNotFoundError):
            ip.removebg("/nonexistent/image.jpg")


class TestProcessImages(unittest.TestCase):
    """process_images — mock removebg to avoid rembg dependency."""

    def test_skips_existing_remove_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # put a _remove.png in there — it should not be re-processed
            skip = os.path.join(tmpdir, "already_remove.png")
            make_rgba_image().save(skip)
            real = os.path.join(tmpdir, "garment.jpg")
            make_rgba_image().convert("RGB").save(real)

            processed = []
            def fake_removebg(path):
                processed.append(path)
                out = path.replace(".jpg", "_remove.png")
                make_rgba_image().save(out)
                return out

            with mock.patch("image_processing.removebg", side_effect=fake_removebg):
                ip.process_images(tmpdir)

            self.assertEqual(len(processed), 1)
            self.assertIn("garment.jpg", processed[0])

    def test_raises_on_empty_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(ValueError):
                ip.process_images(tmpdir)


class TestPurgeImages(unittest.TestCase):

    def test_purge_removes_only_remove_files(self):
        with tempfile.TemporaryDirectory() as img_dir, \
             tempfile.TemporaryDirectory() as out_dir:
            keeper = os.path.join(img_dir, "garment.jpg")
            removee = os.path.join(img_dir, "garment_remove.png")
            make_rgba_image().convert("RGB").save(keeper)
            make_rgba_image().save(removee)

            ip.purge_images(img_dir, out_dir, purge_all=False)

            self.assertTrue(os.path.isfile(keeper))
            self.assertFalse(os.path.isfile(removee))

    def test_purge_all_removes_everything(self):
        with tempfile.TemporaryDirectory() as img_dir, \
             tempfile.TemporaryDirectory() as out_dir:
            make_rgba_image().convert("RGB").save(os.path.join(img_dir, "a.jpg"))
            make_rgba_image().save(os.path.join(img_dir, "b_remove.png"))
            make_rgba_image().save(os.path.join(img_dir, "c.tif"))

            ip.purge_images(img_dir, out_dir, purge_all=True)

            self.assertEqual(os.listdir(img_dir), [])


# ===========================================================================
# Integration — image_prep -> sample_pixels -> kmeans_clustering
# ===========================================================================

class TestPipelineIntegration(unittest.TestCase):
    """Light integration test: image_prep feeds into clustering correctly."""

    def test_full_prep_to_kmeans(self):
        pil_img = make_rgba_image(h=50, w=50, rgb=(200, 100, 50))
        ab_pixels, mask = ip.image_prep(pil_img, mask_threshold=128)
        sample          = ip.sample_pixels(ab_pixels, rand_int=0)
        _, labels       = ip.kmeans_clustering(sample, ab_pixels,
                                               n_clusters=2, rand_int=0)
        self.assertEqual(labels.shape[0], ab_pixels.shape[0])
        self.assertLessEqual(labels.max(), 1)

    def test_apply_ccm_feeds_into_image_prep(self):
        """CCM-corrected PIL Image passes through image_prep without error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "garment.png")
            make_rgba_image(h=30, w=30).save(path)
            corrected = ip.apply_ccm_to_image(path, ccm=make_ccm())
            ab_pixels, mask = ip.image_prep(corrected, mask_threshold=128)
        self.assertGreater(ab_pixels.shape[0], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)