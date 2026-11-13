from PIL import Image

from lettuce_benchmark.preprocessing import watermark_safe_crop


def test_safe_crop_is_fixed_size_and_drops_bottom_right_corner() -> None:
    image = Image.new("RGB", (1024, 1024), color=(0, 0, 0))
    image.paste((255, 255, 0), (944, 1008, 1024, 1024))  # colour bar location
    cropped = watermark_safe_crop(image)
    assert cropped.size == (960, 960)
    assert cropped.getextrema() == ((0, 0), (0, 0), (0, 0))


def test_safe_crop_handles_smallest_archive_size() -> None:
    assert watermark_safe_crop(Image.new("RGB", (1015, 999))).size == (960, 960)
