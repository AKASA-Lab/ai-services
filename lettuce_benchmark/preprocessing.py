"""Shared image preprocessing that removes dataset-specific shortcut cues.

See ``docs/dataset_audit.md`` §4: every 1024x1024 image carries a DALL·E colour bar in
the bottom-right corner and that subset is strongly class-dependent. Cropping every image
to the same fixed window removes the bar and the aspect-ratio difference.
"""

from __future__ import annotations

from PIL import Image

SAFE_CROP_SIZE = 960


def watermark_safe_crop(image: Image.Image, size: int = SAFE_CROP_SIZE) -> Image.Image:
    """Crop a ``size`` x ``size`` window, horizontally centred and starting at the top row.

    The smallest image in the archive is 1015x999, so a 960 window always fits and always
    excludes the bottom 39+ rows, where the 16-pixel colour bar sits.
    """
    width, height = image.size
    if width < size or height < size:
        raise ValueError(f"Image {width}x{height} is smaller than the {size}px safe crop")
    left = (width - size) // 2
    return image.crop((left, 0, left + size, size))
