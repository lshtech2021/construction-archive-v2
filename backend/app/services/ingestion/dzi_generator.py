import os
from pathlib import Path

import pyvips


class DZIGenerator:
    def generate(self, image_path: str, output_prefix: str, output_dir: str) -> str:
        """Generate a Deep Zoom Image pyramid from a JPEG.

        Returns the path to the .dzi manifest file.
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        image = pyvips.Image.new_from_file(image_path, access="sequential")
        output_base = os.path.join(output_dir, output_prefix)
        image.dzsave(
            output_base,
            tile_size=256,
            overlap=1,
            depth="onetile",
            suffix=".jpg",
        )
        return f"{output_base}.dzi"
