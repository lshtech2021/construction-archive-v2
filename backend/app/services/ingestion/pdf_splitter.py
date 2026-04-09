import os
from pathlib import Path

import fitz  # PyMuPDF

from app.schemas.ingestion import PageImage


class PDFSplitter:
    def split_to_images(
        self, pdf_path: str, output_dir: str, dpi: int = 300
    ) -> list[PageImage]:
        """Rasterize each page of a PDF to a JPEG at the given DPI."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        doc = fitz.open(pdf_path)
        results: list[PageImage] = []

        for page_index in range(len(doc)):
            page = doc[page_index]
            mat = fitz.Matrix(dpi / 72, dpi / 72)
            pixmap = page.get_pixmap(matrix=mat)
            image_path = os.path.join(output_dir, f"page_{page_index:03d}.jpg")
            pixmap.save(image_path, "jpeg")
            results.append(PageImage(page_index=page_index, image_path=image_path))

        doc.close()
        return results
