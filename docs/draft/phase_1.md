


To implement **Phase 1: Intelligent PDF Ingestion Pipeline**, we will build a Python-based processing engine. 

This implementation uses **PyMuPDF (`fitz`)** for blazingly fast PDF splitting, **`pyvips`** to generate Deep Zoom Image (DZI) pyramids for the frontend viewer, and **OpenAI's GPT-4o** (Vision) with **Pydantic** to reliably extract Title Block metadata and automatically route disciplines.

### 1. Prerequisites & Installation

First, install the necessary Python libraries:
```bash
pip install pymupdf openai pydantic pyvips python-dotenv azure-ai-documentintelligence
```
*(Note: `pyvips` requires the `libvips` system dependency. On Ubuntu: `sudo apt install libvips`, on Mac: `brew install vips`).*

### 2. Python Implementation (`ingestion_pipeline.py`)

```python
import os
import re
import fitz  # PyMuPDF
import pyvips
import base64
import json
from concurrent.futures import ThreadPoolExecutor
from pydantic import BaseModel, Field
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Initialize OpenAI Client (for Title Block Extraction)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- PYDANTIC MODELS FOR AI METADATA EXTRACTION ---

class SheetMetadata(BaseModel):
    sheet_number: str = Field(description="The architectural sheet number, e.g., A-101, S2.0, MEP-01")
    sheet_title: str = Field(description="The title of the sheet, e.g., First Floor Plan, Section Details")
    revision_date: str = Field(description="The latest revision date found in the title block. Use YYYY-MM-DD if possible, or 'Unknown'.")
    scale: str = Field(description="The primary scale noted, e.g., 1/8\" = 1'-0\" or 'As Noted'.")

class ProcessedSheet(BaseModel):
    page_index: int
    image_path: str
    dzi_path: str
    metadata: SheetMetadata
    discipline: str
    full_text_ocr_path: str = ""

# --- PIPELINE CLASS ---

class ConstructionIngestionPipeline:
    def __init__(self, output_dir: str = "./archive_output"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def _encode_image_to_base64(self, image_path: str) -> str:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def get_discipline(self, sheet_number: str) -> str:
        """Rule-based routing for discipline grouping based on standard naming conventions."""
        sheet_num_upper = sheet_number.upper()
        if re.match(r'^A[-0-9]', sheet_num_upper): return "Architectural"
        if re.match(r'^S[-0-9]', sheet_num_upper): return "Structural"
        if re.match(r'^M[-0-9]', sheet_num_upper): return "Mechanical"
        if re.match(r'^E[-0-9]', sheet_num_upper): return "Electrical"
        if re.match(r'^P[-0-9]', sheet_num_upper): return "Plumbing"
        if re.match(r'^C[-0-9]', sheet_num_upper): return "Civil"
        if re.match(r'^L[-0-9]', sheet_num_upper): return "Landscape"
        if re.match(r'^FP[-0-9]', sheet_num_upper): return "Fire Protection"
        return "General / Undefined"

    def extract_metadata_with_vision(self, image_path: str) -> SheetMetadata:
        """Uses GPT-4o Vision to read the title block and extract structured metadata."""
        base64_image = self._encode_image_to_base64(image_path)
        
        response = client.beta.chat.completions.parse(
            model="gpt-4o-2024-08-06",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert construction project manager. Examine the provided construction plan sheet, locate the Title Block (usually bottom right or right edge), and extract the required metadata."
                },
                {
                    "role": "user",
                    "content":[
                        {"type": "text", "text": "Extract the Sheet Number, Title, Revision Date, and Scale."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}", "detail": "high"}}
                    ]
                }
            ],
            response_format=SheetMetadata,
            temperature=0.1
        )
        
        return response.choices[0].message.parsed

    def generate_deep_zoom(self, source_image_path: str, output_prefix: str) -> str:
        """Uses pyvips to generate OpenSeadragon-compatible Deep Zoom Images (DZI)."""
        image = pyvips.Image.new_from_file(source_image_path)
        dzi_output_path = os.path.join(self.output_dir, "tiles", output_prefix)
        os.makedirs(os.path.dirname(dzi_output_path), exist_ok=True)
        
        # Generates a .dzi file and an _files folder with the image pyramid
        image.dzsave(dzi_output_path)
        return dzi_output_path + ".dzi"

    def run_deep_ocr(self, document_path: str):
        """
        Placeholder for Azure Document Intelligence.
        In production, send the image/PDF here to extract all text, tables, and bounding boxes.
        """
        # from azure.ai.documentintelligence import DocumentIntelligenceClient
        # client = DocumentIntelligenceClient(endpoint=endpoint, credential=credential)
        # poller = client.begin_analyze_document("prebuilt-layout", analyze_request=document_path)
        # return poller.result()
        return "OCR data successfully saved to DB."

    def process_page(self, page_index: int, page: fitz.Page) -> ProcessedSheet:
        """Process a single page: rasterize, extract metadata, generate DZI, assign discipline."""
        print(f"--> Processing Page {page_index + 1}...")
        
        # 1. Rasterize Page to High-Res JPEG (300 DPI)
        pix = page.get_pixmap(dpi=300)
        image_filename = f"page_{page_index}.jpg"
        image_path = os.path.join(self.output_dir, image_filename)
        pix.save(image_path)

        # 2. Vision AI Metadata Extraction (Title Block)
        print(f"    Extracting Metadata for Page {page_index + 1}...")
        metadata = self.extract_metadata_with_vision(image_path)

        # 3. Discipline Grouping
        discipline = self.get_discipline(metadata.sheet_number)

        # 4. Generate Deep Zoom Tiles for UI
        print(f"    Generating Deep Zoom Tiles for {metadata.sheet_number}...")
        dzi_path = self.generate_deep_zoom(image_path, f"{metadata.sheet_number}_tiles")

        # 5. Run Deep OCR (Azure)
        # ocr_result = self.run_deep_ocr(image_path)

        processed = ProcessedSheet(
            page_index=page_index,
            image_path=image_path,
            dzi_path=dzi_path,
            metadata=metadata,
            discipline=discipline
        )
        
        print(f"    [Success] {metadata.sheet_number}: {metadata.sheet_title} ({discipline})")
        return processed

    def process_pdf(self, pdf_path: str):
        """Main orchestrator: loads PDF, splits pages, and processes them concurrently."""
        doc = fitz.open(pdf_path)
        print(f"Started ingestion of {pdf_path}. Total pages: {len(doc)}")
        
        processed_sheets =[]
        
        # Using ThreadPoolExecutor to process pages in parallel
        # (Ideal for network-bound tasks like calling OpenAI/Azure APIs)
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures =[]
            for i in range(len(doc)):
                # Load page into memory to pass to processor
                futures.append(executor.submit(self.process_page, i, doc.load_page(i)))
            
            for future in futures:
                processed_sheets.append(future.result())
                
        # Save manifest
        manifest_path = os.path.join(self.output_dir, "ingestion_manifest.json")
        with open(manifest_path, "w") as f:
            json.dump([sheet.model_dump() for sheet in processed_sheets], f, indent=4)
            
        print(f"\nIngestion Complete. Manifest saved to {manifest_path}")
        return processed_sheets

# --- USAGE EXAMPLE ---
if __name__ == "__main__":
    # Ensure you have a sample construction PDF named 'sample_plans.pdf' and OPENAI_API_KEY in .env
    pipeline = ConstructionIngestionPipeline(output_dir="./processed_plans")
    
    try:
        results = pipeline.process_pdf("sample_plans.pdf")
    except Exception as e:
        print(f"Pipeline failed: {e}")
```

### 3. Key Technical Decisions in this Code:

1.  **Concurrent Processing (`ThreadPoolExecutor`)**: Processing a 100-page PDF sequentially takes too long. Because extracting metadata and deep OCR are network-bound (waiting on OpenAI/Azure APIs), utilizing Thread Pools vastly speeds up ingestion.
2.  **`pyvips` for Deep Zoom Generation**: Standard JPEG viewers crash when users try to zoom into an $8000 \times 6000$ pixel blueprint. `pyvips` converts the image into an image pyramid (`.dzi`), breaking it down into hundreds of small $256 \times 256$ tiles. The frontend viewer (OpenSeadragon) will load *only* the tiles the user is currently zooming into.
3.  **Structured Outputs (`pydantic` + OpenAI 2024-08-06 model)**: We use OpenAI's new `response_format` feature with Pydantic. This guarantees that the LLM returns perfectly formatted JSON matching our `SheetMetadata` schema, completely eliminating parsing errors.
4.  **Deterministic Routing**: The `get_discipline` regex relies on the AI extracting the string (e.g., `S-101`) perfectly. Once extracted, the Regex applies standard AIA (American Institute of Architects) naming conventions to tag the file.

### 4. Next Steps for Phase 1 (Productionization):
To push this into a production environment, you would wrap the `process_pdf` execution in a **Celery Worker**. 
When a user uploads a PDF in the Next.js frontend, the FastAPI backend will save the PDF to AWS S3, drop a message into a Redis queue, and immediately return a `202 Accepted` to the user. The Celery worker will then pick up this script, process the file in the background, upload the `.dzi` tiles back to S3, and save the manifest to PostgreSQL.