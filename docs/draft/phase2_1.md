


To implement **Phase 2: Automated Discipline Grouping & Graph Linking**, we need to build a hybrid routing engine. 

Construction documents are notoriously inconsistent. Modern projects follow the **U.S. National CAD Standard (NCS)** (e.g., `A-101` for Architectural), but older or smaller projects might just use `Page 1, Page 2` or custom codes. 

Here is the complete implementation in Python. It uses deterministic **Regex** first (zero cost, 1 millisecond), falls back to **GPT-4o-mini** (low cost, vision-based) if regex fails, and uses **NetworkX / PostgreSQL** concepts to build the relational graph of sheet hyperlinks.

### 1. Python Implementation (`discipline_graph_engine.py`)

First, install the necessary libraries:
```bash
pip install pydantic openai networkx pillow
```

```python
import os
import re
import base64
from enum import Enum
from typing import List, Optional, Tuple
from pydantic import BaseModel, Field
from openai import OpenAI
from PIL import Image
import networkx as nx

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- 1. SCHEMAS & ENUMS ---

class DisciplineEnum(str, Enum):
    ARCHITECTURAL = "Architectural"
    STRUCTURAL = "Structural"
    MECHANICAL = "Mechanical"
    ELECTRICAL = "Electrical"
    PLUMBING = "Plumbing"
    FIRE_PROTECTION = "Fire Protection"
    CIVIL = "Civil"
    LANDSCAPE = "Landscape"
    GENERAL = "General"
    UNKNOWN = "Unknown"

class SemanticClassification(BaseModel):
    discipline: DisciplineEnum = Field(description="The assigned construction discipline.")
    confidence: float = Field(description="Confidence score between 0.0 and 1.0")
    reasoning: str = Field(description="Why this discipline was chosen based on the image and title.")

class SheetNode(BaseModel):
    sheet_id: str
    sheet_title: str
    discipline: DisciplineEnum

class CalloutEdge(BaseModel):
    source_sheet: str
    target_sheet: str
    context_text: str
    bounding_box: List[float] # [x, y, w, h]

# --- 2. THE ENGINE ---

class DisciplineAndGraphManager:
    def __init__(self):
        # We use NetworkX in-memory to build the graph before saving to a Database
        self.project_graph = nx.DiGraph()

    # ==========================================
    # STEP 1: DETERMINISTIC RULE-BASED ROUTING
    # ==========================================
    def get_discipline_by_regex(self, sheet_number: str) -> Optional[DisciplineEnum]:
        """
        Follows U.S. National CAD Standard (NCS) heuristics.
        Returns None if the format is non-standard.
        """
        sheet_num_upper = sheet_number.upper().strip()
        
        # e.g., A101, A-101, A2.0
        if re.match(r'^A[-.\s]?\d+', sheet_num_upper): return DisciplineEnum.ARCHITECTURAL
        if re.match(r'^S[-.\s]?\d+', sheet_num_upper): return DisciplineEnum.STRUCTURAL
        if re.match(r'^M[-.\s]?\d+', sheet_num_upper): return DisciplineEnum.MECHANICAL
        if re.match(r'^E[-.\s]?\d+', sheet_num_upper): return DisciplineEnum.ELECTRICAL
        if re.match(r'^P[-.\s]?\d+', sheet_num_upper): return DisciplineEnum.PLUMBING
        if re.match(r'^C[-.\s]?\d+', sheet_num_upper): return DisciplineEnum.CIVIL
        if re.match(r'^L[-.\s]?\d+', sheet_num_upper): return DisciplineEnum.LANDSCAPE
        if re.match(r'^FP[-.\s]?\d+', sheet_num_upper): return DisciplineEnum.FIRE_PROTECTION
        if re.match(r'^G[-.\s]?\d+', sheet_num_upper): return DisciplineEnum.GENERAL
        
        return None # Trigger Semantic Fallback

    # ==========================================
    # STEP 2: SEMANTIC AI FALLBACK
    # ==========================================
    def _resize_image_for_ai(self, image_path: str, max_size=(1024, 1024)) -> str:
        """Resizes high-res blueprint to low-res to save massive LLM token costs."""
        temp_path = "temp_low_res.jpg"
        with Image.open(image_path) as img:
            img.thumbnail(max_size)
            img.save(temp_path, format="JPEG", quality=85)
            
        with open(temp_path, "rb") as image_file:
            b64_str = base64.b64encode(image_file.read()).decode('utf-8')
            
        os.remove(temp_path)
        return b64_str

    def get_discipline_semantically(self, sheet_title: str, image_path: str) -> DisciplineEnum:
        """
        Uses gpt-4o-mini (fast/cheap) to classify non-standard sheets.
        e.g., Sheet Number is "001", but Title is "Foundation Rebar Details".
        """
        print(f"Regex failed. Falling back to semantic AI for '{sheet_title}'...")
        base64_image = self._resize_image_for_ai(image_path)

        try:
            response = client.beta.chat.completions.parse(
                model="gpt-4o-mini", # Lightweight model for basic classification
                messages=[
                    {
                        "role": "system",
                        "content": "You are a construction discipline classifier. Look at the blueprint image and the title. Classify it into exactly one standard discipline. Look for visual cues: rebar/concrete = Structural, toilets/pipes = Plumbing, ducts/HVAC = Mechanical."
                    },
                    {
                        "role": "user",
                        "content":[
                            {"type": "text", "text": f"Sheet Title extracted from OCR: '{sheet_title}'"},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}", "detail": "low"}}
                        ]
                    }
                ],
                response_format=SemanticClassification,
                temperature=0.1
            )
            
            result: SemanticClassification = response.choices[0].message.parsed
            print(f"AI Classification: {result.discipline} (Confidence: {result.confidence}) - {result.reasoning}")
            
            # Require high confidence, else default to General
            if result.confidence > 0.7:
                return result.discipline
            else:
                return DisciplineEnum.GENERAL
                
        except Exception as e:
            print(f"AI Classification failed: {e}")
            return DisciplineEnum.UNKNOWN

    # ==========================================
    # STEP 3: CALLOUT GRAPH LINKING
    # ==========================================
    def extract_and_build_graph(self, ocr_data: dict, current_sheet_num: str, all_project_sheets: set):
        """
        Scans all text blocks on a sheet for callouts (e.g., 'A-201')
        If the referenced sheet exists in the project, an Edge is created in the Graph.
        """
        # Ensure current sheet is a node
        if not self.project_graph.has_node(current_sheet_num):
            self.project_graph.add_node(current_sheet_num)

        # Regex to find potential sheet references (e.g., "See A-101", "4/S-200")
        callout_pattern = re.compile(r'\b([A-Z]{1,2}[-\s]?\d{1,3}[A-Z]?)\b')

        found_edges =[]

        for word in ocr_data.get("words",[]):
            text = word.get("content", "")
            matches = callout_pattern.findall(text)

            for match in matches:
                # Normalize "A - 101" to "A-101"
                target_sheet = match.replace(" ", "").upper()

                # Rule out self-references and verify target exists in project
                if target_sheet != current_sheet_num and target_sheet in all_project_sheets:
                    
                    # Convert OCR Polygon to simplified [x, y, w, h]
                    poly = word["polygon"]
                    x_coords = [p[0] for p in poly]
                    y_coords = [p[1] for p in poly]
                    bbox =[min(x_coords), min(y_coords), max(x_coords)-min(x_coords), max(y_coords)-min(y_coords)]

                    edge = CalloutEdge(
                        source_sheet=current_sheet_num,
                        target_sheet=target_sheet,
                        context_text=text,
                        bounding_box=bbox
                    )
                    found_edges.append(edge)

                    # Add directed edge to the graph: Source -> Target
                    self.project_graph.add_edge(
                        current_sheet_num, 
                        target_sheet, 
                        context=text, 
                        bbox=bbox
                    )

        return found_edges

# --- USAGE EXAMPLE ---
if __name__ == "__main__":
    manager = DisciplineAndGraphManager()

    # 1. Test Deterministic Routing
    print(manager.get_discipline_by_regex("A-101")) # Returns: DisciplineEnum.ARCHITECTURAL
    print(manager.get_discipline_by_regex("S2.0"))  # Returns: DisciplineEnum.STRUCTURAL
    print(manager.get_discipline_by_regex("001"))   # Returns: None (Non-standard)

    # 2. Test Semantic Fallback (Requires a sample image and OPENAI_API_KEY)
    # discipline = manager.get_discipline_semantically("Foundation Rebar Details", "sample_rebar.jpg")
    
    # 3. Test Graph Linking (Simulated OCR Payload from Phase 1)
    mock_ocr = {
        "words":[
            {"content": "SEE DETAIL 4/S-201", "polygon": [[10,10], [50,10], [50,20], [10,20]]},
            {"content": "ELEVATION A-102", "polygon": [[100,100],[150,100], [150,120], [100,120]]}
        ]
    }
    
    # Simulate a project that contains these 3 sheets
    project_sheets = {"S-201", "A-102", "M-101"} 
    
    edges = manager.extract_and_build_graph(
        ocr_data=mock_ocr, 
        current_sheet_num="A-101", 
        all_project_sheets=project_sheets
    )
    
    print("\nExtracted Graph Links:")
    for edge in edges:
        print(f"Link created: {edge.source_sheet} --> {edge.target_sheet}")
```

### 2. How the Graph is Stored in Production (Database Architecture)

While `NetworkX` is great for in-memory operations during the Python processing phase, you need to store this data permanently. Since this is an enterprise application, you don't necessarily need a dedicated Graph DB like Neo4j unless you are doing complex traversal algorithms. **PostgreSQL** handles this perfectly via Adjacency Lists.

In PostgreSQL, create a table specifically for links:

```sql
CREATE TABLE sheet_hyperlinks (
    id SERIAL PRIMARY KEY,
    source_sheet_id VARCHAR(50) REFERENCES sheets(sheet_number),
    target_sheet_id VARCHAR(50) REFERENCES sheets(sheet_number),
    context_text VARCHAR(255),
    bbox_x REAL,
    bbox_y REAL,
    bbox_w REAL,
    bbox_h REAL
);

-- Indexing for instantaneous graph lookups
CREATE INDEX idx_source_sheet ON sheet_hyperlinks(source_sheet_id);
CREATE INDEX idx_target_sheet ON sheet_hyperlinks(target_sheet_id);
```

### 3. Key Benefits of Phase 2 Implementation

1.  **Cost & Speed Optimization:** 90% of construction sets *do* follow the NCS naming standard. By running the regex engine first, you bypass the LLM entirely for 90% of your pages, saving API costs and slashing ingestion time from hours to minutes.
2.  **`gpt-4o-mini` with Low-Res Images:** When regex *does* fail, sending an 8K resolution blueprint to standard GPT-4o would cost roughly $0.05 per page and take 10 seconds. By downscaling the image to 1024x1024 and passing `detail: "low"` to the lightweight `gpt-4o-mini` model, you reduce the cost to less than `$0.001` per page, and it returns the correct discipline in < 1 second.
3.  **Relational Context for AI (Vision-RAG enhancement):** Building this graph doesn't just benefit the user's UI clicks. You can feed this graph logic to the RAG system in Phase 3. If a user asks *"How is the AHU mounted?"*, the system can retrieve the Mechanical plan, follow the database graph edge to the structural detail sheet, and feed *both* images to GPT-4o to give a comprehensive, multi-sheet answer.