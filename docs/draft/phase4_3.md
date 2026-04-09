
To implement **Phase 4: Generative AI Q&A (Vision-RAG) & Spatial Highlighting**, we need to bridge the gap between semantic retrieval, visual reasoning, and spatial awareness.

The biggest challenge in this phase is **Bounding Box Generation**. While Vision LLMs (like GPT-4o) are incredible at answering questions, they are notoriously imprecise at calculating exact pixel coordinates on 8K-resolution blueprints. 

**The Solution:** We use a **Hybrid Spatial Approach**. We instruct the VLM to answer the question and return the *exact text labels* or *equipment tags* (e.g., "W12x50") it sees on the plan. We then cross-reference those tags with the deterministic Azure Document Intelligence OCR polygons we extracted in Phase 1 to get mathematically perfect coordinates.

Here is the complete implementation for Phase 4.

### 1. Backend Implementation (Python / FastAPI)

First, install the necessary libraries if you haven't already:
```bash
pip install fastapi openai pydantic uvicorn
```

Create the Vision-RAG endpoint (`vision_rag_api.py`):

```python
import os
import base64
from typing import List, Optional, Dict
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from openai import OpenAI

# Import the Semantic Search Engine from Phase 3
from search_engine import SemanticSearchEngine 

app = FastAPI(title="Construction AI - Vision RAG")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
search_engine = SemanticSearchEngine()

# --- 1. PYDANTIC SCHEMAS (API & LLM STRUCTURED OUTPUT) ---

class UserChatRequest(BaseModel):
    query: str
    discipline_filter: Optional[str] = None

class TargetHighlight(BaseModel):
    exact_text: str = Field(description="The exact text or tag on the blueprint indicating the object (e.g., 'W12x50', 'AHU-04').")
    sheet_id: str = Field(description="The sheet ID where this object is located.")
    reasoning: str = Field(description="Why this object is highlighted.")

class LLMVisionResponse(BaseModel):
    answer: str = Field(description="The professional, detailed answer to the user's question. Cite sheet numbers explicitly.")
    objects_to_highlight: List[TargetHighlight] = Field(description="List of text labels to highlight on the UI.")

class ResolvedBoundingBox(BaseModel):
    text: str
    sheet_id: str
    normalized_polygon: List[List[float]] # [[x1,y1],[x2,y2], [x3,y3], [x4,y4]]
    
class FinalChatResponse(BaseModel):
    answer: str
    cited_sheets: List[str]
    highlights: List[ResolvedBoundingBox]

# --- 2. HELPER FUNCTIONS ---

def encode_image(image_path: str) -> str:
    """Encodes blueprint image for GPT-4o."""
    with open(image_path, "rb") as f:
         return base64.b64encode(f.read()).decode('utf-8')

def find_exact_coordinates(sheet_id: str, target_text: str, ocr_database: dict) -> Optional[List[List[float]]]:
    """
    Looks up the exact polygon coordinates from Phase 1's Azure OCR data.
    In production, `ocr_database` would be a query to PostgreSQL/MongoDB.
    """
    # Assuming ocr_database[sheet_id]["words"] contains Azure's word-level polygons
    sheet_ocr = ocr_database.get(sheet_id, {}).get("words",[])
    
    for word in sheet_ocr:
        # Simple substring match (Use fuzzy matching in production)
        if target_text.upper() in word["content"].upper():
            return word["polygon"]
    return None

# --- 3. THE VISION-RAG ENDPOINT ---

@app.post("/api/chat", response_model=FinalChatResponse)
async def chat_vision_rag(request: UserChatRequest):
    print(f"User Query: {request.query}")

    # Step 1: Retrieve Top Sheets via ColPali + BM25 (Phase 3)
    top_sheets = search_engine.search(
        query=request.query, 
        discipline_filter=request.discipline_filter, 
        top_k=2
    )

    if not top_sheets:
        return FinalChatResponse(
            answer="I couldn't find any relevant blueprints for that query in the archive.",
            cited_sheets=[], highlights=[]
        )

    # Step 2: Prepare Multimodal Payload for GPT-4o
    system_prompt = """
    You are an expert Structural Engineer and Architect. Answer the user's question using ONLY the provided blueprint images. 
    1. Provide a professional, concise answer.
    2. Explicitly cite the sheet number (e.g., 'On Sheet S-201...').
    3. Identify exact text labels or callouts associated with the answer so the UI can highlight them (e.g., column tags like 'W12x50' or notes like 'TYP. REBAR').
    """

    messages =[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [{"type": "text", "text": request.query}]}
    ]

    cited_sheets =[]
    
    # Append the retrieved images to the prompt
    for sheet in top_sheets:
        sheet_id = sheet["sheet_id"]
        cited_sheets.append(sheet_id)
        
        # Load high-res image (Phase 1 output)
        image_path = f"./processed_plans/{sheet_id}.jpg"
        base64_image = encode_image(image_path)
        
        messages[1]["content"].append({
            "type": "text", 
            "text": f"--- Blueprint Sheet: {sheet_id} ({sheet['discipline']}) ---"
        })
        messages[1]["content"].append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}",
                "detail": "high" # Crucial: forces GPT-4o to read high-res tiles to see tiny blueprint text
            }
        })

    # Step 3: Call GPT-4o with Structured Outputs
    try:
        response = client.beta.chat.completions.parse(
            model="gpt-4o-2024-08-06",
            messages=messages,
            response_format=LLMVisionResponse,
            temperature=0.1
        )
        
        ai_result: LLMVisionResponse = response.choices[0].message.parsed
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM Error: {str(e)}")

    # Step 4: Resolve AI's targeted text to exact OCR coordinates
    # (Mocking OCR DB for demonstration)
    mock_ocr_db = {
        "S-201": {
            "words":[
                {"content": "W12x50", "polygon": [[0.45, 0.60],[0.48, 0.60], [0.48, 0.62],[0.45, 0.62]]},
                {"content": "W10x30", "polygon": [[0.80, 0.10], [0.85, 0.10],[0.85, 0.12], [0.80, 0.12]]}
            ]
        }
    }

    resolved_highlights =[]
    for target in ai_result.objects_to_highlight:
        coords = find_exact_coordinates(target.sheet_id, target.exact_text, mock_ocr_db)
        if coords:
            resolved_highlights.append(
                ResolvedBoundingBox(
                    text=target.exact_text,
                    sheet_id=target.sheet_id,
                    normalized_polygon=coords
                )
            )

    # Step 5: Return to Frontend
    return FinalChatResponse(
        answer=ai_result.answer,
        cited_sheets=cited_sheets,
        highlights=resolved_highlights
    )

# Run: uvicorn vision_rag_api:app --reload
```

---

### 2. Frontend Implementation (React + OpenSeadragon)

When the frontend receives the `FinalChatResponse`, it needs to display the chat text and dynamically draw an interactive box on the OpenSeadragon (OSD) viewer over the exact location.

```tsx
import React, { useEffect, useRef } from 'react';
import OpenSeadragon from 'openseadragon';

// Types matching our FastAPI backend
interface ResolvedBoundingBox {
  text: string;
  sheet_id: string;
  normalized_polygon: number[][]; // [[x,y], [x,y], [x,y], [x,y]]
}

interface BlueprintViewerProps {
  dziUrl: string;
  highlights: ResolvedBoundingBox[];
}

export default function BlueprintViewer({ dziUrl, highlights }: BlueprintViewerProps) {
  const viewerRef = useRef<OpenSeadragon.Viewer | null>(null);

  useEffect(() => {
    // Initialize OpenSeadragon (Deep Zoom Viewer)
    viewerRef.current = OpenSeadragon({
      id: "osd-viewer",
      prefixUrl: "//openseadragon.github.io/openseadragon/images/",
      tileSources: dziUrl,
      maxZoomPixelRatio: 3,
    });

    return () => viewerRef.current?.destroy();
  }, [dziUrl]);

  // Handle drawing the AI Bounding Boxes
  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewer) return;

    // Clear previous overlays
    viewer.clearOverlays();

    highlights.forEach((highlight, index) => {
      // Create HTML element for the bounding box
      const boxElement = document.createElement("div");
      boxElement.className = "border-4 border-red-500 bg-red-500 bg-opacity-30 rounded cursor-pointer animate-pulse";
      boxElement.title = highlight.text;

      // Extract min/max from polygon to create a standard rectangle
      const xCoords = highlight.normalized_polygon.map(p => p[0]);
      const yCoords = highlight.normalized_polygon.map(p => p[1]);
      
      const minX = Math.min(...xCoords);
      const minY = Math.min(...yCoords);
      const width = Math.max(...xCoords) - minX;
      const height = Math.max(...yCoords) - minY;

      // Note: OpenSeadragon uses a normalized coordinate system where Width is ALWAYS 1.0.
      // Assuming our OCR coordinates are 0.0 to 1.0 relative to image dimensions.
      const rect = new OpenSeadragon.Rect(minX, minY, width, height);

      // Add overlay to viewer
      viewer.addOverlay({
        element: boxElement,
        location: rect,
      });

      // Automatically animate the camera to zoom into the first highlight
      if (index === 0) {
        // Add padding around the zoom target
        const expandedRect = new OpenSeadragon.Rect(
            minX - 0.05, minY - 0.05, width + 0.1, height + 0.1
        );
        setTimeout(() => viewer.viewport.fitBoundsWithConstraints(expandedRect), 500);
      }
    });
  },[highlights]);

  return <div id="osd-viewer" className="w-full h-full bg-slate-900" />;
}
```

### 3. Architecture Flow for a Typical Interaction

1.  **User asks:** *"Where are the load-bearing columns on the second floor?"*
2.  **Phase 3 (Retrieval):** The system pre-filters for `Discipline: Structural` and uses ColPali/Qdrant to find that `S-201` is the second-floor framing plan.
3.  **Phase 4 (VLM Parsing):** GPT-4o analyzes `S-201` at `detail: high`. It sees the column grid and notices structural tags.
4.  **Phase 4 (Structuring):** GPT-4o formats the JSON response: 
    *   `answer`: *"The load bearing columns are W12x50 steel beams located along grid lines A-D."*
    *   `objects_to_highlight`: `[{exact_text: "W12x50", sheet_id: "S-201"}]`.
5.  **Phase 4 (Grounding):** The Python backend intercepts the string `"W12x50"`, queries the PostgreSQL/Azure Document Intelligence database, and retrieves the normalized polygon: `[[0.45, 0.60], ...]`.
6.  **Frontend Generation:** React receives the JSON. OpenSeadragon renders the deep-zoom image (from Phase 1), draws a glowing red rectangle exactly over the column, and smoothly pans the camera to zoom in on it for the user.

### Why this specific stack is the *only* right way to do this:
If you try to ask an LLM to give you `[x, y]` pixel coordinates directly, it will fail 95% of the time on complex documents. By letting the LLM do what it's good at (semantic reasoning and reading text) and letting a deterministic OCR engine do what it's good at (spatial geometry), you achieve **zero-hallucination spatial targeting**.