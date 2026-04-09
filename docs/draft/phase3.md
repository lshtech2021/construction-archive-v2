



To implement **Phase 3: Generative AI Q&A Chat & Viewer**, we must bridge the backend AI engine with the frontend user interface. 

This phase consists of two main parts:
1.  **The Vision-RAG Backend (FastAPI)**: An orchestrator that takes a user’s question, retrieves the relevant sheets using Phase 2's search engine, and feeds the actual blueprint images into GPT-4o to generate a highly accurate, context-aware answer.
2.  **The High-Performance Viewer (React + OpenSeadragon)**: The frontend component that loads the Deep Zoom Images (DZI) generated in Phase 1, allowing the user to seamlessly pan and zoom across massive blueprints while chatting with the AI.

### 1. The Vision-RAG Backend Orchestrator (Python / FastAPI)

First, install FastAPI and an ASGI server:
```bash
pip install fastapi uvicorn pydantic openai
```

Here is the implementation of the RAG Chat API (`chat_api.py`):

```python
import os
import base64
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI

# Import the search engine from Phase 2
from search_engine import SemanticSearchEngine, SearchResult 

app = FastAPI(title="Construction AI Archive - Chat API")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
search_engine = SemanticSearchEngine()

# --- PYDANTIC SCHEMAS ---

class ChatRequest(BaseModel):
    query: str
    discipline_filter: Optional[str] = None
    chat_history: Optional[List[dict]] =[]

class Citation(BaseModel):
    sheet_number: str
    sheet_title: str
    dzi_path: str

class ChatResponse(BaseModel):
    answer: str
    citations: List[Citation]

# --- HELPER FUNCTIONS ---

def encode_image_to_base64(image_path: str) -> str:
    """Reads a local image and encodes it for GPT-4o Vision."""
    # Note: In production, download from S3 to memory if not local.
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        return ""

# --- API ENDPOINTS ---

@app.post("/api/chat", response_model=ChatResponse)
async def chat_with_blueprints(request: ChatRequest):
    print(f"Received query: {request.query}")

    # 1. Retrieve the top 2 most relevant blueprint sheets
    top_sheets: List[SearchResult] = search_engine.search(
        query=request.query,
        discipline_filter=request.discipline_filter,
        limit=2  # Keep limit low to avoid exceeding LLM vision token limits
    )

    if not top_sheets:
        return ChatResponse(
            answer="I couldn't find any relevant blueprints for your query.",
            citations=[]
        )

    # 2. Prepare Multimodal Prompt for GPT-4o
    system_prompt = """
    You are an expert Principal Architect and Construction Manager.
    You are assisting a user in navigating a construction archive.
    I will provide you with a user query, along with images and metadata of the most relevant blueprint sheets.
    
    RULES:
    1. Base your answer ONLY on the provided blueprint images and metadata.
    2. Explicitly cite the sheet number in your response (e.g., "According to Sheet A-101...").
    3. DO NOT hallucinate dimensions. Only state measurements if they are explicitly written on the plan.
    4. Keep your answer professional, concise, and focused on the user's specific question.
    """

    # Build the message content list dynamically
    user_content =[{"type": "text", "text": f"User Query: {request.query}\n\nHere are the relevant sheets:"}]
    citations =[]

    for sheet in top_sheets:
        # Reconstruct original high-res image path (Assumes Phase 1 saved it alongside DZI)
        # E.g., if dzi_path is /tiles/A-101.dzi, the image might be A-101.jpg
        base_filename = sheet.dzi_path.split("/")[-1].replace(".dzi", ".jpg")
        image_path = os.path.join("./processed_plans", base_filename) 
        
        base64_image = encode_image_to_base64(image_path)
        
        if base64_image:
            user_content.append({"type": "text", "text": f"--- Sheet {sheet.sheet_number}: {sheet.sheet_title} ({sheet.discipline}) ---"})
            user_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}",
                    "detail": "high" # "high" allows GPT-4o to read microscopic blueprint text
                }
            })
            citations.append(Citation(
                sheet_number=sheet.sheet_number,
                sheet_title=sheet.sheet_title,
                dzi_path=sheet.dzi_path
            ))

    # 3. Call GPT-4o Vision-RAG
    try:
        response = client.chat.completions.create(
            model="gpt-4o-2024-08-06",
            messages=[
                {"role": "system", "content": system_prompt},
                *request.chat_history, # Append history for conversational context
                {"role": "user", "content": user_content}
            ],
            temperature=0.2, # Low temperature for factual accuracy
        )
        
        answer = response.choices[0].message.content

        return ChatResponse(answer=answer, citations=citations)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run with: uvicorn chat_api:app --reload
```

---

### 2. The Frontend Deep-Zoom Viewer (React / Next.js)

Standard `<img>` tags or PDF renderers will lag and crash the browser when rendering a 50MB, 600-DPI architectural sheet. By using the `.dzi` format generated in Phase 1, we use **OpenSeadragon**, which works exactly like Google Maps—it only downloads the specific image tiles you are currently zooming in on.

Install frontend dependencies:
```bash
npm install openseadragon react-openseadragon axios
```

Here is a conceptual React component showing the UI split between the chat and the viewer (`BlueprintViewer.tsx`):

```tsx
import React, { useState, useEffect, useRef } from 'react';
import OpenSeadragon from 'openseadragon';
import axios from 'axios';

export default function ConstructionArchiveApp() {
  const [query, setQuery] = useState("");
  const [chatLog, setChatLog] = useState<{role: string, content: string}[]>([]);
  const [activeDziUrl, setActiveDziUrl] = useState<string | null>(null);
  const viewerRef = useRef<OpenSeadragon.Viewer | null>(null);

  // Initialize OpenSeadragon Viewer
  useEffect(() => {
    if (activeDziUrl) {
      if (viewerRef.current) {
        // If viewer exists, just change the image smoothly
        viewerRef.current.open(activeDziUrl);
      } else {
        // Initialize new viewer
        viewerRef.current = OpenSeadragon({
          id: "openseadragon-viewer",
          prefixUrl: "https://cdnjs.cloudflare.com/ajax/libs/openseadragon/3.0.0/images/",
          tileSources: activeDziUrl, // e.g., "http://localhost:8000/tiles/A-101.dzi"
          animationTime: 0.5,
          blendTime: 0.1,
          constrainDuringPan: true,
          maxZoomPixelRatio: 2, // Allows deep zooming into blueprint text
        });
      }
    }
    return () => {
      // Cleanup on unmount
      if (viewerRef.current) viewerRef.current.destroy();
    };
  }, [activeDziUrl]);

  const handleSendMessage = async () => {
    if (!query) return;

    // 1. Add user message to UI
    const newChatLog =[...chatLog, { role: "user", content: query }];
    setChatLog(newChatLog);
    setQuery("");

    try {
      // 2. Call our FastAPI Vision-RAG endpoint
      const response = await axios.post("http://localhost:8000/api/chat", {
        query: query,
        discipline_filter: null // Can be set via a dropdown in UI
      });

      const { answer, citations } = response.data;

      // 3. Update Chat UI with AI Answer
      setChatLog([...newChatLog, { role: "assistant", content: answer }]);

      // 4. Automatically load the most relevant blueprint into the Viewer!
      if (citations.length > 0) {
        // Construct the full URL to the static DZI file hosted on backend/S3
        setActiveDziUrl(`http://localhost:8000${citations[0].dzi_path}`);
      }
    } catch (error) {
      console.error("Chat error:", error);
    }
  };

  return (
    <div className="flex h-screen w-full bg-gray-900 text-white">
      
      {/* LEFT PANEL: Chat Interface */}
      <div className="w-1/3 border-r border-gray-700 flex flex-col">
        <div className="p-4 border-b border-gray-700 text-xl font-bold">
          AI Site Superintendent
        </div>
        
        {/* Chat History */}
        <div className="flex-1 overflow-y-auto p-4 space-y-4">
          {chatLog.map((msg, idx) => (
            <div key={idx} className={`p-3 rounded-lg ${msg.role === 'user' ? 'bg-blue-600 ml-8' : 'bg-gray-800 mr-8'}`}>
              {msg.content}
            </div>
          ))}
        </div>

        {/* Chat Input */}
        <div className="p-4 bg-gray-800 flex gap-2">
          <input 
            type="text" 
            className="flex-1 p-2 rounded bg-gray-700 text-white focus:outline-none"
            placeholder="Ask about details, rooms, or pipes..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleSendMessage()}
          />
          <button onClick={handleSendMessage} className="bg-blue-500 px-4 py-2 rounded font-bold">
            Ask
          </button>
        </div>
      </div>

      {/* RIGHT PANEL: OpenSeadragon Blueprint Viewer */}
      <div className="w-2/3 h-full relative">
        {activeDziUrl ? (
           <div id="openseadragon-viewer" className="w-full h-full bg-black"></div>
        ) : (
           <div className="flex items-center justify-center h-full text-gray-500">
             Ask a question to load the relevant blueprint.
           </div>
        )}
        
        {/* Viewer Overlay UI (Optional) */}
        {activeDziUrl && (
          <div className="absolute top-4 right-4 bg-gray-800 p-2 rounded shadow-lg text-sm">
            Current Sheet: {activeDziUrl.split('/').pop()?.replace('.dzi', '')}
          </div>
        )}
      </div>

    </div>
  );
}
```

### 3. Key Technical Capabilities Introduced in Phase 3

1.  **Contextual Grounding (The `detail: "high"` parameter)**: Standard VLM API calls downsize images, destroying the tiny text on construction documents. By passing `detail: "high"`, OpenAI slices the blueprint into a grid of 512x512 squares, embedding and analyzing each one independently. This allows the AI to accurately read a 1/8" scale note on a massive floor plan.
2.  **Generative Viewer Driving**: The architecture dynamically alters the frontend based on semantic meaning. If the user asks *"Where is the primary chiller located?"*, the backend determines it is on Sheet MEP-302. It returns the answer, but the citations array immediately triggers OpenSeadragon to load `MEP-302.dzi`, visually teleporting the user to the exact document they need.
3.  **Hallucination Prevention System Prompt**: Construction is a high-risk industry. If an AI hallucinates a dimension, a contractor could pour concrete in the wrong place. The system prompt explicitly enforces rules: *"DO NOT hallucinate dimensions. Only state measurements if they are explicitly written."* This forces the LLM to rely on Phase 1's extracted OCR data and the visual pixels rather than guessing. 

### 4. What's Next (Phase 4)?
The core system is now fully functional. You can ingest a PDF, search it semantically, view it instantly in 8K resolution, and chat with it. 

**Phase 4 (Advanced Features)** would involve:
*   **Coordinate Generation (Bounding Boxes)**: Asking GPT-4o to return `[x, y, w, h]` coordinates of the objects it finds, and drawing a red translucent `div` overlay directly on top of the OpenSeadragon viewer to highlight the answer visually.
*   **Version Comparison**: Implementing pixel-differencing algorithms (like structural similarity index) to highlight the exact changes between Revision 1 and Revision 2 in the viewer.