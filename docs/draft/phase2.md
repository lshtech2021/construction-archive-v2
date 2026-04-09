


To implement **Phase 2: Semantic Sheet Search Engine**, we will build a hybrid search system using **Qdrant** as our Vector Database. 

Construction searches require both **semantic meaning** (e.g., matching "water pipes" to "plumbing fixtures") and **exact keyword matching** (e.g., matching exact room numbers like "Room 402" or equipment codes like "AHU-1"). To achieve this, we will implement a **Hybrid Search (Dense Vectors + Sparse BM25 Vectors)** with **Metadata Filtering** (to filter by Discipline).

### 1. Prerequisites & Installation

Install the Qdrant client, FastEmbed (for lightweight local BM25 sparse embeddings), and the OpenAI SDK (for dense semantic embeddings).

```bash
pip install qdrant-client fastembed openai pydantic
```

### 2. Python Implementation (`search_engine.py`)

This code takes the extracted data from Phase 1, embeds it, stores it in Qdrant, and provides a powerful query engine.

```python
import os
import uuid
from typing import List, Optional
from openai import OpenAI
from qdrant_client import QdrantClient, models
from fastembed.sparse import SparseTextEmbedding
from pydantic import BaseModel

# We reuse the SheetMetadata from Phase 1
class SheetMetadata(BaseModel):
    sheet_number: str
    sheet_title: str
    revision_date: str
    scale: str

class SearchResult(BaseModel):
    score: float
    sheet_number: str
    sheet_title: str
    discipline: str
    dzi_path: str
    match_reason: str # A snippet of why it matched

class SemanticSearchEngine:
    def __init__(self, collection_name: str = "construction_blueprints"):
        self.collection_name = collection_name
        
        # Initialize OpenAI for Dense Vectors (Semantic meaning)
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Initialize FastEmbed for Sparse Vectors (Exact keyword/BM25 matching)
        self.sparse_embedding_model = SparseTextEmbedding(model_name="prithvida/Splade_PP_en_v1")
        
        # Initialize Qdrant Client (Using local persistent storage for this example)
        # In production, point this to Qdrant Cloud or a Docker container: url="http://localhost:6333"
        self.qdrant = QdrantClient(path="./qdrant_db")
        
        self._setup_collection()

    def _setup_collection(self):
        """Creates a Qdrant collection configured for Hybrid Search."""
        if not self.qdrant.collection_exists(self.collection_name):
            self.qdrant.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "dense": models.VectorParams(
                        size=1536, # OpenAI text-embedding-3-small dimension
                        distance=models.Distance.COSINE,
                    )
                },
                sparse_vectors_config={
                    "sparse": models.SparseVectorParams(
                        modifier=models.Modifier.IDF
                    )
                }
            )
            # Create payload indices for fast database filtering
            self.qdrant.create_payload_index(
                collection_name=self.collection_name,
                field_name="discipline",
                field_schema=models.PayloadSchemaType.KEYWORD
            )

    def _get_dense_embedding(self, text: str) -> List[float]:
        """Generate OpenAI dense embeddings for semantic search."""
        response = self.openai_client.embeddings.create(
            input=text,
            model="text-embedding-3-small"
        )
        return response.data[0].embedding

    def _get_sparse_embedding(self, text: str):
        """Generate SPLADE/BM25 sparse embeddings for exact keyword search."""
        # FastEmbed returns a generator, we take the first item
        sparse_result = list(self.sparse_embedding_model.embed([text]))[0]
        return models.SparseVector(
            indices=sparse_result.indices.tolist(),
            values=sparse_result.values.tolist()
        )

    def index_sheet(self, sheet_metadata: SheetMetadata, discipline: str, dzi_path: str, ocr_text: str, visual_summary: str):
        """
        Indexes a blueprint sheet into the Vector DB.
        In a Vision-RAG setup, we combine OCR text and a VLM-generated visual summary.
        """
        print(f"Indexing {sheet_metadata.sheet_number} into Vector DB...")
        
        # 1. Create a rich text representation for the embedding
        searchable_text = f"""
        Sheet Number: {sheet_metadata.sheet_number}
        Title: {sheet_metadata.sheet_title}
        Discipline: {discipline}
        Visual Summary: {visual_summary}
        OCR Text / Notes: {ocr_text}
        """

        # 2. Generate Dense & Sparse Vectors
        dense_vec = self._get_dense_embedding(searchable_text)
        sparse_vec = self._get_sparse_embedding(searchable_text)

        # 3. Define the Payload (Metadata to be stored alongside the vector)
        payload = {
            "sheet_number": sheet_metadata.sheet_number,
            "sheet_title": sheet_metadata.sheet_title,
            "discipline": discipline,
            "dzi_path": dzi_path,
            "searchable_text": searchable_text  # Stored to show context on retrieval
        }

        # 4. Upsert into Qdrant
        self.qdrant.upsert(
            collection_name=self.collection_name,
            points=[
                models.PointStruct(
                    id=str(uuid.uuid5(uuid.NAMESPACE_DNS, sheet_metadata.sheet_number)), # Deterministic ID based on sheet number
                    vector={
                        "dense": dense_vec,
                        "sparse": sparse_vec
                    },
                    payload=payload
                )
            ]
        )

    def search(self, query: str, limit: int = 5, discipline_filter: Optional[str] = None) -> List[SearchResult]:
        """
        Performs a Hybrid Search with optional metadata filtering.
        """
        print(f"\nSearching for: '{query}'" + (f" in [{discipline_filter}]" if discipline_filter else ""))
        
        # 1. Embed the user query
        query_dense = self._get_dense_embedding(query)
        query_sparse = self._get_sparse_embedding(query)

        # 2. Build the Filter (Optional)
        query_filter = None
        if discipline_filter:
            query_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="discipline",
                        match=models.MatchValue(value=discipline_filter)
                    )
                ]
            )

        # 3. Execute Hybrid Query in Qdrant using Reciprocal Rank Fusion (RRF)
        # Qdrant's query_points automatically fuses dense and sparse scores
        results = self.qdrant.query_points(
            collection_name=self.collection_name,
            prefetch=[
                models.Prefetch(
                    query=query_dense,
                    using="dense",
                    limit=limit * 2,
                ),
                models.Prefetch(
                    query=query_sparse,
                    using="sparse",
                    limit=limit * 2,
                )
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            query_filter=query_filter,
            limit=limit,
            with_payload=True
        )

        # 4. Format Output
        search_results =[]
        for point in results.points:
            search_results.append(SearchResult(
                score=point.score,
                sheet_number=point.payload["sheet_number"],
                sheet_title=point.payload["sheet_title"],
                discipline=point.payload["discipline"],
                dzi_path=point.payload["dzi_path"],
                # Just grabbing the first 150 chars of the text as a snippet preview
                match_reason=point.payload["searchable_text"].replace('\n', ' ')[:150] + "..."
            ))
            
        return search_results

# --- USAGE EXAMPLE ---
if __name__ == "__main__":
    search_engine = SemanticSearchEngine()

    # --- SIMULATE INDEXING DATA FROM PHASE 1 ---
    # In production, this data comes directly from the output of the ConstructionIngestionPipeline
    mock_metadata_1 = SheetMetadata(sheet_number="S-201", sheet_title="Second Floor Framing Plan", revision_date="2023-10-01", scale="1/8")
    mock_ocr_1 = "W12x50 steel beams, load-bearing columns on grid lines A and C. Concrete slab 6 inches."
    mock_visual_1 = "Drawing shows a structural framing layout. Blue grid lines indicate column locations."

    mock_metadata_2 = SheetMetadata(sheet_number="A-101", sheet_title="First Floor Plan", revision_date="2023-10-01", scale="1/4")
    mock_ocr_2 = "Main lobby, egress stairs, Room 102 Conference. All interior partitions are standard drywall."
    mock_visual_2 = "Architectural floor plan showing room layouts, door swings, and egress routes."

    search_engine.index_sheet(mock_metadata_1, discipline="Structural", dzi_path="/tiles/S-201.dzi", ocr_text=mock_ocr_1, visual_summary=mock_visual_1)
    search_engine.index_sheet(mock_metadata_2, discipline="Architectural", dzi_path="/tiles/A-101.dzi", ocr_text=mock_ocr_2, visual_summary=mock_visual_2)

    # --- SIMULATE SEARCHING ---
    print("-" * 40)
    
    # Example 1: Semantic Search (No exact keyword matches, relies on Dense Vector meaning)
    res_1 = search_engine.search("Where are the main load bearing elements?", limit=1)
    for r in res_1:
        print(f"[{r.score:.3f}] {r.sheet_number} ({r.discipline}): {r.sheet_title}")

    # Example 2: Exact Match Search (Relies on Sparse/BM25 Vector)
    res_2 = search_engine.search("Room 102", limit=1)
    for r in res_2:
        print(f"[{r.score:.3f}] {r.sheet_number} ({r.discipline}): {r.sheet_title}")

    # Example 3: Filtered Search (Forces DB to only look at Structural drawings)
    res_3 = search_engine.search("stairs", discipline_filter="Structural", limit=1)
    if not res_3:
        print("No structural drawings found containing 'stairs'.")
```

### 3. Key Technical Capabilities Introduced

1.  **Hybrid Search (Dense + Sparse):** 
    *   **Dense Vectors (`text-embedding-3-small`)** capture the *meaning* of the text. If a user searches for "HVAC," it knows to pull up a sheet containing the text "Air Handling Unit."
    *   **Sparse Vectors (`SPLADE/BM25`)** maintain a sparse matrix of exact words. This prevents the LLM from trying to guess what "Room 402" means semantically, ensuring it matches the exact alphanumeric string.
2.  **Reciprocal Rank Fusion (RRF):** The code uses Qdrant's built-in `FusionQuery`. It executes both the dense and sparse searches in parallel, normalizes their scores, and optimally merges the results.
3.  **Payload Filtering (`discipline_filter`):** Vector searches become inaccurate if they search across too much data. By indexing the `discipline` as a Keyword Payload, we can pass a hard filter. If a user asks a plumbing question, Qdrant skips the Architectural and Structural embeddings entirely, cutting compute time and hallucination risk drastically.
4.  **VLM Visual Summarization Injection:** Notice the `visual_summary` variable. To make a PDF *image* semantically searchable via text, we use GPT-4o (from Phase 1) to generate a paragraph describing what the image looks like visually, then embed that text.

### 4. What's Next (Phase 3)?
Now that the documents are split, tiled into DZI formats, and stored in a searchable Vector database, the next step is **Phase 3: Generative AI Q&A Chat & Viewer**. 

In Phase 3, we will build the RAG (Retrieval-Augmented Generation) orchestrator. It will take the user's chat prompt, call `SemanticSearchEngine.search()`, retrieve the matching DZI images and OCR text, pass them back into GPT-4o Vision, and stream a conversational answer back to the user interface.