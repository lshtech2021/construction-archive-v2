// Mirror of backend Pydantic schemas

export interface SearchResult {
  score: number;
  sheet_id: string;
  sheet_number: string;
  sheet_title: string;
  discipline: string;
  dzi_path: string;
  image_path: string;
}

export interface ResolvedBoundingBox {
  text: string;
  sheet_id: string;
  normalized_polygon: [number, number][]; // [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
}

export interface Citation {
  sheet_number: string;
  sheet_title: string;
  dzi_path: string;
}

export interface ChatRequest {
  query: string;
  project_id: string;
  discipline_filter?: string;
  chat_history?: { role: string; content: string }[];
}

export interface ChatResponse {
  answer: string;
  citations: Citation[];
  highlights: ResolvedBoundingBox[];
}

export interface CalloutLink {
  target_sheet: string;
  target_detail_number: string | null;
  bounding_box: [number | null, number | null, number | null, number | null];
}

export interface IngestionStatus {
  status: "pending" | "complete" | "failed" | string;
  progress_pct?: number;
  sheet_count?: number;
  error?: string;
}

export interface RevisionDiffResult {
  status: "complete" | "failed" | string;
  diff_dzi_path?: string;
  similarity_score?: number;
  change_count?: number;
}
