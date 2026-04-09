import axios from "axios";
import type {
  CalloutLink,
  ChatRequest,
  ChatResponse,
  IngestionStatus,
  RevisionDiffResult,
  SearchResult,
} from "@/types/api";

const BASE = process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000";

const http = axios.create({ baseURL: BASE });

export async function searchSheets(
  query: string,
  projectId: string,
  discipline?: string,
  limit = 5
): Promise<SearchResult[]> {
  const params: Record<string, string | number> = { q: query, project_id: projectId, limit };
  if (discipline) params.discipline = discipline;
  const { data } = await http.get("/api/v1/search", { params });
  return data;
}

export async function sendChat(request: ChatRequest): Promise<ChatResponse> {
  const { data } = await http.post("/api/v1/chat", request);
  return data;
}

export async function getSheetLinks(sheetId: string): Promise<CalloutLink[]> {
  const { data } = await http.get(`/api/v1/sheets/${sheetId}/links`);
  return data;
}

export async function uploadPdf(
  projectId: string,
  file: File
): Promise<{ task_id: string }> {
  const form = new FormData();
  form.append("file", file);
  const { data } = await http.post(`/api/v1/projects/${projectId}/ingest`, form);
  return data;
}

export async function pollIngestionStatus(
  projectId: string,
  taskId: string
): Promise<IngestionStatus> {
  const { data } = await http.get(`/api/v1/projects/${projectId}/ingest/${taskId}`);
  return data;
}

export async function startDiff(
  projectId: string,
  sheetIdV1: string,
  sheetIdV2: string
): Promise<{ task_id: string }> {
  const { data } = await http.post("/api/v1/diff", {
    project_id: projectId,
    sheet_id_v1: sheetIdV1,
    sheet_id_v2: sheetIdV2,
  });
  return data;
}

export async function pollDiffStatus(taskId: string): Promise<RevisionDiffResult> {
  const { data } = await http.get(`/api/v1/diff/${taskId}`);
  return data;
}
