"use client";

import { useState, useCallback } from "react";
import { sendChat } from "@/lib/api";
import type { ChatResponse, Citation, ResolvedBoundingBox } from "@/types/api";

export interface Message {
  role: "user" | "assistant";
  content: string;
  citations?: Citation[];
  highlights?: ResolvedBoundingBox[];
}

export function useChat(projectId: string) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const sendMessage = useCallback(
    async (
      query: string,
      disciplineFilter?: string,
      onCitation?: (dzi: string, highlights: ResolvedBoundingBox[]) => void
    ) => {
      setLoading(true);
      setError(null);
      setMessages((prev) => [...prev, { role: "user", content: query }]);

      try {
        const history = messages.map((m) => ({ role: m.role, content: m.content }));
        const response: ChatResponse = await sendChat({
          query,
          project_id: projectId,
          discipline_filter: disciplineFilter,
          chat_history: history,
        });

        setMessages((prev) => [
          ...prev,
          {
            role: "assistant",
            content: response.answer,
            citations: response.citations,
            highlights: response.highlights,
          },
        ]);

        // Auto-load first cited sheet
        if (response.citations.length > 0 && onCitation) {
          onCitation(response.citations[0].dzi_path, response.highlights);
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : "Request failed");
      } finally {
        setLoading(false);
      }
    },
    [messages, projectId]
  );

  return { messages, loading, error, sendMessage };
}
