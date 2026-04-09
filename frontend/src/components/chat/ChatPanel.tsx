"use client";

import { useState, useRef, useEffect } from "react";
import type { ResolvedBoundingBox } from "@/types/api";
import { useChat } from "@/hooks/useChat";

interface ChatPanelProps {
  projectId: string;
  onCitation: (dziUrl: string, highlights: ResolvedBoundingBox[]) => void;
}

export function ChatPanel({ projectId, onCitation }: ChatPanelProps) {
  const [input, setInput] = useState("");
  const [discipline, setDiscipline] = useState("");
  const { messages, loading, error, sendMessage } = useChat(projectId);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleSend = async () => {
    if (!input.trim() || loading) return;
    const query = input.trim();
    setInput("");
    await sendMessage(
      query,
      discipline || undefined,
      (dziUrl, highlights) => onCitation(dziUrl, highlights)
    );
  };

  return (
    <div className="flex flex-col h-full bg-gray-800 text-white">
      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-3">
        {messages.length === 0 && (
          <p className="text-gray-400 text-sm">Ask a question about the blueprints...</p>
        )}
        {messages.map((msg, i) => (
          <div key={i} className={msg.role === "user" ? "text-right" : "text-left"}>
            <div
              className={`inline-block max-w-full p-3 rounded-lg text-sm ${
                msg.role === "user"
                  ? "bg-blue-600 text-white"
                  : "bg-gray-700 text-gray-100"
              }`}
            >
              <p className="whitespace-pre-wrap">{msg.content}</p>

              {msg.citations && msg.citations.length > 0 && (
                <div className="mt-2 flex flex-wrap gap-1">
                  {msg.citations.map((c, j) => (
                    <button
                      key={j}
                      onClick={() => onCitation(c.dzi_path, msg.highlights || [])}
                      className="text-xs bg-blue-800 hover:bg-blue-700 text-blue-200 px-2 py-1 rounded"
                    >
                      {c.sheet_number}: {c.sheet_title}
                    </button>
                  ))}
                </div>
              )}
            </div>
          </div>
        ))}

        {loading && (
          <div className="text-left">
            <div className="inline-block bg-gray-700 p-3 rounded-lg text-sm text-gray-400 animate-pulse">
              Analyzing blueprints...
            </div>
          </div>
        )}

        {error && (
          <div className="text-xs text-red-400 text-center">{error}</div>
        )}
        <div ref={bottomRef} />
      </div>

      {/* Discipline filter */}
      <div className="px-4 pt-2">
        <select
          value={discipline}
          onChange={(e) => setDiscipline(e.target.value)}
          className="w-full text-xs bg-gray-700 text-gray-300 border border-gray-600 rounded px-2 py-1"
        >
          <option value="">All disciplines</option>
          <option value="Architectural">Architectural</option>
          <option value="Structural">Structural</option>
          <option value="Mechanical">Mechanical</option>
          <option value="Electrical">Electrical</option>
          <option value="Plumbing">Plumbing</option>
          <option value="Fire Protection">Fire Protection</option>
          <option value="Civil">Civil</option>
        </select>
      </div>

      {/* Input */}
      <div className="flex gap-2 p-4">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && handleSend()}
          placeholder="Ask about blueprints..."
          className="flex-1 bg-gray-700 text-white text-sm border border-gray-600 rounded px-3 py-2 focus:outline-none focus:border-blue-500"
        />
        <button
          onClick={handleSend}
          disabled={loading || !input.trim()}
          className="bg-blue-600 hover:bg-blue-500 disabled:bg-gray-600 text-white text-sm px-4 py-2 rounded"
        >
          Send
        </button>
      </div>
    </div>
  );
}
