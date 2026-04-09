"use client";

import { useState } from "react";
import { use } from "react";
import { ChatPanel } from "@/components/chat/ChatPanel";
import { BlueprintViewer } from "@/components/viewer/BlueprintViewer";
import { PdfDropzone } from "@/components/upload/PdfDropzone";
import { useSheetLinks } from "@/hooks/useSheetLinks";
import type { ResolvedBoundingBox } from "@/types/api";

interface PageProps {
  params: Promise<{ projectId: string }>;
}

export default function ProjectWorkspace({ params }: PageProps) {
  const { projectId } = use(params);
  const [activeDziUrl, setActiveDziUrl] = useState<string | null>(null);
  const [activeHighlights, setActiveHighlights] = useState<ResolvedBoundingBox[]>([]);
  const [activeSheetId, setActiveSheetId] = useState<string | null>(null);
  const [showUpload, setShowUpload] = useState(false);

  const { links: calloutLinks } = useSheetLinks(activeSheetId);

  const handleCitation = (dziUrl: string, highlights: ResolvedBoundingBox[]) => {
    setActiveDziUrl(dziUrl);
    setActiveHighlights(highlights);
    if (highlights.length > 0) {
      setActiveSheetId(highlights[0].sheet_id);
    }
  };

  const handleSheetSelect = (targetSheet: string) => {
    // In production, look up DZI URL for targetSheet from the sheet registry
    // For now, signal intent and clear highlights
    setActiveSheetId(null);
    setActiveHighlights([]);
    // targetSheet lookup would happen here
    console.info("Navigate to sheet:", targetSheet);
  };

  return (
    <div className="flex h-screen overflow-hidden">
      {/* Left panel: Chat (1/3) */}
      <div className="w-1/3 flex flex-col border-r border-gray-700">
        <div className="flex items-center justify-between px-4 py-3 border-b border-gray-700">
          <h2 className="text-sm font-semibold text-gray-200">
            Project: {projectId}
          </h2>
          <button
            onClick={() => setShowUpload((v) => !v)}
            className="text-xs text-blue-400 hover:text-blue-300"
          >
            {showUpload ? "Hide upload" : "Upload PDF"}
          </button>
        </div>

        {showUpload && (
          <div className="p-4 border-b border-gray-700">
            <PdfDropzone
              projectId={projectId}
              onComplete={() => setShowUpload(false)}
            />
          </div>
        )}

        <div className="flex-1 overflow-hidden">
          <ChatPanel projectId={projectId} onCitation={handleCitation} />
        </div>
      </div>

      {/* Right panel: Blueprint viewer (2/3) */}
      <div className="flex-1 p-2">
        {activeDziUrl ? (
          <BlueprintViewer
            dziUrl={activeDziUrl}
            highlights={activeHighlights}
            calloutLinks={calloutLinks}
            onSheetSelect={handleSheetSelect}
          />
        ) : (
          <div className="flex h-full items-center justify-center text-gray-500 text-sm">
            Ask a question to load a blueprint
          </div>
        )}
      </div>
    </div>
  );
}
