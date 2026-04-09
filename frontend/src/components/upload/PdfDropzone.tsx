"use client";

import { useState, useRef } from "react";
import { uploadPdf, pollIngestionStatus } from "@/lib/api";

interface PdfDropzoneProps {
  projectId: string;
  onComplete?: (sheetCount: number) => void;
}

export function PdfDropzone({ projectId, onComplete }: PdfDropzoneProps) {
  const [status, setStatus] = useState<string>("");
  const [uploading, setUploading] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFile = async (file: File) => {
    if (!file.name.endsWith(".pdf")) {
      setStatus("Please upload a PDF file.");
      return;
    }
    setUploading(true);
    setStatus("Uploading...");

    try {
      const { task_id } = await uploadPdf(projectId, file);
      setStatus("Processing... (this may take a few minutes)");

      // Poll every 5 seconds
      const poll = setInterval(async () => {
        const result = await pollIngestionStatus(projectId, task_id);
        if (result.status === "complete") {
          clearInterval(poll);
          setStatus(`Done! Ingested ${result.sheet_count ?? 0} sheets.`);
          setUploading(false);
          onComplete?.(result.sheet_count ?? 0);
        } else if (result.status === "failed") {
          clearInterval(poll);
          setStatus(`Error: ${result.error}`);
          setUploading(false);
        }
      }, 5000);
    } catch (err) {
      setStatus(`Upload failed: ${err instanceof Error ? err.message : String(err)}`);
      setUploading(false);
    }
  };

  return (
    <div
      className="border-2 border-dashed border-gray-600 rounded-lg p-8 text-center cursor-pointer hover:border-blue-500 transition-colors"
      onClick={() => inputRef.current?.click()}
      onDragOver={(e) => e.preventDefault()}
      onDrop={(e) => {
        e.preventDefault();
        const file = e.dataTransfer.files[0];
        if (file) handleFile(file);
      }}
    >
      <input
        ref={inputRef}
        type="file"
        accept=".pdf"
        className="hidden"
        onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])}
      />
      <p className="text-gray-400 text-sm">
        {uploading ? status : "Drop a PDF here or click to browse"}
      </p>
      {status && !uploading && (
        <p className="mt-2 text-xs text-green-400">{status}</p>
      )}
    </div>
  );
}
