"use client";

import { useState, use } from "react";
import { BlueprintViewer } from "@/components/viewer/BlueprintViewer";
import { startDiff, pollDiffStatus } from "@/lib/api";

interface PageProps {
  params: Promise<{ projectId: string }>;
}

export default function DiffPage({ params }: PageProps) {
  const { projectId } = use(params);
  const [sheetIdV1, setSheetIdV1] = useState("");
  const [sheetIdV2, setSheetIdV2] = useState("");
  const [diffDziUrl, setDiffDziUrl] = useState<string | null>(null);
  const [similarityScore, setSimilarityScore] = useState<number | null>(null);
  const [changeCount, setChangeCount] = useState<number | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleCompare = async () => {
    if (!sheetIdV1 || !sheetIdV2) return;
    setLoading(true);
    setError("");
    setDiffDziUrl(null);

    try {
      const { task_id } = await startDiff(projectId, sheetIdV1, sheetIdV2);

      const poll = setInterval(async () => {
        const result = await pollDiffStatus(task_id);
        if (result.status === "complete") {
          clearInterval(poll);
          setDiffDziUrl(result.diff_dzi_path ?? null);
          setSimilarityScore(result.similarity_score ?? null);
          setChangeCount(result.change_count ?? null);
          setLoading(false);
        } else if (result.status === "failed") {
          clearInterval(poll);
          setError(result.error ?? "Comparison failed");
          setLoading(false);
        }
      }, 3000);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Error starting comparison");
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-screen p-6 space-y-4">
      <h1 className="text-xl font-bold">Revision Comparison</h1>

      <div className="flex gap-4 items-end">
        <div className="flex-1">
          <label className="text-xs text-gray-400">Sheet ID (v1)</label>
          <input
            value={sheetIdV1}
            onChange={(e) => setSheetIdV1(e.target.value)}
            className="w-full mt-1 bg-gray-700 text-white border border-gray-600 rounded px-3 py-2 text-sm"
            placeholder="UUID of old sheet"
          />
        </div>
        <div className="flex-1">
          <label className="text-xs text-gray-400">Sheet ID (v2)</label>
          <input
            value={sheetIdV2}
            onChange={(e) => setSheetIdV2(e.target.value)}
            className="w-full mt-1 bg-gray-700 text-white border border-gray-600 rounded px-3 py-2 text-sm"
            placeholder="UUID of new sheet"
          />
        </div>
        <button
          onClick={handleCompare}
          disabled={loading || !sheetIdV1 || !sheetIdV2}
          className="bg-blue-600 hover:bg-blue-500 disabled:bg-gray-600 text-white px-6 py-2 rounded text-sm"
        >
          {loading ? "Comparing..." : "Compare"}
        </button>
      </div>

      {error && <p className="text-red-400 text-sm">{error}</p>}

      {similarityScore !== null && (
        <div className="flex gap-4 text-sm text-gray-300">
          <span>Similarity: {(similarityScore * 100).toFixed(1)}%</span>
          <span>Changes detected: {changeCount}</span>
          <span className="flex items-center gap-1">
            <span className="w-3 h-3 bg-blue-500 inline-block" /> Additions
          </span>
          <span className="flex items-center gap-1">
            <span className="w-3 h-3 bg-red-500 inline-block" /> Removals
          </span>
        </div>
      )}

      <div className="flex-1">
        {diffDziUrl ? (
          <BlueprintViewer dziUrl={diffDziUrl} />
        ) : (
          <div className="flex h-full items-center justify-center text-gray-500 text-sm border border-dashed border-gray-700 rounded-lg">
            Diff result will appear here
          </div>
        )}
      </div>
    </div>
  );
}
