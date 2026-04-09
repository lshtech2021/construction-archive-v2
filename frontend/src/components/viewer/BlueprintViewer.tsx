"use client";

import { useEffect, useId } from "react";
import type { ResolvedBoundingBox, CalloutLink } from "@/types/api";
import { useOpenSeadragon } from "@/hooks/useOpenSeadragon";
import OpenSeadragon from "openseadragon";
import { polygonToOSDRect } from "@/lib/osdHelpers";

interface BlueprintViewerProps {
  dziUrl: string | null;
  highlights?: ResolvedBoundingBox[];
  calloutLinks?: CalloutLink[];
  onSheetSelect?: (targetSheet: string) => void;
  aspectRatio?: number;
}

export function BlueprintViewer({
  dziUrl,
  highlights = [],
  calloutLinks = [],
  onSheetSelect,
  aspectRatio = 0.75,
}: BlueprintViewerProps) {
  const containerId = useId().replace(/:/g, "_");
  const { viewerRef, loadSheet, addHighlights } = useOpenSeadragon(containerId);

  // Load sheet when dziUrl changes
  useEffect(() => {
    if (dziUrl) loadSheet(dziUrl);
  }, [dziUrl, loadSheet]);

  // Render highlights
  useEffect(() => {
    if (highlights.length > 0) {
      addHighlights(highlights, aspectRatio);
    }
  }, [highlights, addHighlights, aspectRatio]);

  // Render callout hyperlinks
  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewer || calloutLinks.length === 0) return;

    calloutLinks.forEach((link) => {
      const [bx, by, bw, bh] = link.bounding_box;
      if (bx == null || by == null || bw == null || bh == null) return;

      const rect = new OpenSeadragon.Rect(bx, by * aspectRatio, bw, bh * aspectRatio);
      const el = document.createElement("div");
      el.className =
        "cursor-pointer border border-blue-400 border-opacity-0 hover:border-opacity-100 hover:bg-blue-400 hover:bg-opacity-20 transition-all duration-200";
      el.title = `Go to ${link.target_sheet}`;

      const tracker = new OpenSeadragon.MouseTracker({
        element: el,
        clickHandler: () => {
          onSheetSelect?.(link.target_sheet);
        },
        // Prevent OSD pan on mousedown over the overlay
        pressHandler: (event) => {
          (event as MouseEvent).stopPropagation?.();
        },
      });
      tracker.setTracking(true);

      viewer.addOverlay({ element: el, location: rect });
    });
  }, [calloutLinks, viewerRef, onSheetSelect, aspectRatio]);

  return (
    <div
      id={containerId}
      className="w-full h-full bg-gray-900 rounded-lg"
      style={{ minHeight: 400 }}
    />
  );
}
