"use client";

import { useEffect, useRef, useCallback } from "react";
import OpenSeadragon from "openseadragon";
import type { ResolvedBoundingBox } from "@/types/api";
import { polygonToOSDRect, fitToRect } from "@/lib/osdHelpers";

const BASE = process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000";

export function useOpenSeadragon(containerId: string) {
  const viewerRef = useRef<OpenSeadragon.Viewer | null>(null);

  useEffect(() => {
    const viewer = OpenSeadragon({
      id: containerId,
      prefixUrl: "https://cdn.jsdelivr.net/npm/openseadragon@4.1/build/openseadragon/images/",
      animationTime: 0.5,
      blendTime: 0.1,
      constrainDuringPan: true,
      maxZoomPixelRatio: 2,
      minZoomLevel: 0.1,
      visibilityRatio: 1,
      zoomPerScroll: 1.2,
    });
    viewerRef.current = viewer;

    return () => {
      viewer.destroy();
      viewerRef.current = null;
    };
  }, [containerId]);

  const loadSheet = useCallback((dziPath: string) => {
    const viewer = viewerRef.current;
    if (!viewer) return;
    // Resolve relative paths served from FastAPI /files/
    const url = dziPath.startsWith("http") ? dziPath : `${BASE}/files/${dziPath}`;
    viewer.open({ type: "image", url } as OpenSeadragon.Options);
  }, []);

  const clearOverlays = useCallback(() => {
    viewerRef.current?.clearOverlays();
  }, []);

  const addHighlights = useCallback(
    (highlights: ResolvedBoundingBox[], aspectRatio = 0.75) => {
      const viewer = viewerRef.current;
      if (!viewer) return;
      viewer.clearOverlays();

      highlights.forEach((h, i) => {
        const rect = polygonToOSDRect(
          h.normalized_polygon as [number, number][],
          aspectRatio
        );
        const el = document.createElement("div");
        el.className =
          "border-4 border-red-500 bg-red-500 bg-opacity-20 rounded cursor-pointer animate-pulse";
        el.title = h.text;
        viewer.addOverlay({ element: el, location: rect });

        if (i === 0) {
          fitToRect(viewer, rect);
        }
      });
    },
    []
  );

  return { viewerRef, loadSheet, clearOverlays, addHighlights };
}
