import OpenSeadragon from "openseadragon";

/**
 * Convert a normalized polygon [[x1,y1],...] to an OSD Rect.
 * OSD coordinate space: width = 1.0, height = aspect-ratio based.
 * For a horizontally-oriented blueprint, aspect ratio < 1 is typical.
 */
export function polygonToOSDRect(
  polygon: [number, number][],
  aspectRatio: number // imageHeight / imageWidth
): OpenSeadragon.Rect {
  const xs = polygon.map((p) => p[0]);
  const ys = polygon.map((p) => p[1]);
  const x = Math.min(...xs);
  const y = Math.min(...ys);
  const w = Math.max(...xs) - x;
  const h = Math.max(...ys) - y;
  // In OSD: x and w are in image-width units (1.0 = full width)
  // y and h must be scaled by aspectRatio
  return new OpenSeadragon.Rect(x, y * aspectRatio, w, h * aspectRatio);
}

/**
 * Animate the OSD viewport to fit a Rect with optional padding.
 */
export function fitToRect(
  viewer: OpenSeadragon.Viewer,
  rect: OpenSeadragon.Rect,
  padding = 0.05
): void {
  const padded = rect.getIntegerBoundingBox
    ? rect
    : new OpenSeadragon.Rect(
        rect.x - padding,
        rect.y - padding,
        rect.width + padding * 2,
        rect.height + padding * 2
      );
  viewer.viewport.fitBoundsWithConstraints(padded, false);
}
