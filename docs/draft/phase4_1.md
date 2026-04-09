
**Phase 4** transforms the archive from a "smart search engine" into an **Interactive Spatial Workspace**. In construction, finding the document is only half the battle; users also need to spot microscopic revisions, locate specific components visually, and navigate seamlessly between hundreds of detailed views.

Here is the implementation plan for Phase 4, focusing on **Spatial Bounding Boxes**, **Automated Revision Comparison**, and **Auto-Hyperlinking Callouts**.

---

### Feature 1: Spatial Vision-RAG (Bounding Box Overlays)

Instead of just telling the user *"The chiller is on sheet MEP-302"*, the AI will draw a red box exactly around the chiller on the blueprint. 

**Backend Logic (Updating GPT-4o Prompt):**
GPT-4o natively supports spatial localization. By prompting it correctly, it can return normalized coordinates (0 to 1000) for objects it identifies in the image.

1.  **Update Prompt**: Add this to the system prompt in FastAPI:
    > *"When identifying an object, also provide its bounding box coordinates. Use the format `[ymin, xmin, ymax, xmax]` normalized to a 1000x1000 grid where [0,0] is top-left and[1000,1000] is bottom-right. Return this in a JSON block."*

2.  **Frontend Implementation (OpenSeadragon Overlays)**:
    When the frontend receives these coordinates, it converts them into OpenSeadragon (OSD) Viewport coordinates and draws an interactive `<div>`.

```tsx
// Frontend: React + OpenSeadragon Overlay Logic
function drawBoundingBox(viewer: OpenSeadragon.Viewer, gptCoords: number[], label: string) {
    const [ymin, xmin, ymax, xmax] = gptCoords;

    // Convert 1000x1000 normalized coordinates to OpenSeadragon's viewport coordinates (0 to 1)
    const osdX = xmin / 1000;
    const osdY = ymin / 1000;
    const osdWidth = (xmax - xmin) / 1000;
    const osdHeight = (ymax - ymin) / 1000;

    // Create DOM element for the highlight
    const highlightElement = document.createElement("div");
    highlightElement.className = "border-4 border-red-500 bg-red-500/20 cursor-pointer group";
    
    // Add tooltip
    const tooltip = document.createElement("span");
    tooltip.className = "hidden group-hover:block absolute -top-8 left-0 bg-black text-white p-1 text-xs rounded";
    tooltip.innerText = label;
    highlightElement.appendChild(tooltip);

    // Add to OpenSeadragon
    viewer.addOverlay({
        element: highlightElement,
        location: new OpenSeadragon.Rect(osdX, osdY, osdWidth, osdHeight)
    });

    // Animate camera to zoom into the specific bounding box
    viewer.viewport.fitBounds(new OpenSeadragon.Rect(osdX, osdY, osdWidth, osdHeight), false);
}
```

---

### Feature 2: Automated Revision Comparison (Pixel Differencing)

Construction updates (Addendums/Bulletins) are notoriously difficult to track. Even if an architect clouds a revision, things are often missed. We will use **OpenCV (Computer Vision)** to compare an old sheet (v1) and a new sheet (v2), highlighting additions in **Blue** and deletions in **Red**.

Because scanned/exported PDFs might shift by a few pixels, we first **align (register)** the images before comparing them.

**Python Backend Implementation (`revision_compare.py`):**
```python
import cv2
import numpy as np

def align_and_compare_blueprints(old_img_path: str, new_img_path: str, output_path: str):
    """Aligns two blueprint revisions and generates a Red/Blue difference map."""
    
    # 1. Load images in grayscale
    img_old = cv2.imread(old_img_path, cv2.IMREAD_GRAYSCALE)
    img_new = cv2.imread(new_img_path, cv2.IMREAD_GRAYSCALE)

    # 2. Image Registration (Alignment) using ORB feature matching
    # This fixes slight shifts or rotations between v1 and v2 PDF exports
    orb = cv2.ORB_create(5000)
    keypoints1, descriptors1 = orb.detectAndCompute(img_old, None)
    keypoints2, descriptors2 = orb.detectAndCompute(img_new, None)

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)
    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # Find Homography (Transformation Matrix)
    h_matrix, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
    height, width = img_new.shape
    
    # Warp old image to perfectly align with new image
    img_old_aligned = cv2.warpPerspective(img_old, h_matrix, (width, height))

    # 3. Compute Pixel Differences
    # Threshold images to pure black/white to remove anti-aliasing noise
    _, old_thresh = cv2.threshold(img_old_aligned, 200, 255, cv2.THRESH_BINARY)
    _, new_thresh = cv2.threshold(img_new, 200, 255, cv2.THRESH_BINARY)

    # Compare:
    # Removed items: Black in old, White in new
    removed_mask = cv2.bitwise_and(cv2.bitwise_not(old_thresh), new_thresh)
    # Added items: White in old, Black in new
    added_mask = cv2.bitwise_and(old_thresh, cv2.bitwise_not(new_thresh))

    # 4. Generate Composite Overlay
    # Start with the new blueprint as a faded background
    composite = cv2.cvtColor(new_thresh, cv2.COLOR_GRAY2BGR)
    composite = cv2.addWeighted(composite, 0.3, np.full_like(composite, 255), 0.7, 0) # Fade

    # Highlight additions in BLUE (BGR format)
    composite[added_mask > 0] = [255, 0, 0] 
    # Highlight removals in RED
    composite[removed_mask > 0] =[0, 0, 255] 

    # Save to disk (This will then be converted to DZI for the frontend)
    cv2.imwrite(output_path, composite)
    print(f"Comparison saved to {output_path}")

# Example Usage:
# align_and_compare_blueprints("A101_v1.jpg", "A101_v2.jpg", "A101_diff.jpg")
```

---

### Feature 3: Auto-Hyperlinking of Detail Callouts (Graph Linking)

Blueprints are full of symbols that say `"See Detail 4 on Sheet S-200"`. Navigating these manually is tedious. We can automate this during the Ingestion Phase (Phase 1) using OCR data.

**How it works:**
1.  **Extract Bounding Boxes**: When Azure Document Intelligence runs during ingestion, it outputs every word and its bounding box `[x, y, width, height]`.
2.  **Regex Matching**: The backend scans the OCR text for standard callout patterns. 
    *   Regex Example: `r'\b(\d{1,2})\s*/\s*([A-Z]+-?\d{1,3})\b'` (Matches "4 / S-200" or "12/A101").
3.  **Database Mapping**: The system checks if the target sheet (e.g., `S-200`) exists in our Postgres database. 
4.  **Frontend Clickable Zones**: If a match is found, the backend saves a "Link Node" associated with that sheet. When the user opens the sheet, OpenSeadragon renders an invisible clickable `<button>` over that text. 

**Database Schema Update (PostgreSQL):**
```sql
CREATE TABLE CalloutLinks (
    id UUID PRIMARY KEY,
    source_sheet_id UUID REFERENCES Sheets(id),
    target_sheet_number VARCHAR(50), 
    target_detail_number VARCHAR(10), -- e.g., "4"
    bbox_x FLOAT,
    bbox_y FLOAT,
    bbox_w FLOAT,
    bbox_h FLOAT
);
```

**Frontend User Experience:**
*   A user is looking at a Floor Plan.
*   They see a section cut symbol labeled **`3 / A-401`**. 
*   The symbol is highlighted with a faint blue glow.
*   The user clicks it. The OpenSeadragon viewer instantly unloads the current sheet, loads the DZI for `A-401`, and directly zooms the camera into Detail `3`.

---

### Summary of the Final System

By completing Phase 4, the archive operates at the bleeding edge of ConTech (Construction Technology):
1.  **Phase 1 (Ingestion)** accurately dissects 500-page 4K resolution PDFs into manageable, metadata-rich, deep-zoom tiles.
2.  **Phase 2 (Retrieval)** enables visual search ("show me all electrical panel schedules").
3.  **Phase 3 (Vision-Chat)** gives PMs and field workers an AI superintendent that instantly answers questions by looking at the drawings.
4.  **Phase 4 (Advanced Spatial)** turns those static drawings into an interactive, version-controlled web of navigable data, reducing RFI (Request For Information) delays by instantly highlighting changes and answers visually.