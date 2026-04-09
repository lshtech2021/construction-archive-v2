import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

from app.services.ingestion.dzi_generator import DZIGenerator


@dataclass
class RevisionDiffResult:
    diff_dzi_path: str
    similarity_score: float
    change_count: int


class RevisionComparator:
    def __init__(self) -> None:
        self._dzi = DZIGenerator()

    def compare(self, path_v1: str, path_v2: str, output_dir: str) -> RevisionDiffResult:
        """Align v1 to v2 using ORB feature matching, then produce an SSIM-based diff DZI.

        Blue = additions (content in v2 not in v1)
        Red  = removals  (content in v1 not in v2)
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        img_v1 = cv2.imread(path_v1)
        img_v2 = cv2.imread(path_v2)

        if img_v1 is None or img_v2 is None:
            raise FileNotFoundError("Could not load one or both input images")

        # Resize v1 to match v2 dimensions if needed
        if img_v1.shape[:2] != img_v2.shape[:2]:
            img_v1 = cv2.resize(img_v1, (img_v2.shape[1], img_v2.shape[0]))

        gray_v1 = cv2.cvtColor(img_v1, cv2.COLOR_BGR2GRAY)
        gray_v2 = cv2.cvtColor(img_v2, cv2.COLOR_BGR2GRAY)

        # ORB feature alignment
        orb = cv2.ORB_create(nfeatures=2000)
        kp1, des1 = orb.detectAndCompute(gray_v1, None)
        kp2, des2 = orb.detectAndCompute(gray_v2, None)

        aligned_v1 = img_v1
        if des1 is not None and des2 is not None and len(kp1) >= 4 and len(kp2) >= 4:
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = matcher.match(des1, des2)
            matches = sorted(matches, key=lambda m: m.distance)
            good = matches[: max(10, len(matches) // 3)]

            if len(good) >= 4:
                src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                if H is not None:
                    h, w = img_v2.shape[:2]
                    aligned_v1 = cv2.warpPerspective(img_v1, H, (w, h))

        # Binary threshold both images
        _, bin_v1 = cv2.threshold(
            cv2.cvtColor(aligned_v1, cv2.COLOR_BGR2GRAY),
            200, 255, cv2.THRESH_BINARY,
        )
        _, bin_v2 = cv2.threshold(gray_v2, 200, 255, cv2.THRESH_BINARY)

        # SSIM on grayscale
        score, diff_map = ssim(
            cv2.cvtColor(aligned_v1, cv2.COLOR_BGR2GRAY),
            gray_v2,
            full=True,
        )
        diff_uint8 = (diff_map * 255).astype(np.uint8)
        _, diff_thresh = cv2.threshold(
            diff_uint8, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
        )

        # Find contours of changed areas
        contours, _ = cv2.findContours(
            diff_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        contours = [c for c in contours if cv2.contourArea(c) >= 10]

        # Build composite diff image
        composite = img_v2.copy()
        # Fade base image to 30% so overlays are visible
        faded = (img_v2 * 0.3).astype(np.uint8)
        np.copyto(composite, faded)

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            mask = np.zeros(gray_v2.shape, dtype=np.uint8)
            cv2.drawContours(mask, [contour], -1, 255, -1)

            added = np.count_nonzero(cv2.bitwise_and(bin_v2, mask))
            removed = np.count_nonzero(cv2.bitwise_and(bin_v1[:gray_v2.shape[0], :gray_v2.shape[1]], mask))

            if added >= removed:
                # Blue = addition
                composite[mask > 0] = [255, 0, 0]
            else:
                # Red = removal
                composite[mask > 0] = [0, 0, 255]
            cv2.rectangle(composite, (x, y), (x + w, y + h), [128, 128, 128], 1)

        # Save composite and convert to DZI
        composite_path = os.path.join(output_dir, "diff_composite.jpg")
        cv2.imwrite(composite_path, composite)

        dzi_path = self._dzi.generate(composite_path, "diff", output_dir)

        return RevisionDiffResult(
            diff_dzi_path=dzi_path,
            similarity_score=float(score),
            change_count=len(contours),
        )
