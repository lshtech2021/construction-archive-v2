"""Phase 4 unit tests — revision comparison."""
import numpy as np
import pytest
import cv2

from app.services.diff.revision_compare import RevisionComparator


def _make_white_image(path: str, width: int = 200, height: int = 200):
    img = np.ones((height, width, 3), dtype=np.uint8) * 255
    cv2.imwrite(path, img)


def test_identical_images_high_similarity(tmp_path):
    v1 = str(tmp_path / "v1.jpg")
    v2 = str(tmp_path / "v2.jpg")
    _make_white_image(v1)
    _make_white_image(v2)

    result = RevisionComparator().compare(v1, v2, str(tmp_path / "out"))
    assert result.similarity_score > 0.95
    assert result.change_count == 0


def test_modified_image_detects_changes(tmp_path):
    v1 = str(tmp_path / "v1.jpg")
    v2 = str(tmp_path / "v2.jpg")

    # v1: all white
    img1 = np.ones((200, 200, 3), dtype=np.uint8) * 255
    cv2.imwrite(v1, img1)

    # v2: black rectangle added
    img2 = img1.copy()
    cv2.rectangle(img2, (50, 50), (150, 150), (0, 0, 0), -1)
    cv2.imwrite(v2, img2)

    result = RevisionComparator().compare(v1, v2, str(tmp_path / "out"))
    assert result.change_count > 0
    assert result.diff_dzi_path.endswith(".dzi")


def test_diff_dzi_file_exists(tmp_path):
    import os
    v1 = str(tmp_path / "v1.jpg")
    v2 = str(tmp_path / "v2.jpg")
    _make_white_image(v1)
    _make_white_image(v2)

    result = RevisionComparator().compare(v1, v2, str(tmp_path / "out"))
    assert os.path.exists(result.diff_dzi_path)
