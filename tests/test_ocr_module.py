"""
Unit tests for the OCR Preprocessing and Detection module.
"""

import unittest
import numpy as np
import cv2
from ocr_module import ImagePreprocessor, OCREngine


class TestOCRModule(unittest.TestCase):

    def setUp(self):
        # Create a synthetic high-contrast document image (white background, black text)
        self.image = np.ones((300, 500, 3), dtype=np.uint8) * 255
        # Draw simulated newspaper text lines
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(self.image, "PARLIAMENT PASSES NEW DEFENCE REFORM BILL", (30, 60), font, 0.7, (0, 0, 0), 2)
        cv2.putText(self.image, "New Delhi: The Union Government today tabled", (30, 110), font, 0.5, (0, 0, 0), 1)
        cv2.putText(self.image, "the landmark modernisation policy for armed forces.", (30, 150), font, 0.5, (0, 0, 0), 1)
        cv2.putText(self.image, "The procurement budget sees a 15% increase.", (30, 190), font, 0.5, (0, 0, 0), 1)

    def test_to_grayscale(self):
        gray = ImagePreprocessor.to_grayscale(self.image)
        self.assertEqual(len(gray.shape), 2)
        self.assertEqual(gray.shape, (300, 500))

    def test_enhance_contrast(self):
        gray = ImagePreprocessor.to_grayscale(self.image)
        enhanced = ImagePreprocessor.enhance_contrast(gray)
        self.assertEqual(enhanced.shape, gray.shape)
        self.assertEqual(enhanced.dtype, np.uint8)

    def test_blur_detection(self):
        gray = ImagePreprocessor.to_grayscale(self.image)
        is_blurry_sharp, sharp_score = ImagePreprocessor.check_blur(gray, threshold=80.0)

        # Create heavily blurred version
        blurred = cv2.GaussianBlur(gray, (25, 25), 0)
        is_blurry_soft, soft_score = ImagePreprocessor.check_blur(blurred, threshold=80.0)

        # Sharp image should have significantly higher Laplacian variance than blurred
        self.assertGreater(sharp_score, soft_score)
        self.assertTrue(is_blurry_soft)

    def test_deskew_angle(self):
        # Rotate our synthetic image by 5 degrees
        (h, w) = self.image.shape[:2]
        center = (w // 2, h // 2)
        rot_mat = cv2.getRotationMatrix2D(center, 5.0, 1.0)
        tilted = cv2.warpAffine(self.image, rot_mat, (w, h), borderValue=(255, 255, 255))

        prep_result = ImagePreprocessor.preprocess_pipeline(tilted)
        self.assertIn("preprocessed_gray", prep_result)
        self.assertIn("deskew_angle", prep_result)
        self.assertIn("blur_score", prep_result)

    def test_preprocess_pipeline_end_to_end(self):
        result = ImagePreprocessor.preprocess_pipeline(self.image)
        self.assertFalse(result["is_blurry"])
        self.assertEqual(result["preprocessed_gray"].shape, (300, 500))


if __name__ == "__main__":
    unittest.main()
