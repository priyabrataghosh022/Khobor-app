"""
Khobor - OCR Extraction & Image Preprocessing Module
====================================================
Designed for Google Colab and local Python environments.

This module processes photographed newspaper clippings for CDS & GATE exam prep:
1. Preprocesses raw photos using OpenCV (grayscale, CLAHE contrast enhancement, deskewing).
2. Detects blurriness using the Laplacian variance method to warn users if the image is too soft.
3. Performs local, free OCR using Tesseract (pytesseract).
4. Includes an optional fallback to Google Cloud Vision API for low-contrast/low-quality clippings.

Educational notes for B.Tech ECE students:
- Images in OpenCV are represented as NumPy multidimensional arrays (Height, Width, Channels).
- Grayscale conversion reduces data from 3 color channels (BGR) to 1 luminance channel (Y = 0.299R + 0.587G + 0.114B).
- CLAHE (Contrast Limited Adaptive Histogram Equalization) prevents over-amplification of noise
  by operating on local image tiles rather than globally.
- Laplacian convolution calculates the second spatial derivative of the image; high variance
  corresponds to sharp edges (in-focus text), while low variance signifies blur.
"""

import os
import io
import shutil
import logging
from typing import Dict, Any, Tuple, Optional, Union
import numpy as np
import cv2
from PIL import Image

# Configure logging for clean terminal/Colab output
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Khobor.OCR")

# Import pytesseract
try:
    import pytesseract
except ImportError:
    pytesseract = None
    logger.warning("pytesseract is not installed. Install via: pip install pytesseract")

# Import Google Cloud Vision (optional / paid fallback)
try:
    from google.cloud import vision
    HAS_CLOUD_VISION = True
except ImportError:
    vision = None
    HAS_CLOUD_VISION = False


class ImagePreprocessor:
    """
    OpenCV-based image preprocessing pipeline tailored for newspaper typography:
    - BGR to Grayscale
    - Contrast Enhancement (CLAHE)
    - Skew Angle Detection & Deskewing via affine rotation
    - Blurriness estimation using Laplacian variance
    """

    @staticmethod
    def load_image(image_input: Union[str, bytes, np.ndarray, Image.Image]) -> np.ndarray:
        """
        Load an image from various input formats into a BGR NumPy array.

        Parameters
        ----------
        image_input : Union[str, bytes, np.ndarray, Image.Image]
            File path, raw bytes, NumPy array, or PIL Image.

        Returns
        -------
        np.ndarray
            OpenCV BGR image array.
        """
        if isinstance(image_input, np.ndarray):
            # Already a numpy array; ensure 3 channels if RGB/BGR
            if len(image_input.shape) == 2:
                return cv2.cvtColor(image_input, cv2.COLOR_GRAY2BGR)
            return image_input.copy()

        if isinstance(image_input, Image.Image):
            # Convert PIL Image to OpenCV BGR format
            rgb_array = np.array(image_input.convert("RGB"))
            return cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)

        if isinstance(image_input, bytes):
            # Decode byte buffer
            nparr = np.frombuffer(image_input, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Could not decode image bytes.")
            return img

        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                raise FileNotFoundError(f"Image file does not exist: {image_input}")
            img = cv2.imread(image_input, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError(f"OpenCV could not load image from path: {image_input}")
            return img

        raise TypeError(f"Unsupported image input type: {type(image_input)}")

    @staticmethod
    def to_grayscale(image_bgr: np.ndarray) -> np.ndarray:
        """
        Convert BGR image to single-channel Grayscale.
        """
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def enhance_contrast(image_gray: np.ndarray, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
        """
        Enhance newspaper text contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization).

        Unlike standard histogram equalization which operates globally and often washes out
        newsprint paper textures or blows out highlights, CLAHE divides the image into small tiles
        (e.g., 8x8 pixels) and equalizes contrast locally with a clip limit to suppress background noise.
        """
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        return clahe.apply(image_gray)

    @staticmethod
    def check_blur(image_gray: np.ndarray, threshold: float = 80.0) -> Tuple[bool, float]:
        """
        Estimate image blurriness using the variance of the Laplacian operator.

        The Laplacian kernel measures second-order spatial derivatives:
        [ 0,  1,  0 ]
        [ 1, -4,  1 ]
        [ 0,  1,  0 ]
        In an in-focus newspaper photograph, sharp edges between black ink and off-white paper
        produce high variance. Blurry photos have smooth gradients and very low variance.

        Parameters
        ----------
        image_gray : np.ndarray
            Single-channel grayscale image.
        threshold : float
            Threshold below which the image is considered blurry (typical range: 60 - 100).

        Returns
        -------
        Tuple[bool, float]
            (is_blurry, variance_score)
        """
        laplacian = cv2.Laplacian(image_gray, cv2.CV_64F)
        score = float(laplacian.var())
        is_blurry = score < threshold
        return is_blurry, score

    @staticmethod
    def detect_skew_angle(image_gray: np.ndarray) -> float:
        """
        Detect text skew angle (in degrees) using minimum area bounding box on text pixels.

        Returns
        -------
        float
            Detected skew angle between -45.0 and +45.0 degrees.
        """
        # Step 1: Invert and threshold image so text characters are foreground (white pixels)
        # Otsu's thresholding automatically finds the bimodal threshold separating ink from paper
        _, thresh = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

        # Step 2: Grab the (x, y) coordinates of all foreground pixels
        coords = np.column_stack(np.where(thresh > 0))
        if coords.shape[0] < 50:
            # Not enough text pixels to reliably detect skew
            return 0.0

        # Step 3: Compute rotated minimum bounding box enclosing the coordinates
        rect = cv2.minAreaRect(coords)
        angle = rect[-1]

        # In modern OpenCV (>=4.5), angle is in [-90, 0) or [0, 90) depending on orientation
        if angle < -45:
            angle = -(90 + angle)
        elif angle > 45:
            angle = 90 - angle
        else:
            angle = -angle

        # Guard against extreme rotations: newspaper clippings are usually tilted by < 30 degrees
        if abs(angle) > 30.0:
            logger.debug(f"Detected large angle {angle:.2f}°; ignoring to prevent accidental 90° flip.")
            return 0.0

        return float(angle)

    @classmethod
    def deskew(cls, image: np.ndarray, angle: Optional[float] = None) -> Tuple[np.ndarray, float]:
        """
        Rotate the image to correct skew using an affine transformation matrix.

        Parameters
        ----------
        image : np.ndarray
            Grayscale or BGR image.
        angle : Optional[float]
            If provided, rotate by this angle; otherwise auto-detect skew.

        Returns
        -------
        Tuple[np.ndarray, float]
            (deskewed_image, angle_used)
        """
        gray = cls.to_grayscale(image) if len(image.shape) == 3 else image
        if angle is None:
            angle = cls.detect_skew_angle(gray)

        # If angle is negligible (< 0.25 degrees), avoid interpolation loss
        if abs(angle) < 0.25:
            return image, 0.0

        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)

        # Construct 2D affine rotation matrix: [alpha, beta, (1-alpha)*cx - beta*cy]
        rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Warp affine with white background border for newspaper paper look
        deskewed = cv2.warpAffine(
            image,
            rot_matrix,
            (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255) if len(image.shape) == 3 else 255,
        )
        return deskewed, angle

    @classmethod
    def preprocess_pipeline(cls, image_input: Union[str, bytes, np.ndarray, Image.Image]) -> Dict[str, Any]:
        """
        Execute full preprocessing: Load -> Grayscale -> Blur Check -> Contrast -> Deskew.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing original image, preprocessed grayscale image,
            blur metrics, and deskew angle.
        """
        original_bgr = cls.load_image(image_input)
        gray = cls.to_grayscale(original_bgr)

        # Check blurriness on raw grayscale before altering edges
        is_blurry, blur_score = cls.check_blur(gray)

        # Enhance contrast with CLAHE
        enhanced_gray = cls.enhance_contrast(gray)

        # Correct tilt/skew
        deskewed_gray, skew_angle = cls.deskew(enhanced_gray)

        return {
            "original_bgr": original_bgr,
            "preprocessed_gray": deskewed_gray,
            "is_blurry": is_blurry,
            "blur_score": blur_score,
            "deskew_angle": skew_angle,
        }


class OCREngine:
    """
    Dual-engine OCR processor:
    - Default: Free, local Pytesseract (Tesseract OCR).
    - Fallback: Google Cloud Vision API (optional/paid tier for blurry or damaged clippings).
    """

    def __init__(self, tesseract_cmd: Optional[str] = None):
        """
        Initialize the OCR engine. Auto-detects Tesseract executable paths on
        Google Colab (/usr/bin/tesseract) and Windows.
        """
        self._setup_tesseract_path(tesseract_cmd)

    def _setup_tesseract_path(self, user_cmd: Optional[str] = None) -> None:
        """Find and configure the Tesseract binary location."""
        if pytesseract is None:
            return

        if user_cmd and os.path.exists(user_cmd):
            pytesseract.pytesseract.tesseract_cmd = user_cmd
            logger.info(f"Tesseract executable set to: {user_cmd}")
            return

        # Check standard PATH
        which_tesseract = shutil.which("tesseract")
        if which_tesseract:
            pytesseract.pytesseract.tesseract_cmd = which_tesseract
            logger.info(f"Tesseract found in system PATH: {which_tesseract}")
            return

        # Check standard Colab / Linux path
        if os.path.exists("/usr/bin/tesseract"):
            pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"
            return

        # Check common Windows installation paths
        windows_candidates = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            os.path.expanduser(r"~\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"),
        ]
        for candidate in windows_candidates:
            if os.path.exists(candidate):
                pytesseract.pytesseract.tesseract_cmd = candidate
                logger.info(f"Tesseract found on Windows at: {candidate}")
                return

        logger.warning(
            "Tesseract executable not found. "
            "In Google Colab, run: !apt-get install -y tesseract-ocr\n"
            "On Windows, download from: https://github.com/UB-Mannheim/tesseract/wiki"
        )

    def extract_tesseract(self, image_gray: np.ndarray, psm: int = 3) -> str:
        """
        Run Pytesseract on the preprocessed grayscale image.

        Page Segmentation Modes (PSM):
        - PSM 3: Fully automatic page segmentation (default for multi-column newspapers).
        - PSM 1: Automatic page segmentation with OSD (Orientation and Script Detection).
        - PSM 6: Assume a single uniform block of text.
        """
        if pytesseract is None:
            raise RuntimeError("pytesseract package is not installed.")

        # OEM 3: Default, based on what is available (usually LSTM neural network)
        # PSM 3: Automatic page segmentation without OSD
        custom_config = f"--oem 3 --psm {psm}"

        try:
            raw_text = pytesseract.image_to_string(image_gray, config=custom_config)
            return raw_text.strip()
        except pytesseract.TesseractNotFoundError:
            raise RuntimeError(
                "Tesseract is not installed or not in PATH.\n"
                "In Colab: run `!apt-get install -y tesseract-ocr`\n"
                "In Windows: install Tesseract-OCR and configure PATH."
            )
        except Exception as e:
            logger.error(f"Tesseract OCR failed: {str(e)}")
            return ""

    def extract_google_cloud_vision(self, image_input: Union[str, bytes, np.ndarray]) -> str:
        """
        Fallback OCR via Google Cloud Vision API (Optional / Paid-tier).

        Requires:
        - `pip install google-cloud-vision`
        - `GOOGLE_APPLICATION_CREDENTIALS` environment variable pointing to a service account JSON,
          or active Google Cloud authenticated environment.
        """
        if not HAS_CLOUD_VISION:
            raise RuntimeError("google-cloud-vision library is not installed. Run: pip install google-cloud-vision")

        # Prepare JPEG image bytes
        if isinstance(image_input, np.ndarray):
            success, encoded_img = cv2.imencode(".jpg", image_input)
            if not success:
                raise ValueError("Could not encode image array to JPEG for Cloud Vision.")
            content = encoded_img.tobytes()
        elif isinstance(image_input, bytes):
            content = image_input
        elif isinstance(image_input, str):
            with open(image_input, "rb") as f:
                content = f.read()
        else:
            raise TypeError("Unsupported image input type for Cloud Vision.")

        try:
            client = vision.ImageAnnotatorClient()
            vision_image = vision.Image(content=content)

            # Use DOCUMENT_TEXT_DETECTION for dense newspaper text (more accurate than TEXT_DETECTION)
            response = client.document_text_detection(image=vision_image)

            if response.error.message:
                raise RuntimeError(f"Cloud Vision API error: {response.error.message}")

            if response.full_text_annotation and response.full_text_annotation.text:
                return response.full_text_annotation.text.strip()

            return ""
        except Exception as e:
            logger.error(f"Google Cloud Vision OCR failed: {str(e)}")
            raise

    def process_article_image(
        self,
        image_input: Union[str, bytes, np.ndarray, Image.Image],
        enable_cloud_fallback: bool = True,
        blur_threshold: float = 80.0,
        min_word_threshold: int = 15,
    ) -> Dict[str, Any]:
        """
        End-to-end OCR pipeline for newspaper articles:
        1. Preprocesses image (grayscale, contrast, deskew).
        2. Measures blurriness and warns if image is degraded.
        3. Attempts local Tesseract OCR.
        4. If Tesseract yields poor results (< min_word_threshold) and fallback is enabled,
           attempts Google Cloud Vision.

        Returns
        -------
        Dict[str, Any]
            {
                "raw_text": str,
                "engine_used": "tesseract" | "cloud_vision" | "failed",
                "is_blurry": bool,
                "blur_score": float,
                "deskew_angle": float,
                "word_count": int,
                "preprocessed_image": np.ndarray
            }
        """
        # Step 1: Preprocess
        prep_results = ImagePreprocessor.preprocess_pipeline(image_input)
        preprocessed_gray = prep_results["preprocessed_gray"]
        is_blurry = prep_results["is_blurry"]
        blur_score = prep_results["blur_score"]
        deskew_angle = prep_results["deskew_angle"]

        if is_blurry:
            logger.warning(
                f"[Khobor OCR Alert] Image appears blurry (variance score {blur_score:.1f} < {blur_threshold}). "
                "OCR character accuracy may be degraded."
            )

        # Step 2: Run Tesseract
        engine_used = "tesseract"
        extracted_text = ""
        tesseract_error = None

        try:
            extracted_text = self.extract_tesseract(preprocessed_gray, psm=3)
        except Exception as err:
            tesseract_error = str(err)
            logger.warning(f"Local Tesseract OCR encountered an issue: {tesseract_error}")

        word_count = len(extracted_text.split()) if extracted_text else 0

        # Step 3: Check if fallback to Cloud Vision is needed
        needs_fallback = (word_count < min_word_threshold) or (tesseract_error is not None)

        if needs_fallback and enable_cloud_fallback and HAS_CLOUD_VISION:
            # Check if Google Cloud credentials exist before attempting
            if os.getenv("GOOGLE_APPLICATION_CREDENTIALS") or os.getenv("GOOGLE_CLOUD_PROJECT"):
                logger.info(
                    f"Tesseract yielded only {word_count} words. Attempting Google Cloud Vision fallback (paid tier)..."
                )
                try:
                    cloud_text = self.extract_google_cloud_vision(preprocessed_gray)
                    if len(cloud_text.split()) > word_count:
                        extracted_text = cloud_text
                        engine_used = "cloud_vision"
                        word_count = len(extracted_text.split())
                        logger.info("Google Cloud Vision fallback succeeded.")
                except Exception as cv_err:
                    logger.warning(f"Google Cloud Vision fallback unavailable: {str(cv_err)}")
            else:
                logger.info(
                    "Google Cloud Vision fallback skipped: GOOGLE_APPLICATION_CREDENTIALS not configured in environment."
                )

        if not extracted_text:
            engine_used = "failed"
            logger.error("OCR extraction failed to extract any readable text from the provided image.")

        return {
            "raw_text": extracted_text,
            "engine_used": engine_used,
            "is_blurry": is_blurry,
            "blur_score": blur_score,
            "deskew_angle": deskew_angle,
            "word_count": word_count,
            "preprocessed_image": preprocessed_gray,
        }
