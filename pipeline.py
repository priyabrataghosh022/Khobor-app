"""
Khobor - Unified End-to-End Pipeline Runner & CLI
==================================================
Runs the complete newspaper-to-flashcard pipeline in a single command or function call:
1. Image Load & OpenCV Preprocessing (CLAHE, Deskew, Blur Check)
2. OCR Extraction (Tesseract with Cloud Vision fallback)
3. Gemini AI Analysis (CDS & GATE vocabulary, current affairs, summary)
4. Google Sheets Storage (Articles, Vocabulary, CurrentAffairs tabs)
5. SM-2 Spaced Repetition Initialization
6. Anki Deck Generation (.apkg files with deduplication)

Usage via CLI:
    python pipeline.py --image photo.jpg --sheet "Khobor_Exam_Prep"
    python pipeline.py --demo
"""

import os
import sys
import argparse
import logging
from typing import Dict, Any, Optional

from ocr_module import OCREngine, ImagePreprocessor
from gemini_extraction import GeminiExtractor
from sheets_storage import SheetsManager
from anki_export import AnkiDeckExporter
from srs_engine import get_initial_srs_state

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Khobor.Pipeline")


def run_pipeline(
    image_path: str,
    gemini_api_key: Optional[str] = None,
    spreadsheet_name: str = "Khobor_Exam_Prep",
    offline_sheets: bool = False,
    deck_prefix: str = "Readloop",
    output_dir: str = "./anki_decks",
    gemini_model: str = "gemini-1.5-flash",
) -> Dict[str, Any]:
    """
    Execute the entire Khobor pipeline end-to-end on a single newspaper photograph.

    Parameters
    ----------
    image_path : str
        Path to image file on disk.
    gemini_api_key : Optional[str]
        Gemini API key. Defaults to GEMINI_API_KEY environment variable.
    spreadsheet_name : str
        Name of target Google Sheet workbook.
    offline_sheets : bool
        If True, saves to local JSON backup files instead of calling Google Sheets API.
    deck_prefix : str
        Prefix for generated Anki decks (default: "Readloop").
    output_dir : str
        Directory where .apkg files are saved.
    gemini_model : str
        Gemini model variant.

    Returns
    -------
    Dict[str, Any]
        Complete results dictionary with OCR metrics, extracted articles,
        storage summary, and Anki deck paths.
    """
    logger.info(f"===> Step 1/5: Loading & Preprocessing Image: {image_path}")
    ocr_engine = OCREngine()
    ocr_res = ocr_engine.process_article_image(image_path, enable_cloud_fallback=True)

    raw_text = ocr_res["raw_text"]
    if not raw_text or not raw_text.strip():
        logger.error("OCR extracted no text. Pipeline halted.")
        return {
            "success": False,
            "error": "OCR failed to detect text in image.",
            "ocr_metrics": ocr_res,
        }

    logger.info(
        f"OCR extraction successful ({ocr_res['word_count']} words, "
        f"Engine: {ocr_res['engine_used']}, Blur score: {ocr_res['blur_score']:.1f}, "
        f"Deskew: {ocr_res['deskew_angle']:.2f}°)"
    )

    logger.info("===> Step 2/5: Analyzing Article Content with Gemini AI")
    extractor = GeminiExtractor(api_key=gemini_api_key, model_name=gemini_model)
    articles = extractor.extract(raw_text)
    logger.info(f"Gemini successfully segmented and analyzed {len(articles)} article(s).")

    logger.info(f"===> Step 3/5: Saving Records to Google Sheets ('{spreadsheet_name}')")
    sheets_mgr = SheetsManager(spreadsheet_name=spreadsheet_name, offline_mode=offline_sheets)
    if not offline_sheets:
        sheets_mgr.connect_or_create()

    storage_res = sheets_mgr.save_pipeline_output(
        articles=articles,
        raw_ocr_text=raw_text,
        backup_dir="./backups",
    )
    logger.info(
        f"Storage complete: {storage_res['articles_saved']} article(s), "
        f"{storage_res['vocab_saved']} vocab card(s), {storage_res['facts_saved']} fact card(s)."
    )

    logger.info("===> Step 4/5: Exporting SM-2 Spaced Repetition Decks to Anki (.apkg)")
    exporter = AnkiDeckExporter(registry_path="./anki_registry.json", deck_prefix=deck_prefix)
    anki_res = exporter.export_decks(
        vocab_items=storage_res["vocabulary_cards"],
        fact_items=storage_res["current_affairs_cards"],
        output_dir=output_dir,
    )

    logger.info(
        f"Anki Export complete: {anki_res['vocab_cards_added']} vocab added, "
        f"{anki_res['facts_cards_added']} facts added."
    )
    logger.info(f"Vocabulary Deck: {anki_res['vocab_deck_path']}")
    logger.info(f"Current Affairs Deck: {anki_res['facts_deck_path']}")

    return {
        "success": True,
        "ocr_metrics": {
            "engine": ocr_res["engine_used"],
            "word_count": ocr_res["word_count"],
            "blur_score": ocr_res["blur_score"],
            "is_blurry": ocr_res["is_blurry"],
            "deskew_angle": ocr_res["deskew_angle"],
        },
        "articles": articles,
        "storage": storage_res,
        "anki": anki_res,
    }


def create_demo_image(output_path: str = "demo_newspaper_article.png") -> str:
    """Create a sample newspaper clipping for testing."""
    import cv2
    import numpy as np

    img = np.ones((650, 900, 3), dtype=np.uint8) * 248
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_bold = cv2.FONT_HERSHEY_COMPLEX

    cv2.putText(img, "DAILY DEFENCE CHRONICLE", (260, 45), font_bold, 0.8, (20, 20, 20), 2)
    cv2.line(img, (40, 60), (860, 60), (40, 40, 40), 2)

    cv2.putText(img, "PARLIAMENT RATIFIES INDIGENOUS DEFENCE PROCUREMENT POLICY", (50, 110), font, 0.6, (0, 0, 0), 2)
    lines = [
        "NEW DELHI: The Union Parliament on Wednesday ratified the Defence Indigenisation Policy 2026.",
        "The legislation establishes a 65% domestic sourcing mandate for armed forces capital procurements.",
        "Addressing the house, the Raksha Mantri stated that indigenous self-reliance is paramount",
        "to mitigate geopolitical vulnerabilities and curtail fiscal outflows to foreign defence contractors.",
        "The newly constituted National Defence Acquisition Council (NDAC), headed by the Chief of Defence Staff,",
        "will oversee strategic acquisitions under an expeditious timeline. Opposition leaders requested laconic reports.",
    ]
    y = 150
    for line in lines:
        cv2.putText(img, line, (50, y), font, 0.44, (30, 30, 30), 1)
        y += 32

    cv2.imwrite(output_path, img)
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Khobor - Newspaper to Flashcard Pipeline")
    parser.add_argument("--image", type=str, help="Path to newspaper article image")
    parser.add_argument("--demo", action="store_true", help="Generate and process demo newspaper image")
    parser.add_argument("--sheet", type=str, default="Khobor_Exam_Prep", help="Google Sheet title")
    parser.add_argument("--offline", action="store_true", help="Run Sheets storage in offline backup mode")
    parser.add_argument("--deck-prefix", type=str, default="Readloop", help="Anki deck prefix")
    parser.add_argument("--out-dir", type=str, default="./anki_decks", help="Directory for .apkg files")

    args = parser.parse_args()

    target_image = args.image
    if args.demo or not target_image:
        target_image = create_demo_image()
        print(f"Created demo newspaper clipping: {target_image}")

    res = run_pipeline(
        image_path=target_image,
        spreadsheet_name=args.sheet,
        offline_sheets=args.offline,
        deck_prefix=args.deck_prefix,
        output_dir=args.out_dir,
    )

    if res.get("success"):
        print("\n" + "=" * 60)
        print("🎉 KHOBOR PIPELINE EXECUTED SUCCESSFULLY!")
        print("=" * 60)
        print(f"• Articles Processed : {len(res['articles'])}")
        print(f"• Vocabulary Cards   : {res['storage']['vocab_saved']}")
        print(f"• Current Affairs    : {res['storage']['facts_saved']}")
        print(f"• Vocab Anki Deck    : {res['anki']['vocab_deck_path']}")
        print(f"• Facts Anki Deck    : {res['anki']['facts_deck_path']}")
        if res['storage'].get('sheet_url'):
            print(f"• Google Sheets Link : {res['storage']['sheet_url']}")
        print("=" * 60)
    else:
        print(f"Pipeline error: {res.get('error')}")


if __name__ == "__main__":
    main()
