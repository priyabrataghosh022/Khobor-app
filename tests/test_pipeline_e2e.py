"""
End-to-end integration test verifying the full pipeline flow:
Image -> Preprocessing -> Structured Extraction -> Sheets Storage -> SRS -> Anki Export.
"""

import os
import shutil
import tempfile
import unittest
import numpy as np
import cv2

from ocr_module import ImagePreprocessor
from gemini_extraction import GeminiExtractor
from sheets_storage import SheetsManager
from anki_export import AnkiDeckExporter
from srs_engine import sm2_update


class TestEndToEndPipeline(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.image_path = os.path.join(self.temp_dir, "test_newspaper.png")

        # Generate synthetic image
        img = np.ones((400, 600, 3), dtype=np.uint8) * 250
        cv2.putText(img, "TEST DEFENCE BILATERAL TREATY", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(img, "New Delhi and Paris ratify joint maritime patrol accord.", (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
        cv2.imwrite(self.image_path, img)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_full_pipeline_flow(self):
        # 1. Preprocessing
        prep = ImagePreprocessor.preprocess_pipeline(self.image_path)
        self.assertIn("preprocessed_gray", prep)
        self.assertFalse(prep["is_blurry"])

        # 2. Mock extracted article matching Gemini schema
        extracted_articles = [
            {
                "category": "International",
                "headline": "Indo-French Maritime Patrol Accord Ratified",
                "ocr_confidence": "High",
                "summary": "India and France have ratified an agreement for joint maritime surveillance. The accord enhances Indian Ocean regional stability. Both nations emphasized freedom of navigation.",
                "vocabulary": [
                    {
                        "word": "bilateral",
                        "meaning": "having or relating to two sides; affecting both sides",
                        "usage_sentence": "A landmark bilateral maritime treaty.",
                        "exam_tier": "Both",
                    },
                    {
                        "word": "surveillance",
                        "meaning": "close observation, especially of a suspected spy or criminal",
                        "usage_sentence": "Joint maritime surveillance flights initiated.",
                        "exam_tier": "CDS",
                    },
                ],
                "key_sentences": [
                    "India and France ratify joint maritime patrol accord in New Delhi."
                ],
                "current_affairs_facts": [
                    {
                        "fact": "India and France signed joint surveillance pact in Indian Ocean Region in 2026.",
                        "likely_relevance": "CDS International Relations & Defence Pacts",
                    }
                ],
            }
        ]

        # Validate schema
        validated_articles = [GeminiExtractor.validate_article_schema(a) for a in extracted_articles]
        self.assertEqual(len(validated_articles), 1)

        # 3. Sheets storage (offline mode)
        sheets = SheetsManager(spreadsheet_name="Test_Sheet", offline_mode=True)
        storage_result = sheets.save_pipeline_output(
            articles=validated_articles,
            raw_ocr_text="Raw OCR snippet from synthetic image",
            backup_dir=self.temp_dir,
        )

        self.assertEqual(storage_result["articles_saved"], 1)
        self.assertEqual(storage_result["vocab_saved"], 2)
        self.assertEqual(storage_result["facts_saved"], 1)

        # 4. Anki Export
        exporter = AnkiDeckExporter(
            registry_path=os.path.join(self.temp_dir, "test_registry.json"),
            deck_prefix="Readloop",
        )
        anki_result = exporter.export_decks(
            vocab_items=storage_result["vocabulary_cards"],
            fact_items=storage_result["current_affairs_cards"],
            output_dir=os.path.join(self.temp_dir, "decks"),
        )

        self.assertTrue(os.path.exists(anki_result["vocab_deck_path"]))
        self.assertTrue(os.path.exists(anki_result["facts_deck_path"]))
        self.assertEqual(anki_result["vocab_cards_added"], 2)
        self.assertEqual(anki_result["facts_cards_added"], 1)

        # 5. SM-2 review calculation test
        vocab_card = storage_result["vocabulary_cards"][0]
        ef, interval, reps = sm2_update(
            vocab_card["ease_factor"],
            vocab_card["interval_days"],
            vocab_card["repetitions"],
            quality=5,
        )
        self.assertEqual(reps, 1)
        self.assertEqual(interval, 1)
        self.assertGreater(ef, 2.5)


if __name__ == "__main__":
    unittest.main()
