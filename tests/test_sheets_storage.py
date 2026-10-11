"""
Unit tests for Google Sheets storage manager and fallback local persistence.
"""

import os
import shutil
import tempfile
import unittest
from sheets_storage import SheetsManager


class TestSheetsStorage(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.manager = SheetsManager(
            spreadsheet_name="Test_Khobor_Sheet",
            offline_mode=True,  # Test offline fallback and structure
        )

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_offline_save_creates_backup_json(self):
        articles = [
            {
                "category": "Polity",
                "headline": "Election Commission Upgrades Digital Auditing",
                "ocr_confidence": "High",
                "summary": "New digital auditing system announced. Speeds up booth level verifications.",
                "vocabulary": [
                    {
                        "word": "expeditious",
                        "meaning": "done with speed and efficiency",
                        "usage_sentence": "An expeditious audit process.",
                        "exam_tier": "Both",
                    }
                ],
                "key_sentences": ["Booth level verifications accelerated."],
                "current_affairs_facts": [
                    {
                        "fact": "Election Commission launched digital auditing in all 543 constituencies.",
                        "likely_relevance": "Polity Constitutional Bodies for CDS",
                    }
                ],
            }
        ]

        summary = self.manager.save_pipeline_output(
            articles=articles,
            raw_ocr_text="Raw OCR sample text...",
            backup_dir=self.temp_dir,
        )

        self.assertEqual(summary["articles_saved"], 1)
        self.assertEqual(summary["vocab_saved"], 1)
        self.assertEqual(summary["facts_saved"], 1)
        self.assertFalse(summary["saved_to_sheets"])  # Offline mode
        self.assertIsNotNone(summary["backup_path"])
        self.assertTrue(os.path.exists(summary["backup_path"]))

        # Verify initial SRS fields were attached to records
        vocab_card = summary["vocabulary_cards"][0]
        self.assertEqual(vocab_card["ease_factor"], 2.5)
        self.assertEqual(vocab_card["interval_days"], 1)
        self.assertEqual(vocab_card["repetitions"], 0)
        self.assertTrue(vocab_card["card_id"].startswith("VOC_"))


if __name__ == "__main__":
    unittest.main()
