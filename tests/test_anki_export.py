"""
Unit tests for the Anki deck exporter and deduplication engine.
"""

import os
import shutil
import tempfile
import unittest
from anki_export import AnkiDeckExporter


class TestAnkiExport(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.registry_file = os.path.join(self.temp_dir, "test_registry.json")
        self.decks_dir = os.path.join(self.temp_dir, "test_decks")
        self.exporter = AnkiDeckExporter(
            registry_path=self.registry_file,
            deck_prefix="Readloop",
        )

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_export_decks_creates_apkg_files(self):
        vocab_items = [
            {
                "word": "lucid",
                "meaning": "expressed clearly; easy to understand",
                "usage_sentence": "He gave a lucid explanation of the defence doctrine.",
                "exam_tier": "GATE-aptitude",
                "headline_source": "National Strategy Brief",
                "card_id": "VOC_001",
            },
            {
                "word": "bellicose",
                "meaning": "demonstrating aggression and willingness to fight",
                "usage_sentence": "The ambassador rejected the bellicose statements.",
                "exam_tier": "CDS",
                "headline_source": "Geopolitics Today",
                "card_id": "VOC_002",
            },
        ]

        fact_items = [
            {
                "fact": "India achieved 75% indigenisation in naval vessel construction in 2026.",
                "likely_relevance": "CDS Defence Technology & Atmanirbhar Bharat questions",
                "category": "Polity",
                "headline_source": "Naval Modernisation Report",
                "card_id": "FACT_001",
            }
        ]

        summary = self.exporter.export_decks(vocab_items, fact_items, output_dir=self.decks_dir)

        # Verify .apkg files exist
        self.assertTrue(os.path.exists(summary["vocab_deck_path"]))
        self.assertTrue(os.path.exists(summary["facts_deck_path"]))
        self.assertEqual(summary["vocab_cards_added"], 2)
        self.assertEqual(summary["facts_cards_added"], 1)

    def test_deduplication_on_rerun(self):
        vocab_items = [
            {
                "word": "laconic",
                "meaning": "using very few words",
                "usage_sentence": "His laconic reply stunned the committee.",
                "exam_tier": "Both",
                "headline_source": "Briefings",
                "card_id": "VOC_LACONIC",
            }
        ]
        fact_items = []

        # Run 1: Card should be added
        res1 = self.exporter.export_decks(vocab_items, fact_items, output_dir=self.decks_dir)
        self.assertEqual(res1["vocab_cards_added"], 1)
        self.assertEqual(res1["vocab_cards_skipped"], 0)

        # Run 2: Same card should be recognized as duplicate and skipped
        res2 = self.exporter.export_decks(vocab_items, fact_items, output_dir=self.decks_dir)
        self.assertEqual(res2["vocab_cards_added"], 0)
        self.assertEqual(res2["vocab_cards_skipped"], 1)


if __name__ == "__main__":
    unittest.main()
