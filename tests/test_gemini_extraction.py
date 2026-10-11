"""
Unit tests for Gemini extraction, JSON cleanup, and schema validation.
"""

import unittest
from gemini_extraction import GeminiExtractor, SchemaValidationError


class TestGeminiExtraction(unittest.TestCase):

    def test_clean_json_text_pure_json(self):
        raw = '{"category": "Polity", "headline": "Test Headline"}'
        cleaned = GeminiExtractor.clean_json_text(raw)
        self.assertEqual(cleaned, raw)

    def test_clean_json_text_with_markdown_fences(self):
        raw = '```json\n{"category": "Polity", "headline": "Test"}\n```'
        cleaned = GeminiExtractor.clean_json_text(raw)
        self.assertEqual(cleaned, '{"category": "Polity", "headline": "Test"}')

    def test_clean_json_text_with_preamble(self):
        raw = 'Here is your structured JSON:\n[{"category": "Tech", "headline": "AI Update"}]\nHope this helps!'
        cleaned = GeminiExtractor.clean_json_text(raw)
        self.assertEqual(cleaned, '[{"category": "Tech", "headline": "AI Update"}]')

    def test_validate_article_schema_valid_single_article(self):
        article = {
            "category": "polity",
            "headline": "Defence Modernisation Act",
            "ocr_confidence": "high",
            "summary": "The bill updates defence procurement protocols. It focuses on indigenous tech. Expected to pass this week.",
            "vocabulary": [
                {
                    "word": "indigenous",
                    "meaning": "originating or occurring naturally in a particular place; native",
                    "usage_sentence": "Focusing on indigenous technology.",
                    "exam_tier": "cds",
                }
            ],
            "key_sentences": ["Procurement protocol updated."],
            "current_affairs_facts": [
                {
                    "fact": "Defence Modernisation bill was tabled in Parliament.",
                    "likely_relevance": "CDS Polity & Defence Current Affairs",
                }
            ],
        }

        validated = GeminiExtractor.validate_article_schema(article)
        self.assertEqual(validated["category"], "Polity")
        self.assertEqual(validated["ocr_confidence"], "High")
        self.assertEqual(validated["vocabulary"][0]["exam_tier"], "CDS")
        self.assertEqual(len(validated["vocabulary"]), 1)
        self.assertEqual(len(validated["current_affairs_facts"]), 1)

    def test_validate_article_schema_missing_summary_raises(self):
        invalid_article = {
            "category": "Tech",
            "headline": "Missing summary here",
            "summary": "",
        }
        with self.assertRaises(SchemaValidationError):
            GeminiExtractor.validate_article_schema(invalid_article)


if __name__ == "__main__":
    unittest.main()
