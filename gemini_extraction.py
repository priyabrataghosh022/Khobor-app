"""
Khobor - Gemini Article Extraction & Structuring Module
=======================================================
Extracts high-value exam content (vocabulary, current affairs, summary)
from raw OCR newspaper text using Google's Gemini API (google-generativeai SDK).

Designed specifically for CDS (Combined Defence Services) and GATE-style
general aptitude exam preparation.

Key features:
1. Exact system prompt configured with CDS and GATE vocabulary tagging.
2. Supports multi-article segmentation (handles single object or array).
3. Resilient JSON parser that strips markdown code fences if present.
4. Auto-retry mechanism: sends "your last response was invalid JSON, return only valid JSON"
   on malformed responses.
5. Strict schema validation to prevent malformed data from reaching Google Sheets.
6. Graceful rate-limit and error handling.
"""

import os
import json
import re
import time
import logging
from typing import List, Dict, Any, Union, Optional

# Configure module logger
logger = logging.getLogger("Khobor.Gemini")

try:
    import google.generativeai as genai
    from google.api_core import exceptions as google_exceptions
    HAS_GENAI = True
except ImportError:
    genai = None
    HAS_GENAI = False
    logger.warning("google-generativeai is not installed. Install via: pip install google-generativeai")


# ==============================================================================
# EXACT SYSTEM PROMPT (as specified in project requirements)
# ==============================================================================
EXACT_SYSTEM_PROMPT = """You are an exam-prep assistant helping a student preparing for CDS 
(Combined Defence Services) and GATE-style general aptitude exams 
extract high-value content from a photographed newspaper article.

STEP 0 — Cleanup & validation (do this silently, don't output it):
- The input is raw OCR text and may contain broken words, missing 
  spaces, stray line breaks, or merged columns from adjacent articles. 
  Mentally reconstruct the likely original text before proceeding.
- Check if the text actually contains MORE THAN ONE distinct article. 
  If so, return a JSON ARRAY, one object per article. If single article, 
  return a single JSON object (not wrapped in an array).

STEP 1 — Extract using this exact schema per article:
{
  "category": "Business | Editorial | Opinion | Tech | Sports | Polity | International | Other",
  "headline": "Headline or short title if none given",
  "ocr_confidence": "High | Medium | Low",
  "summary": "3-4 sentence neutral summary",
  "vocabulary": [
    {"word": "...", "meaning": "...", "usage_sentence": "...", "exam_tier": "CDS | GATE-aptitude | Both"}
  ],
  "key_sentences": ["..."],
  "current_affairs_facts": [
    {"fact": "...", "likely_relevance": "..."}
  ]
}

Rules:
- Vocabulary: 5-10 words. CDS favors formal/literary register; 
  GATE-aptitude favors precise/logical register. Tag accordingly.
- Only include concrete, testable current_affairs_facts.
- If OCR reconstruction is too unreliable, say so in summary instead 
  of guessing.
- Return ONLY the JSON — no preamble, no markdown fences."""


VALID_CATEGORIES = {
    "Business", "Editorial", "Opinion", "Tech", "Sports", "Polity", "International", "Other"
}
VALID_CONFIDENCE = {"High", "Medium", "Low"}
VALID_EXAM_TIERS = {"CDS", "GATE-aptitude", "Both"}


class SchemaValidationError(Exception):
    """Raised when extracted JSON does not match the required schema."""
    pass


class GeminiExtractor:
    """
    Interface for extracting exam content from OCR text using Google Gemini.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-1.5-flash",
    ):
        """
        Initialize the Gemini Extractor.

        Parameters
        ----------
        api_key : Optional[str]
            Gemini API key. If not passed, checks GEMINI_API_KEY environment variable.
        model_name : str
            Gemini model to use. Defaults to 'gemini-1.5-flash' for fast, high-quality responses.
        """
        if not HAS_GENAI:
            raise RuntimeError(
                "google-generativeai package is missing. "
                "Run `pip install google-generativeai` in your terminal or Colab."
            )

        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            logger.warning(
                "No Gemini API key supplied. Provide api_key or set GEMINI_API_KEY environment variable."
            )

        self.model_name = model_name
        self._init_client()

    def _init_client(self) -> None:
        """Configure genai with the API key and set up the model."""
        if not self.api_key:
            return

        genai.configure(api_key=self.api_key)

        # GenerativeModel initialization with the exact system instruction
        generation_config = {
            "temperature": 0.2,       # Low temperature for factual consistency and stable JSON
            "top_p": 0.95,
            "max_output_tokens": 4096,
        }

        # Check if the model supports JSON response_mime_type (available on gemini-1.5-* and 2.0-*)
        try:
            generation_config["response_mime_type"] = "application/json"
        except Exception:
            pass

        self.model = genai.GenerativeModel(
            model_name=self.model_name,
            generation_config=generation_config,
            system_instruction=EXACT_SYSTEM_PROMPT,
        )

    @staticmethod
    def clean_json_text(raw_text: str) -> str:
        """
        Strip markdown formatting (```json ... ```) or preamble text if the LLM
        inadvertently included it.

        Parameters
        ----------
        raw_text : str
            Raw text from LLM response.

        Returns
        -------
        str
            Cleaned JSON string ready for json.loads().
        """
        text = raw_text.strip()

        # Remove markdown code fences like ```json and ```
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
            text = re.sub(r"\s*```$", "", text)
            text = text.strip()

        # In case there is any leading text before the first '{' or '['
        first_brace = text.find("{")
        first_bracket = text.find("[")

        if first_brace != -1 and (first_bracket == -1 or first_brace < first_bracket):
            # Object starts with '{'
            last_brace = text.rfind("}")
            if last_brace != -1:
                text = text[first_brace : last_brace + 1]
        elif first_bracket != -1:
            # Array starts with '['
            last_bracket = text.rfind("]")
            if last_bracket != -1:
                text = text[first_bracket : last_bracket + 1]

        return text

    @classmethod
    def validate_article_schema(cls, article: Dict[str, Any], article_index: int = 0) -> Dict[str, Any]:
        """
        Validate a single article dictionary against the required schema.
        Sanitizes minor discrepancies and raises SchemaValidationError on missing mandatory fields.

        Parameters
        ----------
        article : Dict[str, Any]
            Extracted article dictionary.
        article_index : int
            Index of article for logging / error context.

        Returns
        -------
        Dict[str, Any]
            Sanitized and validated article dictionary.
        """
        if not isinstance(article, dict):
            raise SchemaValidationError(f"Article at index {article_index} is not a JSON object: {type(article)}")

        # Check category
        category = str(article.get("category", "Other")).strip()
        matched_cat = next((c for c in VALID_CATEGORIES if c.lower() == category.lower()), "Other")
        article["category"] = matched_cat

        # Check headline
        headline = str(article.get("headline", "")).strip()
        if not headline:
            article["headline"] = "Untitled Newspaper Article"
        else:
            article["headline"] = headline

        # Check OCR confidence
        conf = str(article.get("ocr_confidence", "Medium")).capitalize().strip()
        if conf not in VALID_CONFIDENCE:
            conf = "Medium"
        article["ocr_confidence"] = conf

        # Check summary
        summary = str(article.get("summary", "")).strip()
        if not summary:
            raise SchemaValidationError(f"Article '{article['headline']}' is missing required 'summary' field.")
        article["summary"] = summary

        # Check vocabulary list
        vocab_list = article.get("vocabulary", [])
        if not isinstance(vocab_list, list):
            logger.warning(f"Vocabulary field is not a list in article '{article['headline']}'; resetting to [].")
            vocab_list = []

        validated_vocab = []
        for v in vocab_list:
            if not isinstance(v, dict):
                continue
            word = str(v.get("word", "")).strip()
            meaning = str(v.get("meaning", "")).strip()
            sentence = str(v.get("usage_sentence", "")).strip()
            tier = str(v.get("exam_tier", "Both")).strip()

            if not word or not meaning:
                continue

            # Standardize tier
            matched_tier = "Both"
            for valid_t in VALID_EXAM_TIERS:
                if valid_t.lower() == tier.lower():
                    matched_tier = valid_t
                    break

            validated_vocab.append({
                "word": word,
                "meaning": meaning,
                "usage_sentence": sentence,
                "exam_tier": matched_tier,
            })
        article["vocabulary"] = validated_vocab

        # Check key sentences
        key_sentences = article.get("key_sentences", [])
        if not isinstance(key_sentences, list):
            key_sentences = [str(key_sentences)] if key_sentences else []
        article["key_sentences"] = [str(s).strip() for s in key_sentences if str(s).strip()]

        # Check current affairs facts
        facts = article.get("current_affairs_facts", [])
        if not isinstance(facts, list):
            facts = []

        validated_facts = []
        for f in facts:
            if not isinstance(f, dict):
                continue
            fact_str = str(f.get("fact", "")).strip()
            relevance = str(f.get("likely_relevance", "")).strip()
            if fact_str:
                validated_facts.append({
                    "fact": fact_str,
                    "likely_relevance": relevance,
                })
        article["current_affairs_facts"] = validated_facts

        return article

    def _parse_and_validate(self, response_text: str) -> List[Dict[str, Any]]:
        """
        Parse raw model response string and validate against schema.
        Handles both single-object and array responses.
        """
        cleaned_text = self.clean_json_text(response_text)
        try:
            data = json.loads(cleaned_text)
        except json.JSONDecodeError as decode_err:
            raise SchemaValidationError(f"Invalid JSON syntax: {str(decode_err)}") from decode_err

        # Requirement: "If so, return a JSON ARRAY, one object per article. If single article,
        # return a single JSON object (not wrapped in an array)."
        # We normalize internal pipeline output to a List[Dict[str, Any]]
        if isinstance(data, dict):
            articles = [data]
        elif isinstance(data, list):
            articles = data
        else:
            raise SchemaValidationError(f"Expected JSON object or array, got {type(data).__name__}")

        validated_articles = []
        for idx, art in enumerate(articles):
            validated_articles.append(self.validate_article_schema(art, article_index=idx))

        return validated_articles

    def extract(self, raw_ocr_text: str, max_retries: int = 1) -> List[Dict[str, Any]]:
        """
        Send raw OCR text to Gemini, parse response, and validate schema.
        On malformed JSON, retries once with the exact required follow-up message:
        "your last response was invalid JSON, return only valid JSON".

        Parameters
        ----------
        raw_ocr_text : str
            Uncleaned text extracted by OCR.
        max_retries : int
            Number of retries on malformed JSON (default 1 as specified).

        Returns
        -------
        List[Dict[str, Any]]
            List of validated article data dictionaries.
        """
        if not raw_ocr_text or not raw_ocr_text.strip():
            logger.warning("Empty raw OCR text passed to GeminiExtractor.")
            return []

        if not hasattr(self, "model") or self.model is None:
            self._init_client()
            if not hasattr(self, "model") or self.model is None:
                raise RuntimeError(
                    "Gemini client not initialized. Please provide a valid Gemini API key."
                )

        # Prepare user prompt with the OCR text
        user_prompt = (
            "Here is the raw OCR text from a photographed newspaper article:\n\n"
            "-------------------- OCR TEXT START --------------------\n"
            f"{raw_ocr_text}\n"
            "-------------------- OCR TEXT END ----------------------\n\n"
            "Extract the content according to the system instructions and return ONLY valid JSON."
        )

        # Start a multi-turn chat session so we can easily send a retry follow-up if needed
        chat = self.model.start_chat(history=[])

        # Attempt 1: Initial extraction
        try:
            logger.info("Sending raw OCR text to Gemini API for exam analysis...")
            response = chat.send_message(user_prompt)
            return self._parse_and_validate(response.text)

        except (SchemaValidationError, json.JSONDecodeError) as parse_error:
            logger.warning(
                f"Attempt 1 yielded malformed JSON/schema error: {str(parse_error)}. "
                "Retrying with follow-up instruction as per requirements..."
            )

            # Retry once with follow-up message
            if max_retries >= 1:
                try:
                    retry_message = "your last response was invalid JSON, return only valid JSON"
                    retry_response = chat.send_message(retry_message)
                    return self._parse_and_validate(retry_response.text)
                except Exception as retry_err:
                    logger.error(f"Retry attempt also failed: {str(retry_err)}")
                    raise

            raise

        except google_exceptions.ResourceExhausted as quota_err:
            logger.error(
                "Gemini API rate limit (ResourceExhausted / 429) encountered. "
                "Please wait 30 seconds or check your Gemini API quota."
            )
            raise quota_err

        except google_exceptions.GoogleAPICallError as api_err:
            logger.error(f"Gemini API call failed: {str(api_err)}")
            raise api_err
