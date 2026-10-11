"""
Khobor - Google Sheets Storage Module
=====================================
Manages persistent storage of articles, vocabulary, and current affairs in Google Sheets.

Features:
- Authenticates using Google Colab's built-in auth helper (`google.colab.auth`) or
  service account credentials outside Colab.
- Creates or connects to a Google Sheet with three distinct tabs:
  1. "Articles"
  2. "Vocabulary"
  3. "CurrentAffairs"
- Enforces relational links: every vocabulary and fact card includes `article_id`
  and `headline_source` referencing its source article.
- Stores SM-2 spaced repetition tracking fields (`ease_factor`, `interval_days`,
  `repetitions`, `next_review_date`, `last_reviewed`).
- Resilient error handling: if Sheets write fails (e.g. quota limit, network glitch),
  it logs the issue, dumps a local JSON backup, and continues the pipeline without crashing.
"""

import os
import json
import uuid
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger("Khobor.Sheets")

# Import gspread and auth helpers
try:
    import gspread
    from google.auth import default
    HAS_GSPREAD = True
except ImportError:
    gspread = None
    HAS_GSPREAD = False
    logger.warning("gspread or google-auth not installed. Install via: pip install gspread google-auth")


# Standard Header Definitions
ARTICLES_HEADERS = [
    "article_id",
    "timestamp",
    "category",
    "headline",
    "ocr_confidence",
    "summary",
    "key_sentences",
    "raw_ocr_snippet",
]

VOCABULARY_HEADERS = [
    "card_id",
    "timestamp",
    "article_id",
    "headline_source",
    "word",
    "meaning",
    "usage_sentence",
    "exam_tier",
    "ease_factor",
    "interval_days",
    "repetitions",
    "next_review_date",
    "last_reviewed",
]

CURRENT_AFFAIRS_HEADERS = [
    "card_id",
    "timestamp",
    "article_id",
    "headline_source",
    "fact",
    "likely_relevance",
    "category",
    "ease_factor",
    "interval_days",
    "repetitions",
    "next_review_date",
    "last_reviewed",
]


class SheetsManager:
    """
    Manages Google Sheets authentication, worksheet creation, and row insertion.
    """

    def __init__(
        self,
        spreadsheet_name: str = "Khobor_Exam_Prep",
        service_account_file: Optional[str] = None,
        offline_mode: bool = False,
    ):
        """
        Initialize the Sheets manager.

        Parameters
        ----------
        spreadsheet_name : str
            Title of the Google Sheet workbook.
        service_account_file : Optional[str]
            Path to service account JSON (for local runs outside Colab).
        offline_mode : bool
            If True, skips Google auth and saves to local JSON backup files.
        """
        self.spreadsheet_name = spreadsheet_name
        self.service_account_file = service_account_file
        self.offline_mode = offline_mode
        self.client: Optional[Any] = None
        self.spreadsheet: Optional[Any] = None

        if not self.offline_mode and HAS_GSPREAD:
            self._authenticate()

    def _authenticate(self) -> None:
        """
        Authenticate with Google Sheets API.
        Prioritizes Colab auth, then service account file, then default Google credentials.
        """
        try:
            # 1. Attempt Google Colab Authentication
            try:
                from google.colab import auth  # type: ignore
                logger.info("Detected Google Colab environment. Triggering Colab authentication...")
                auth.authenticate_user()
                creds, _ = default()
                self.client = gspread.authorize(creds)
                logger.info("Google Colab authorization successful.")
                return
            except (ImportError, Exception):
                logger.debug("Google Colab auth not applicable or user skipped auth.")

            # 2. Attempt Service Account JSON file (for local development)
            if self.service_account_file and os.path.exists(self.service_account_file):
                logger.info(f"Authenticating with service account JSON: {self.service_account_file}")
                self.client = gspread.service_account(filename=self.service_account_file)
                return

            # 3. Attempt default environment credentials (e.g. gcloud CLI)
            creds, _ = default()
            self.client = gspread.authorize(creds)
            logger.info("Google default credentials authorization successful.")

        except Exception as auth_err:
            logger.warning(
                f"Google Sheets authentication could not be completed ({str(auth_err)}). "
                "Switching to offline fallback mode (local JSON storage)."
            )
            self.offline_mode = True

    def connect_or_create(self) -> bool:
        """
        Open existing spreadsheet or create a new one with the 3 required tabs:
        'Articles', 'Vocabulary', 'CurrentAffairs'.

        Returns
        -------
        bool
            True if connected to Google Sheets, False if running offline.
        """
        if self.offline_mode or self.client is None:
            logger.info(f"Running in offline mode. Records will be saved to local backup files.")
            return False

        try:
            try:
                self.spreadsheet = self.client.open(self.spreadsheet_name)
                logger.info(f"Connected to existing Google Sheet: '{self.spreadsheet_name}'")
            except gspread.SpreadsheetNotFound:
                logger.info(f"Spreadsheet '{self.spreadsheet_name}' not found. Creating a new one...")
                self.spreadsheet = self.client.create(self.spreadsheet_name)
                logger.info(f"Created new Google Sheet: '{self.spreadsheet_name}' (ID: {self.spreadsheet.id})")

            # Ensure all three worksheets exist with correct headers
            self._ensure_tab("Articles", ARTICLES_HEADERS)
            self._ensure_tab("Vocabulary", VOCABULARY_HEADERS)
            self._ensure_tab("CurrentAffairs", CURRENT_AFFAIRS_HEADERS)

            # Clean up default "Sheet1" if our custom tabs exist
            try:
                sheet1 = self.spreadsheet.worksheet("Sheet1")
                # Only delete Sheet1 if we have other sheets
                if len(self.spreadsheet.worksheets()) > 1:
                    self.spreadsheet.del_worksheet(sheet1)
            except (gspread.WorksheetNotFound, Exception):
                pass

            return True

        except Exception as err:
            logger.error(f"Failed to connect or create Google Sheet: {str(err)}. Falling back to offline mode.")
            self.offline_mode = True
            return False

    def _ensure_tab(self, tab_name: str, expected_headers: List[str]) -> Any:
        """Ensure a worksheet tab exists and contains header row."""
        try:
            ws = self.spreadsheet.worksheet(tab_name)
        except gspread.WorksheetNotFound:
            logger.info(f"Tab '{tab_name}' not found; creating worksheet...")
            ws = self.spreadsheet.add_worksheet(title=tab_name, rows="100", cols=str(len(expected_headers) + 2))

        # Check existing headers
        try:
            existing_row = ws.row_values(1)
            if not existing_row or existing_row != expected_headers:
                ws.insert_row(expected_headers, index=1)
                logger.info(f"Initialized headers in tab '{tab_name}'.")
        except Exception as header_err:
            logger.warning(f"Could not verify headers in '{tab_name}': {str(header_err)}")

        return ws

    def get_sheet_url(self) -> Optional[str]:
        """Return the Google Sheets web URL for user convenience in Colab."""
        if self.spreadsheet:
            return self.spreadsheet.url
        return None

    def save_pipeline_output(
        self,
        articles: List[Dict[str, Any]],
        raw_ocr_text: str = "",
        backup_dir: str = "./backups",
    ) -> Dict[str, Any]:
        """
        Save parsed articles, vocabulary flashcards, and current affairs facts to Google Sheets.
        If Google Sheets write fails, saves to a local JSON file to ensure no student data is lost.

        Parameters
        ----------
        articles : List[Dict[str, Any]]
            Validated article objects returned by GeminiExtractor.
        raw_ocr_text : str
            Original OCR text for source snippet logging.
        backup_dir : str
            Directory to store local JSON backups if needed.

        Returns
        -------
        Dict[str, Any]
            Summary of saved counts:
            {
                "articles_saved": int,
                "vocab_saved": int,
                "facts_saved": int,
                "sheet_url": Optional[str],
                "saved_to_sheets": bool,
                "backup_path": Optional[str]
            }
        """
        now_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        today_date = datetime.now().strftime("%Y-%m-%d")

        article_rows: List[List[Any]] = []
        vocab_rows: List[List[Any]] = []
        fact_rows: List[List[Any]] = []

        vocab_cards_list: List[Dict[str, Any]] = []
        fact_cards_list: List[Dict[str, Any]] = []

        ocr_snippet = (raw_ocr_text[:300] + "...") if len(raw_ocr_text) > 300 else raw_ocr_text

        for idx, art in enumerate(articles):
            # Unique deterministic article identifier
            art_id = f"ART_{datetime.now().strftime('%Y%m%d')}_{uuid.uuid4().hex[:6].upper()}"
            headline = art.get("headline", "Untitled Article")
            category = art.get("category", "Other")

            # Format key sentences
            key_sents = art.get("key_sentences", [])
            key_sents_str = " \u2022 " + "\n \u2022 ".join(key_sents) if key_sents else ""

            article_rows.append([
                art_id,
                now_ts,
                category,
                headline,
                art.get("ocr_confidence", "Medium"),
                art.get("summary", ""),
                key_sents_str,
                ocr_snippet,
            ])

            # Process Vocabulary Flashcards
            for v_idx, vocab_item in enumerate(art.get("vocabulary", [])):
                v_card_id = f"VOC_{art_id}_{v_idx + 1:02d}"
                # Initial SM-2 parameters
                ease_factor = vocab_item.get("ease_factor", 2.5)
                interval_days = vocab_item.get("interval_days", 1)
                repetitions = vocab_item.get("repetitions", 0)
                next_review = vocab_item.get("next_review_date", today_date)
                last_reviewed = vocab_item.get("last_reviewed", "")

                vocab_record = {
                    "card_id": v_card_id,
                    "timestamp": now_ts,
                    "article_id": art_id,
                    "headline_source": headline,
                    "word": vocab_item.get("word", ""),
                    "meaning": vocab_item.get("meaning", ""),
                    "usage_sentence": vocab_item.get("usage_sentence", ""),
                    "exam_tier": vocab_item.get("exam_tier", "Both"),
                    "ease_factor": ease_factor,
                    "interval_days": interval_days,
                    "repetitions": repetitions,
                    "next_review_date": next_review,
                    "last_reviewed": last_reviewed,
                }
                vocab_cards_list.append(vocab_record)

                vocab_rows.append([
                    v_card_id,
                    now_ts,
                    art_id,
                    headline,
                    vocab_record["word"],
                    vocab_record["meaning"],
                    vocab_record["usage_sentence"],
                    vocab_record["exam_tier"],
                    ease_factor,
                    interval_days,
                    repetitions,
                    next_review,
                    last_reviewed,
                ])

            # Process Current Affairs Flashcards
            for f_idx, fact_item in enumerate(art.get("current_affairs_facts", [])):
                f_card_id = f"FACT_{art_id}_{f_idx + 1:02d}"
                ease_factor = fact_item.get("ease_factor", 2.5)
                interval_days = fact_item.get("interval_days", 1)
                repetitions = fact_item.get("repetitions", 0)
                next_review = fact_item.get("next_review_date", today_date)
                last_reviewed = fact_item.get("last_reviewed", "")

                fact_record = {
                    "card_id": f_card_id,
                    "timestamp": now_ts,
                    "article_id": art_id,
                    "headline_source": headline,
                    "fact": fact_item.get("fact", ""),
                    "likely_relevance": fact_item.get("likely_relevance", ""),
                    "category": category,
                    "ease_factor": ease_factor,
                    "interval_days": interval_days,
                    "repetitions": repetitions,
                    "next_review_date": next_review,
                    "last_reviewed": last_reviewed,
                }
                fact_cards_list.append(fact_record)

                fact_rows.append([
                    f_card_id,
                    now_ts,
                    art_id,
                    headline,
                    fact_record["fact"],
                    fact_record["likely_relevance"],
                    category,
                    ease_factor,
                    interval_days,
                    repetitions,
                    next_review,
                    last_reviewed,
                ])

        # Attempt writing to Google Sheets
        sheets_success = False
        if not self.offline_mode and self.spreadsheet is not None:
            try:
                if article_rows:
                    art_ws = self.spreadsheet.worksheet("Articles")
                    art_ws.append_rows(article_rows, value_input_option="USER_ENTERED")
                    logger.info(f"Appended {len(article_rows)} articles to 'Articles' tab.")

                if vocab_rows:
                    voc_ws = self.spreadsheet.worksheet("Vocabulary")
                    voc_ws.append_rows(vocab_rows, value_input_option="USER_ENTERED")
                    logger.info(f"Appended {len(vocab_rows)} vocabulary flashcards to 'Vocabulary' tab.")

                if fact_rows:
                    fact_ws = self.spreadsheet.worksheet("CurrentAffairs")
                    fact_ws.append_rows(fact_rows, value_input_option="USER_ENTERED")
                    logger.info(f"Appended {len(fact_rows)} facts to 'CurrentAffairs' tab.")

                sheets_success = True
            except Exception as sheet_err:
                logger.error(
                    f"Google Sheets append operation failed: {str(sheet_err)}. "
                    "Creating local backup so no cards are lost."
                )

        # Fallback local backup (always created or created on failure)
        backup_path = None
        if not sheets_success or self.offline_mode:
            os.makedirs(backup_dir, exist_ok=True)
            backup_filename = f"khobor_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            backup_path = os.path.join(backup_dir, backup_filename)
            backup_data = {
                "saved_at": now_ts,
                "articles": articles,
                "raw_ocr_snippet": ocr_snippet,
                "vocabulary_cards": vocab_cards_list,
                "current_affairs_cards": fact_cards_list,
            }
            try:
                with open(backup_path, "w", encoding="utf-8") as f:
                    json.dump(backup_data, f, indent=2, ensure_ascii=False)
                logger.info(f"Saved local backup file: {backup_path}")
            except Exception as backup_err:
                logger.error(f"Failed to write local backup file: {str(backup_err)}")

        return {
            "articles_saved": len(article_rows),
            "vocab_saved": len(vocab_rows),
            "facts_saved": len(fact_rows),
            "sheet_url": self.get_sheet_url() if sheets_success else None,
            "saved_to_sheets": sheets_success,
            "backup_path": backup_path,
            "vocabulary_cards": vocab_cards_list,
            "current_affairs_cards": fact_cards_list,
        }

    def fetch_all_cards_from_sheets(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Read all vocabulary and current affairs cards from Google Sheets.
        Useful for generating Anki decks across multiple uploaded articles.

        Returns
        -------
        Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]
            (vocab_cards, fact_cards)
        """
        vocab_cards: List[Dict[str, Any]] = []
        fact_cards: List[Dict[str, Any]] = []

        if self.offline_mode or self.spreadsheet is None:
            logger.info("Operating offline; unable to fetch from Google Sheets.")
            return vocab_cards, fact_cards

        try:
            voc_ws = self.spreadsheet.worksheet("Vocabulary")
            voc_records = voc_ws.get_all_records()
            vocab_cards = list(voc_records)
            logger.info(f"Retrieved {len(vocab_cards)} vocabulary cards from Google Sheets.")
        except Exception as e:
            logger.warning(f"Could not read Vocabulary tab: {str(e)}")

        try:
            fact_ws = self.spreadsheet.worksheet("CurrentAffairs")
            fact_records = fact_ws.get_all_records()
            fact_cards = list(fact_records)
            logger.info(f"Retrieved {len(fact_cards)} current affairs cards from Google Sheets.")
        except Exception as e:
            logger.warning(f"Could not read CurrentAffairs tab: {str(e)}")

        return vocab_cards, fact_cards
