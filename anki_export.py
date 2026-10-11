"""
Khobor - Anki Flashcard Export Module
=====================================
Generates polished, spaced-repetition Anki decks (.apkg files) using `genanki`.

Features:
- Two dedicated decks:
  1. "Readloop - Vocabulary" (Word -> Meaning, Usage Sentence, CDS/GATE Exam Tier)
  2. "Readloop - Current Affairs" (Fact -> Context, Relevance, Category)
- Stable IDs for Decks and Models to ensure smooth, non-conflicting updates in Anki.
- Deduplication engine: tracks exported card IDs and normalized front text so repeated
  runs never duplicate existing flashcards.
- Deterministic GUID generation (via `genanki.guid_for(front_text)`) so Anki Desktop/AnkiDroid
  automatically recognizes previously imported cards on import.
- Responsive, modern card styling with clean CSS tags and typography tailored for
  serious competitive exam revision.
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional, Set, Tuple

logger = logging.getLogger("Khobor.Anki")

try:
    import genanki
    HAS_GENANKI = True
except ImportError:
    genanki = None
    HAS_GENANKI = False
    logger.warning("genanki is not installed. Install via: pip install genanki")


# ==============================================================================
# STABLE IDENTIFIERS (Deterministic 32-bit Integers for Anki)
# ==============================================================================
# Using fixed IDs ensures Anki modifies existing decks/models rather than
# spawning duplicate entries on subsequent imports.
VOCAB_MODEL_ID = 1607392319
FACTS_MODEL_ID = 1607392320

VOCAB_DECK_ID = 2059400110
FACTS_DECK_ID = 2059400111


# ==============================================================================
# CARD STYLING (CSS)
# Clean, modern, high-contrast design optimized for mobile and desktop review.
# ==============================================================================
ANKI_CARD_CSS = """
.card {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  font-size: 19px;
  text-align: center;
  color: #1e293b;
  background-color: #f8fafc;
  padding: 24px 18px;
  line-height: 1.5;
}

.card-container {
  max-width: 540px;
  margin: 0 auto;
  background: #ffffff;
  border-radius: 14px;
  padding: 28px 24px;
  box-shadow: 0 4px 14px rgba(0, 0, 0, 0.06);
  border: 1px solid #e2e8f0;
}

.badge-row {
  display: flex;
  justify-content: center;
  gap: 8px;
  margin-bottom: 16px;
  flex-wrap: wrap;
}

.badge {
  display: inline-block;
  font-size: 11px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  padding: 4px 10px;
  border-radius: 9999px;
}

.badge-cds {
  background-color: #dbeafe;
  color: #1e40af;
  border: 1px solid #bfdbfe;
}

.badge-gate {
  background-color: #d1fae5;
  color: #065f46;
  border: 1px solid #a7f3d0;
}

.badge-both {
  background-color: #ede9fe;
  color: #5b21b6;
  border: 1px solid #ddd6fe;
}

.badge-category {
  background-color: #f1f5f9;
  color: #475569;
  border: 1px solid #cbd5e1;
}

.card-title {
  font-size: 28px;
  font-weight: 800;
  color: #0f172a;
  letter-spacing: -0.02em;
  margin: 8px 0 14px 0;
}

.card-prompt {
  font-size: 19px;
  font-weight: 600;
  color: #334155;
  line-height: 1.45;
  margin: 8px 0 12px 0;
}

.divider {
  height: 1px;
  background: #e2e8f0;
  margin: 20px 0;
  border: none;
}

.section-label {
  font-size: 12px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: #64748b;
  margin-bottom: 4px;
}

.meaning-text {
  font-size: 20px;
  font-weight: 600;
  color: #0f172a;
  margin-bottom: 16px;
}

.example-box {
  background-color: #f8fafc;
  border-left: 4px solid #3b82f6;
  padding: 12px 16px;
  border-radius: 0 8px 8px 0;
  text-align: left;
  font-size: 15px;
  color: #334155;
  font-style: italic;
  margin: 14px 0;
}

.source-footer {
  font-size: 12px;
  color: #94a3b8;
  margin-top: 18px;
  text-align: center;
}
"""


# ==============================================================================
# VOCABULARY CARD TEMPLATES
# ==============================================================================
VOCAB_FRONT_TEMPLATE = """
<div class="card-container">
  <div class="badge-row">
    {{#ExamTier}}
      <span class="badge {{TierClass}}">{{ExamTier}}</span>
    {{/ExamTier}}
  </div>
  <div class="card-title">{{Word}}</div>
  <div style="font-size: 13px; color: #64748b;">(Recall meaning, exam register & usage)</div>
</div>
"""

VOCAB_BACK_TEMPLATE = """
<div class="card-container">
  <div class="badge-row">
    {{#ExamTier}}
      <span class="badge {{TierClass}}">{{ExamTier}}</span>
    {{/ExamTier}}
  </div>

  <div class="card-title">{{Word}}</div>

  <hr class="divider">

  <div class="section-label">Definition / Meaning</div>
  <div class="meaning-text">{{Meaning}}</div>

  {{#UsageSentence}}
    <div class="section-label" style="text-align: left; margin-top: 12px;">Exam Context / Usage</div>
    <div class="example-box">
      &ldquo;{{UsageSentence}}&rdquo;
    </div>
  {{/UsageSentence}}

  {{#HeadlineSource}}
    <div class="source-footer">
      Source Article: <strong>{{HeadlineSource}}</strong>
    </div>
  {{/HeadlineSource}}
</div>
"""


# ==============================================================================
# CURRENT AFFAIRS CARD TEMPLATES
# ==============================================================================
FACTS_FRONT_TEMPLATE = """
<div class="card-container">
  <div class="badge-row">
    {{#Category}}
      <span class="badge badge-category">{{Category}}</span>
    {{/Category}}
    <span class="badge badge-cds">Current Affairs</span>
  </div>
  <div class="section-label" style="margin-top: 10px;">Fact Verification</div>
  <div class="card-prompt">{{Fact}}</div>
  <div style="font-size: 13px; color: #64748b; margin-top: 14px;">(Recall key facts, background & exam relevance)</div>
</div>
"""

FACTS_BACK_TEMPLATE = """
<div class="card-container">
  <div class="badge-row">
    {{#Category}}
      <span class="badge badge-category">{{Category}}</span>
    {{/Category}}
    <span class="badge badge-cds">Current Affairs</span>
  </div>

  <div class="card-prompt" style="color: #0f172a; font-weight: 700;">{{Fact}}</div>

  <hr class="divider">

  {{#LikelyRelevance}}
    <div class="section-label">Exam Relevance (CDS / GATE-Aptitude)</div>
    <div class="example-box" style="border-left-color: #10b981; font-style: normal;">
      {{LikelyRelevance}}
    </div>
  {{/LikelyRelevance}}

  {{#HeadlineSource}}
    <div class="source-footer">
      Source Article: <strong>{{HeadlineSource}}</strong>
    </div>
  {{/HeadlineSource}}
</div>
"""


def get_vocab_model() -> Any:
    """Create or retrieve the genanki model for vocabulary flashcards."""
    return genanki.Model(
        VOCAB_MODEL_ID,
        "Khobor Vocabulary Model",
        fields=[
            {"name": "Word"},
            {"name": "Meaning"},
            {"name": "UsageSentence"},
            {"name": "ExamTier"},
            {"name": "TierClass"},
            {"name": "HeadlineSource"},
            {"name": "CardID"},
        ],
        templates=[
            {
                "name": "Khobor Vocab Card",
                "qfmt": VOCAB_FRONT_TEMPLATE,
                "afmt": VOCAB_BACK_TEMPLATE,
            }
        ],
        css=ANKI_CARD_CSS,
    )


def get_facts_model() -> Any:
    """Create or retrieve the genanki model for current affairs flashcards."""
    return genanki.Model(
        FACTS_MODEL_ID,
        "Khobor Current Affairs Model",
        fields=[
            {"name": "Fact"},
            {"name": "LikelyRelevance"},
            {"name": "Category"},
            {"name": "HeadlineSource"},
            {"name": "CardID"},
        ],
        templates=[
            {
                "name": "Khobor Fact Card",
                "qfmt": FACTS_FRONT_TEMPLATE,
                "afmt": FACTS_BACK_TEMPLATE,
            }
        ],
        css=ANKI_CARD_CSS,
    )


class AnkiDeckExporter:
    """
    Exports vocabulary and current affairs items to Anki .apkg packages with
    built-in deduplication.
    """

    def __init__(
        self,
        registry_path: str = "./anki_registry.json",
        deck_prefix: str = "Readloop",
    ):
        """
        Initialize the Anki exporter.

        Parameters
        ----------
        registry_path : str
            Path to tracking JSON file storing set of previously exported card IDs / front texts.
        deck_prefix : str
            Prefix for deck names (default "Readloop" as specified, e.g. "Readloop - Vocabulary").
        """
        self.registry_path = registry_path
        self.deck_prefix = deck_prefix
        self.exported_ids: Set[str] = set()
        self.exported_fronts: Set[str] = set()
        self._load_registry()

    def _load_registry(self) -> None:
        """Load exported IDs from disk to prevent duplicates across sessions."""
        if os.path.exists(self.registry_path):
            try:
                with open(self.registry_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.exported_ids = set(data.get("card_ids", []))
                    self.exported_fronts = set(data.get("front_texts", []))
                logger.info(f"Loaded {len(self.exported_ids)} previously exported card IDs from registry.")
            except Exception as e:
                logger.warning(f"Could not load Anki registry ({str(e)}); initializing fresh.")
                self.exported_ids = set()
                self.exported_fronts = set()

    def _save_registry(self) -> None:
        """Persist exported card IDs to disk."""
        try:
            os.makedirs(os.path.dirname(os.path.abspath(self.registry_path)), exist_ok=True)
            with open(self.registry_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "card_ids": sorted(list(self.exported_ids)),
                        "front_texts": sorted(list(self.exported_fronts)),
                    },
                    f,
                    indent=2,
                )
        except Exception as e:
            logger.warning(f"Could not save Anki registry: {str(e)}")

    @staticmethod
    def _tier_to_css_class(tier_str: str) -> str:
        """Map exam tier string to CSS class."""
        t = tier_str.lower()
        if "cds" in t and "gate" in t:
            return "badge-both"
        if "cds" in t:
            return "badge-cds"
        if "gate" in t:
            return "badge-gate"
        return "badge-both"

    def export_decks(
        self,
        vocab_items: List[Dict[str, Any]],
        fact_items: List[Dict[str, Any]],
        output_dir: str = "./anki_decks",
        reset_registry: bool = False,
    ) -> Dict[str, Any]:
        """
        Generate and save Anki .apkg deck files for Vocabulary and Current Affairs.
        Deduplicates against already-exported cards.

        Parameters
        ----------
        vocab_items : List[Dict[str, Any]]
            List of vocabulary card records.
        fact_items : List[Dict[str, Any]]
            List of current affairs card records.
        output_dir : str
            Directory where .apkg files will be saved.
        reset_registry : bool
            If True, clears the duplicate cache and re-exports everything.

        Returns
        -------
        Dict[str, Any]
            Export summary containing file paths and card counts:
            {
                "vocab_deck_path": str,
                "facts_deck_path": str,
                "vocab_cards_added": int,
                "vocab_cards_skipped": int,
                "facts_cards_added": int,
                "facts_cards_skipped": int
            }
        """
        if not HAS_GENANKI:
            raise RuntimeError("genanki is required to build Anki decks. Run: pip install genanki")

        if reset_registry:
            self.exported_ids.clear()
            self.exported_fronts.clear()

        os.makedirs(output_dir, exist_ok=True)

        vocab_model = get_vocab_model()
        facts_model = get_facts_model()

        # Deck 1: "Readloop - Vocabulary" (or custom prefix)
        vocab_deck_title = f"{self.deck_prefix} - Vocabulary"
        vocab_deck = genanki.Deck(VOCAB_DECK_ID, vocab_deck_title)

        # Deck 2: "Readloop - Current Affairs" (or custom prefix)
        facts_deck_title = f"{self.deck_prefix} - Current Affairs"
        facts_deck = genanki.Deck(FACTS_DECK_ID, facts_deck_title)

        vocab_added = 0
        vocab_skipped = 0
        facts_added = 0
        facts_skipped = 0

        # --- Process Vocabulary Cards ---
        for item in vocab_items:
            word = str(item.get("word", "")).strip()
            meaning = str(item.get("meaning", "")).strip()
            usage = str(item.get("usage_sentence", "")).strip()
            tier = str(item.get("exam_tier", "Both")).strip()
            headline = str(item.get("headline_source", "")).strip()
            card_id = str(item.get("card_id", "")).strip()

            if not word or not meaning:
                continue

            # Deduplication key check
            norm_front = word.lower()
            if (card_id and card_id in self.exported_ids) or (norm_front in self.exported_fronts):
                vocab_skipped += 1
                continue

            tier_class = self._tier_to_css_class(tier)
            # Deterministic GUID based on word ensures Anki updates note if re-imported
            guid = genanki.guid_for(f"vocab_{word.lower()}")

            note = genanki.Note(
                model=vocab_model,
                fields=[
                    word,
                    meaning,
                    usage,
                    tier,
                    tier_class,
                    headline,
                    card_id,
                ],
                guid=guid,
                tags=["khobor", "readloop", "vocabulary", tier.lower().replace(" ", "-")],
            )
            vocab_deck.add_note(note)
            vocab_added += 1

            if card_id:
                self.exported_ids.add(card_id)
            self.exported_fronts.add(norm_front)

        # --- Process Current Affairs Cards ---
        for item in fact_items:
            fact = str(item.get("fact", "")).strip()
            relevance = str(item.get("likely_relevance", "")).strip()
            category = str(item.get("category", "General")).strip()
            headline = str(item.get("headline_source", "")).strip()
            card_id = str(item.get("card_id", "")).strip()

            if not fact:
                continue

            # Deduplication key check
            norm_front = fact.lower()
            if (card_id and card_id in self.exported_ids) or (norm_front in self.exported_fronts):
                facts_skipped += 1
                continue

            guid = genanki.guid_for(f"fact_{fact.lower()[:60]}")

            note = genanki.Note(
                model=facts_model,
                fields=[
                    fact,
                    relevance,
                    category,
                    headline,
                    card_id,
                ],
                guid=guid,
                tags=["khobor", "readloop", "current-affairs", category.lower().replace(" ", "-")],
            )
            facts_deck.add_note(note)
            facts_added += 1

            if card_id:
                self.exported_ids.add(card_id)
            self.exported_fronts.add(norm_front)

        # Write Decks to disk
        prefix_slug = self.deck_prefix.replace(" ", "_")
        vocab_path = os.path.join(output_dir, f"{prefix_slug}_Vocabulary.apkg")
        facts_path = os.path.join(output_dir, f"{prefix_slug}_Current_Affairs.apkg")

        # Write Vocabulary Deck
        genanki.Package(vocab_deck).write_to_file(vocab_path)
        logger.info(f"Saved Vocabulary Anki deck to: {vocab_path} ({vocab_added} new cards)")

        # Write Current Affairs Deck
        genanki.Package(facts_deck).write_to_file(facts_path)
        logger.info(f"Saved Current Affairs Anki deck to: {facts_path} ({facts_added} new cards)")

        # Save deduplication registry
        self._save_registry()

        return {
            "vocab_deck_path": vocab_path,
            "facts_deck_path": facts_path,
            "vocab_cards_added": vocab_added,
            "vocab_cards_skipped": vocab_skipped,
            "facts_cards_added": facts_added,
            "facts_cards_skipped": facts_skipped,
            "total_vocab_cards_in_deck": len(vocab_deck.notes),
            "total_facts_cards_in_deck": len(facts_deck.notes),
        }
