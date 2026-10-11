"""
Khobor - Web Application Interface (Streamlit)
==============================================
An interactive, mobile-friendly web app for capturing newspaper clippings,
extracting exam content via Gemini AI, syncing with Google Sheets, and
exporting Anki .apkg flashcard decks.

Run locally:
    streamlit run app.py

Run on Colab:
    !streamlit run app.py & npx localtunnel --port 8501
"""

import os
import json
import tempfile
import cv2
import numpy as np
import streamlit as st
from PIL import Image

from ocr_module import OCREngine, ImagePreprocessor
from gemini_extraction import GeminiExtractor
from sheets_storage import SheetsManager
from anki_export import AnkiDeckExporter
from srs_engine import sm2_update, get_initial_srs_state

# Page configuration
st.set_page_config(
    page_title="Khobor - Newspaper to Flashcards",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #0f172a;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    .badge-cds {
        background-color: #dbeafe;
        color: #1e40af;
        padding: 3px 8px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    .badge-gate {
        background-color: #d1fae5;
        color: #065f46;
        padding: 3px 8px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    .card-box {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar Configuration
with st.sidebar:
    st.image("https://img.icons8.com/color/96/newspaper.png", width=64)
    st.title("Khobor Settings")
    st.markdown("**Exam Target**: CDS (UPSC) & GATE General Aptitude")
    
    api_key = st.text_input(
        "Gemini API Key",
        value=os.getenv("GEMINI_API_KEY", ""),
        type="password",
        help="Get a free key from https://aistudio.google.com/",
    )
    
    model_name = st.selectbox(
        "Gemini Model",
        ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash"],
        index=0,
    )
    
    spreadsheet_name = st.text_input("Google Sheet Name", value="Khobor_Exam_Prep")
    deck_prefix = st.text_input("Anki Deck Prefix", value="Readloop")
    
    st.markdown("---")
    st.markdown("### 📚 How to Study")
    st.markdown("1. Snap or upload article photo.")
    st.markdown("2. Check OCR & Gemini extraction.")
    st.markdown("3. Sync to Google Sheets.")
    st.markdown("4. Download `.apkg` and review in Anki.")

# Header
st.markdown('<div class="main-header">📰 Khobor (Readloop) — Article to Flashcard Pipeline</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-Powered Exam Preparation for CDS & GATE: From Physical Newspaper to Spaced-Repetition Decks</div>', unsafe_allow_html=True)

# Tabs
tab_upload, tab_review_srs, tab_guide = st.tabs(["📸 Photo Upload & Pipeline", "🧠 Interactive SM-2 Review", "📖 Exam Register Guide"])

with tab_upload:
    col_input, col_preview = st.columns([1, 1])
    
    with col_input:
        st.subheader("1. Article Input")
        upload_mode = st.radio("Input Method:", ["Upload File / Gallery", "Smartphone Camera", "Use Demo Clipping"], horizontal=True)
        
        image_bytes = None
        if upload_mode == "Upload File / Gallery":
            uploaded_file = st.file_uploader("Choose a newspaper image (JPG, PNG)", type=["jpg", "jpeg", "png"])
            if uploaded_file is not None:
                image_bytes = uploaded_file.read()
                
        elif upload_mode == "Smartphone Camera":
            camera_file = st.camera_input("Take a photo of the newspaper column")
            if camera_file is not None:
                image_bytes = camera_file.read()
                
        else: # Demo Clipping
            if st.button("Generate Synthetic Demo Clipping", type="secondary"):
                # Generate sample clipping
                from pipeline import create_demo_image
                demo_path = create_demo_image("demo_clipping_temp.png")
                with open(demo_path, "rb") as f:
                    image_bytes = f.read()
                st.success("Sample demo clipping generated!")

    # Image Processing Preview
    prep_data = None
    if image_bytes is not None:
        with col_preview:
            st.subheader("2. OpenCV Preprocessing")
            # Load and preprocess
            nparr = np.frombuffer(image_bytes, np.uint8)
            img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            prep_data = ImagePreprocessor.preprocess_pipeline(img_bgr)
            
            p_col1, p_col2 = st.columns(2)
            with p_col1:
                st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), caption="Original Photo", use_container_width=True)
            with p_col2:
                st.image(prep_data["preprocessed_gray"], caption="CLAHE & Deskewed", use_container_width=True)
                
            blur_status = "⚠️ Blurry (Caution)" if prep_data["is_blurry"] else "✅ Sharp / In Focus"
            st.info(f"**Blur Metric**: {prep_data['blur_score']:.1f} ({blur_status}) | **Deskew Correction**: {prep_data['deskew_angle']:.2f}°")

    st.markdown("---")
    
    # Run Full Pipeline Button
    if prep_data is not None:
        if st.button("🚀 Run Full Extraction Pipeline", type="primary", use_container_width=True):
            if not api_key:
                st.error("Please enter a Gemini API Key in the left sidebar to proceed.")
            else:
                with st.spinner("Processing OCR & Running Gemini Exam Analysis..."):
                    try:
                        # 1. OCR
                        ocr_engine = OCREngine()
                        ocr_res = ocr_engine.process_article_image(prep_data["preprocessed_gray"])
                        raw_text = ocr_res["raw_text"]
                        
                        st.session_state["raw_ocr_text"] = raw_text
                        st.session_state["ocr_res"] = ocr_res
                        
                        # 2. Gemini Extraction
                        gemini_extractor = GeminiExtractor(api_key=api_key, model_name=model_name)
                        articles = gemini_extractor.extract(raw_text)
                        st.session_state["extracted_articles"] = articles
                        
                        # 3. Google Sheets & Local Save
                        sheets_mgr = SheetsManager(spreadsheet_name=spreadsheet_name, offline_mode=False)
                        storage_summary = sheets_mgr.save_pipeline_output(
                            articles=articles,
                            raw_ocr_text=raw_text,
                            backup_dir="./backups",
                        )
                        st.session_state["storage_summary"] = storage_summary
                        
                        # 4. Anki Export
                        anki_exporter = AnkiDeckExporter(registry_path="./anki_registry.json", deck_prefix=deck_prefix)
                        anki_summary = anki_exporter.export_decks(
                            vocab_items=storage_summary["vocabulary_cards"],
                            fact_items=storage_summary["current_affairs_cards"],
                            output_dir="./anki_decks",
                        )
                        st.session_state["anki_summary"] = anki_summary
                        
                        st.success("🎉 Pipeline executed successfully!")
                        
                    except Exception as err:
                        st.error(f"Pipeline error: {str(err)}")

    # Display Results if available
    if "extracted_articles" in st.session_state and st.session_state["extracted_articles"]:
        articles = st.session_state["extracted_articles"]
        storage = st.session_state.get("storage_summary", {})
        anki = st.session_state.get("anki_summary", {})
        
        st.markdown("## 📊 Extracted Content & Decks")
        
        # Download deck buttons
        if anki:
            dl_col1, dl_col2 = st.columns(2)
            with dl_col1:
                vocab_path = anki.get("vocab_deck_path")
                if vocab_path and os.path.exists(vocab_path):
                    with open(vocab_path, "rb") as f:
                        st.download_button(
                            label=f"📥 Download {deck_prefix} - Vocabulary.apkg ({anki.get('vocab_cards_added', 0)} cards)",
                            data=f.read(),
                            file_name=os.path.basename(vocab_path),
                            mime="application/octet-stream",
                            use_container_width=True,
                        )
            with dl_col2:
                facts_path = anki.get("facts_deck_path")
                if facts_path and os.path.exists(facts_path):
                    with open(facts_path, "rb") as f:
                        st.download_button(
                            label=f"📥 Download {deck_prefix} - Current Affairs.apkg ({anki.get('facts_cards_added', 0)} cards)",
                            data=f.read(),
                            file_name=os.path.basename(facts_path),
                            mime="application/octet-stream",
                            use_container_width=True,
                        )
                        
        if storage.get("sheet_url"):
            st.markdown(f"🔗 **[Open Synced Google Sheet Database]({storage['sheet_url']})**")
            
        # Display each article
        for idx, art in enumerate(articles, 1):
            with st.expander(f"📑 Article #{idx}: {art.get('headline')}", expanded=True):
                st.markdown(f"**Category**: `{art.get('category')}` | **Confidence**: `{art.get('ocr_confidence')}`")
                st.markdown(f"**Summary**: {art.get('summary')}")
                
                # Vocabulary cards
                vocabs = art.get("vocabulary", [])
                if vocabs:
                    st.markdown("#### 📖 High-Yield Vocabulary")
                    v_cols = st.columns(len(vocabs) if len(vocabs) <= 3 else 3)
                    for v_idx, v in enumerate(vocabs):
                        c_slot = v_cols[v_idx % len(v_cols)]
                        with c_slot:
                            tier_class = "badge-cds" if v.get("exam_tier") == "CDS" else "badge-gate"
                            st.markdown(
                                f"""
                                <div class="card-box">
                                  <span class="{tier_class}">{v.get('exam_tier')}</span>
                                  <h4 style="margin: 6px 0;">{v.get('word')}</h4>
                                  <p style="font-size: 0.9rem; color: #1e293b;">{v.get('meaning')}</p>
                                  <small style="color: #64748b; font-style: italic;">"{v.get('usage_sentence')}"</small>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
                            
                # Current affairs
                facts = art.get("current_affairs_facts", [])
                if facts:
                    st.markdown("#### 🌐 Current Affairs Facts")
                    for f in facts:
                        st.info(f"**Fact**: {f.get('fact')}\n\n*Exam Relevance*: {f.get('likely_relevance')}")

# Tab 2: Interactive SM-2 Review
with tab_review_srs:
    st.subheader("Interactive SM-2 Review Simulator")
    st.write("Test your recall directly in the browser using the SuperMemo-2 spaced-repetition algorithm.")
    
    if "storage_summary" in st.session_state and st.session_state["storage_summary"].get("vocabulary_cards"):
        cards = st.session_state["storage_summary"]["vocabulary_cards"]
        sel_card_idx = st.selectbox(
            "Select Card to Review:",
            range(len(cards)),
            format_func=lambda i: f"{cards[i]['word']} ({cards[i]['exam_tier']})"
        )
        card = cards[sel_card_idx]
        
        st.markdown(f"### Word: **{card['word']}**")
        if st.checkbox("Reveal Answer / Meaning"):
            st.markdown(f"**Meaning**: {card['meaning']}")
            st.markdown(f"**Usage Sentence**: *{card['usage_sentence']}*")
            st.markdown(f"**Source**: {card['headline_source']}")
            
            st.markdown("---")
            st.markdown("#### Rate your recall (0 to 5):")
            rating_cols = st.columns(6)
            ratings = [
                (0, "0: Blackout"),
                (1, "1: Vague"),
                (2, "2: Incorrect"),
                (3, "3: Hard Pass"),
                (4, "4: Good"),
                (5, "5: Perfect"),
            ]
            for r_val, r_label in ratings:
                if rating_cols[r_val].button(r_label, key=f"rate_{r_val}"):
                    new_ef, new_int, new_reps = sm2_update(
                        card["ease_factor"],
                        card["interval_days"],
                        card["repetitions"],
                        quality=r_val,
                    )
                    st.success(
                        f"Updated SM-2 schedule! Repetitions: {new_reps} | Next Interval: {new_int} day(s) | Ease Factor: {new_ef:.2f}"
                    )
    else:
        st.info("Extract an article in Tab 1 first to practice cards here.")

# Tab 3: Guide
with tab_guide:
    st.subheader("CDS & GATE Exam Preparation Register Guide")
    st.markdown(
        """
        ### 🎖️ CDS (UPSC English Register)
        - **Question Style**: Reading Comprehension, Synonyms/Antonyms, Idiomatic Usage, Sentence Correction.
        - **Vocabulary Register**: Formal, diplomatic, geopolitical, literary (e.g., *bellicose, laconic, status quo, hegemony, multilateral*).
        - **Current Affairs Focus**: Bilateral defence agreements, joint military exercises, indigenisation percentages, constitutional bodies.

        ### ⚙️ GATE General Aptitude (Engineering Register)
        - **Question Style**: Verbal Ability, Logical Deductions, Sentence Equivalence, Analogies.
        - **Vocabulary Register**: Precise, analytical, structural, objective (e.g., *corroborate, lucid, pragmatic, dichotomy, catalyst, indispensable*).
        - **Current Affairs Focus**: Scientific infrastructure, quantum computing, energy transition, data protection and regulatory policies.
        """
    )
