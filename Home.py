# src/app/Home.py

"""
SpiritedData: Emotional Landscape of Studio Ghibli

Main entry point for Streamlit multi-page application.
Emotion Analysis Showcase Landing Page.
"""

import streamlit as st
from utils.config import APP_TITLE, THEME
from utils.data_loader import (
    get_director_comparison,
    get_film_emotion_timeseries_by_title,
    get_film_list,
    get_hero_stats,
    get_top_fearful_film,
    get_top_joyful_film,
)
from utils.theme import apply_custom_css, render_footer, render_glass_card, render_header
from utils.visualization import plot_emotion_preview

# Page config (must be first Streamlit command)
st.set_page_config(
    page_title=APP_TITLE, page_icon="🎬", layout="wide", initial_sidebar_state="expanded"
)

# Apply "Spirit World" Design System
apply_custom_css()

# ============================================================================
# Hero Section
# ============================================================================

render_header(
    "SpiritedData",
    "Exploring the emotional heartbeat of Studio Ghibli films",
)

# Introduction paragraph with gradient background (Echoes-style)
st.markdown(
    """
    <div style="background: linear-gradient(90deg, rgba(56, 189, 248, 0.1), rgba(245, 158, 11, 0.1));
                padding: 1.5rem; border-radius: 12px; margin: 1rem 0 2rem 0;">
        <p style="color: #F1F5F9; margin: 0; font-size: 1.1rem; line-height: 1.7;">
            This project decodes the <b>emotional DNA</b> of Studio Ghibli's filmography using 
            <b>multilingual NLP</b> and <b>signal processing</b>. Every line of dialogue across 
            <b>22 films</b> and <b>5 languages</b> has been analyzed through a 28-dimension 
            emotion classifier to reveal patterns invisible to the naked eye.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================================
# Data Stats Dashboard
# ============================================================================

st.markdown("### 📊 The Ghibli Archive")

# Load actual stats from database
with st.spinner("Loading statistics..."):
    stats = get_hero_stats()

col1, col2, col3, col4 = st.columns(4)

with col1:
    render_glass_card("Films Analyzed", str(stats["film_count"]), "Complete filmography", icon="🎬")

with col2:
    render_glass_card(
        "Emotion Data Points",
        f"{stats['emotion_data_points']:,}",
        "Minute-level emotion tracking",
        icon="📈",
    )

with col3:
    render_glass_card(
        "Languages", str(stats["languages_count"]), "EN • FR • ES • NL • AR", icon="🌍"
    )

with col4:
    render_glass_card(
        "Dialogue Entries", f"{stats['dialogue_entries']:,}", "Parsed subtitle lines", icon="💬"
    )

st.divider()

# ============================================================================
# Quick Insights Section
# ============================================================================

st.markdown("### ✨ Discover the Data")

# Load insights data
with st.spinner("Computing insights..."):
    joyful = get_top_joyful_film()
    fearful = get_top_fearful_film()
    directors = get_director_comparison()

col_i1, col_i2, col_i3 = st.columns(3)

with col_i1:
    st.markdown(
        f"""
    <div class="glass-card">
        <h4 style="color: {THEME['accent_color']};">😊 Most Joyful Film</h4>
        <p style="font-size: 18px; font-weight: bold; margin: 10px 0;">{joyful['film_title']}</p>
        <p style="color: #94A3B8; font-size: 14px;">Joy Score: {joyful['joy_score']:.3f}</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

with col_i2:
    st.markdown(
        f"""
    <div class="glass-card">
        <h4 style="color: {THEME['primary_color']};">😨 Most Fearful Film</h4>
        <p style="font-size: 18px; font-weight: bold; margin: 10px 0;">{fearful['film_title']}</p>
        <p style="color: #94A3B8; font-size: 14px;">Fear Score: {fearful['fear_score']:.3f}</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

with col_i3:
    miyazaki = directors.get("miyazaki", {})
    takahata = directors.get("takahata", {})

    st.markdown(
        f"""
    <div class="glass-card">
        <h4 style="color: {THEME['primary_color']}; margin-bottom: 15px;">🎭 Director Styles</h4>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">
            <div style="border-right: 1px solid rgba(255, 255, 255, 0.1); padding-right: 8px;">
                <p style="font-size: 13px; font-weight: bold; margin: 0 0 8px 0;">Miyazaki</p>
                <p style="font-size: 11px; color: {THEME['accent_color']}; margin: 0 0 8px 0;">{miyazaki.get('film_count', 0)} films</p>
                <p style="font-size: 10px; color: #94A3B8; margin: 0;">{miyazaki.get('style_label', 'Unknown')}</p>
            </div>
            <div style="padding-left: 8px;">
                <p style="font-size: 13px; font-weight: bold; margin: 0 0 8px 0;">Takahata</p>
                <p style="font-size: 11px; color: {THEME['accent_color']}; margin: 0 0 8px 0;">{takahata.get('film_count', 0)} films</p>
                <p style="font-size: 10px; color: #94A3B8; margin: 0;">{takahata.get('style_label', 'Unknown')}</p>
            </div>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

st.divider()

# ============================================================================
# Emotion Journey Explorer
# ============================================================================

st.markdown("### 🎞️ Emotion Journey Preview")

st.markdown(
    """
    <p style="color: #94A3B8; margin-bottom: 1rem;">
        Select a film to preview its emotional arc. Positive emotions appear above the zero line, 
        negative emotions below — revealing the emotional rhythm of Ghibli storytelling.
    </p>
    """,
    unsafe_allow_html=True,
)

# Load film list
films_df = get_film_list()
film_titles = films_df["title"].tolist()

col_sel1, col_sel2 = st.columns([3, 1])

with col_sel1:
    # Default to Princess Kaguya if available
    default_index = 0
    for i, title in enumerate(film_titles):
        if "Princess Kaguya" in title or "Kaguya" in title:
            default_index = i
            break

    selected_film = st.selectbox(
        "Choose a Film",
        film_titles,
        index=default_index,
        key="film_selector",
    )

with col_sel2:
    selected_language = st.selectbox(
        "Language",
        ["en", "fr", "es", "nl", "ar"],
        format_func=lambda x: {
            "en": "English",
            "fr": "French",
            "es": "Spanish",
            "nl": "Dutch",
            "ar": "Arabic",
        }[x],
        key="language_selector",
    )

# Load and plot emotion timeline
if selected_film:
    with st.spinner(f"Loading emotion timeline for {selected_film}..."):
        emotion_df = get_film_emotion_timeseries_by_title(selected_film, selected_language)

    if not emotion_df.empty:
        fig = plot_emotion_preview(emotion_df, selected_film, selected_language)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True})

        st.caption(
            "📈 **Reading the chart:** Lines above zero represent positive emotions (joy, love, excitement). "
            "Lines below zero represent negative emotions (fear, sadness, anger). "
            "The interplay creates the film's emotional signature."
        )

        # Call-to-action
        col_cta1, col_cta2, col_cta3 = st.columns([1, 2, 1])
        with col_cta2:
            if st.button("🎬 Explore Full Analysis →", use_container_width=True, type="primary"):
                st.switch_page("pages/1_🎬_The_Spirit_Archives.py")
    else:
        st.warning(f"No emotion data found for {selected_film} in {selected_language.upper()}")

st.divider()

# ============================================================================
# Navigation Cards (Updated order: Echoes before Architects)
# ============================================================================

st.markdown("### 🌟 Explore the Spirit World")

st.markdown(
    """
    <p style="color: #94A3B8; margin-bottom: 1.5rem;">
        Each page offers a different lens into Studio Ghibli's emotional universe.
    </p>
    """,
    unsafe_allow_html=True,
)

col_nav1, col_nav2, col_nav3 = st.columns(3)

with col_nav1:
    st.markdown(
        f"""
    <div class="glass-card" style="min-height: 180px;">
        <h4 style="color: {THEME['primary_color']};">🎬 The Spirit Archives</h4>
        <p style="color: #94A3B8; font-size: 14px; line-height: 1.5;">
            Dive deep into any film's emotional journey. Visualize timelines,
            explore peak moments, and compare emotional fingerprints across languages.
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )
    if st.button("Explore Archives →", key="nav_archives", use_container_width=True):
        st.switch_page("pages/1_🎬_The_Spirit_Archives.py")

    st.markdown(
        f"""
    <div class="glass-card" style="min-height: 180px;">
        <h4 style="color: {THEME['primary_color']};">📊 The Alchemy of Data</h4>
        <p style="color: #94A3B8; font-size: 14px; line-height: 1.5;">
            Peek behind the curtain. Understand the signal processing,
            data quality metrics, and methodology that powers the analysis.
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )
    if st.button("View Methodology →", key="nav_alchemy", use_container_width=True):
        st.switch_page("pages/4_📊_The_Alchemy_of_Data.py")

with col_nav2:
    st.markdown(
        f"""
    <div class="glass-card" style="min-height: 180px;">
        <h4 style="color: {THEME['accent_color']};">🌍 Echoes Across Languages</h4>
        <p style="color: #94A3B8; font-size: 14px; line-height: 1.5;">
            How does emotion translate? Discover how the same scenes evoke
            different emotional patterns across English, French, Spanish, Dutch, and Arabic.
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )
    if st.button("Compare Languages →", key="nav_echoes", use_container_width=True):
        st.switch_page("pages/2_🌍_Echoes_Across_Languages.py")

    st.markdown(
        f"""
    <div class="glass-card" style="min-height: 180px;">
        <h4 style="color: {THEME['primary_color']};">🧠 Memories of Sora</h4>
        <p style="color: #94A3B8; font-size: 14px; line-height: 1.5;">
            Meet Sora, the AI archivist (currently in retirement). A reflection on
            what worked — and what didn't — in building conversational AI.
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )
    if st.button("Meet Sora →", key="nav_sora", use_container_width=True):
        st.switch_page("pages/5_🧠_Memories_of_Sora.py")

with col_nav3:
    st.markdown(
        f"""
    <div class="glass-card" style="min-height: 180px;">
        <h4 style="color: {THEME['primary_color']};">🎭 Architects of Emotion</h4>
        <p style="color: #94A3B8; font-size: 14px; line-height: 1.5;">
            Miyazaki vs. Takahata: a study in contrasts. Compare directors'
            emotional signatures and trace how their styles evolved over decades.
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )
    if st.button("Compare Directors →", key="nav_architects", use_container_width=True):
        st.switch_page("pages/3_🎭_Architects_of_Emotion.py")

st.divider()

# ============================================================================
# Methodology Expander
# ============================================================================

with st.expander("📚 How does this work?"):
    st.markdown(
        """
        ### The Analysis Pipeline

        **1. Subtitle Collection & Parsing**
        - Subtitles collected for 22 Studio Ghibli films across 5 languages
        - Each subtitle file parsed into timestamped dialogue entries
        - Dialogue cleaned and normalized for analysis

        **2. Emotion Classification**
        - Each dialogue line processed through a multilingual transformer model
        - The model classifies text into 28 emotion dimensions (based on Google's GoEmotions research)
        - Outputs probability scores for each emotion category

        **3. Signal Processing**
        - Raw emotion scores aggregated into minute-level buckets
        - 10-minute rolling average applied for smoothed visualization
        - Peak detection identifies emotionally intense moments

        **4. Cross-Language Analysis**
        - Same scenes compared across different language subtitles
        - Percent difference calculated to identify translation biases
        - Consistency scores measure how similarly emotions translate

        **Why This Matters**
        
        Traditional film analysis relies on human interpretation. This project offers a 
        complementary data-driven perspective — revealing patterns that emerge from the 
        aggregate emotional content of dialogue, rather than individual scene interpretation.
        
        The results aren't "ground truth" — they're a lens for exploration and discussion.
        """
    )

# Footer
render_footer()

# Sidebar info
with st.sidebar:
    st.markdown("<br><br>", unsafe_allow_html=True)

    st.markdown("### About")

    st.markdown(
        """
    **SpiritedData** is an emotion analysis engine showcasing modern Data Engineering 
    practices. Built with Python, dbt, DuckDB, and Streamlit.

    While this demo analyzes Studio Ghibli films, the underlying engine is flexible. 
    You can adapt it to your own datasets via the 
    [GitHub Repo](https://github.com/edjunior/ghibli_pipeline).

    *(If you enjoy the analysis, a star on the repo is greatly appreciated!)*

    ---

    **Note:** Analysis results vary by subtitle version. 
    See "The Alchemy of Data" for methodology details.
    """
    )
