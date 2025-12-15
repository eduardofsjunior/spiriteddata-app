"""
The Spirit Archives - Film Explorer Page

Interactive emotion analysis for Studio Ghibli films across languages.
Features: Film selector, emotion timeline, composition chart,
emotional fingerprint radar chart, smoothed vs raw toggle, CSV export.
"""

import logging
import re
import sys
from pathlib import Path

import streamlit as st

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add parent directory to path for utils imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.config import THEME
from utils.data_loader import (
    compute_peaks_from_timeseries,
    get_film_emotion_summary,
    get_film_emotion_timeseries,
    get_film_list_with_metadata,
    get_peak_dialogues,
    get_raw_emotion_peaks,
    get_validation_status,
)
from utils.data_quality import render_data_quality_warning
from utils.theme import apply_custom_css, render_header
from utils.visualization import (
    get_top_n_emotions,
    plot_emotion_composition,
    plot_emotion_timeline,
    plot_emotional_fingerprint,
)

# Page configuration
st.set_page_config(page_title="The Spirit Archives", page_icon="🎬", layout="wide")
apply_custom_css()

# ============================================================================
# Page Header
# ============================================================================

render_header(
    "🎬 The Spirit Archives",
    "Explore the emotional heartbeat of Studio Ghibli films",
)

# Introduction paragraph with gradient background
st.markdown(
    """
    <div style="background: linear-gradient(90deg, rgba(56, 189, 248, 0.1), rgba(245, 158, 11, 0.1));
                padding: 1rem; border-radius: 8px; margin-bottom: 1.5rem;">
        <p style="color: #F1F5F9; margin: 0;">
            Select any film-language combination to visualize its <b>emotional journey</b>. 
            Watch how joy, fear, sadness, and love weave through the narrative — 
            minute by minute, scene by scene.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================================
# Methodology Expander
# ============================================================================

with st.expander("📚 How to read these visualizations"):
    st.markdown(
        """
        ### Understanding the Charts
        
        **Emotion Timeline**
        - X-axis: Film runtime in minutes
        - Y-axis: Emotion intensity (positive emotions above zero, negative below)
        - ⭐ Star markers indicate peak emotional moments
        - Hover over lines to see exact values at any point
        
        **Emotion Composition**
        - Stacked view showing all emotions simultaneously
        - Upper half: Positive emotions (joy, love, excitement, etc.)
        - Lower half: Negative emotions (fear, sadness, anger, etc.)
        - The overall "thickness" shows emotional density
        
        **Emotional Fingerprint**
        - Radar chart showing the film's overall emotion profile
        - Each spoke represents one of 22 emotion categories
        - Larger area = more emotionally expressive film
        - Use comparison mode to overlay multiple films
        
        **Data Resolution**
        - *Smoothed*: 10-minute rolling average for clearer patterns
        - *Raw*: Original dialogue-level scores (noisier but more detailed)
        """
    )

st.divider()

# ============================================================================
# Film and Language Selector
# ============================================================================

# Load film list
try:
    films = get_film_list_with_metadata()
except Exception as e:
    logger.error(f"Failed to load film list: {e}", exc_info=True)
    st.error(f"Failed to load film list: {e}")
    st.stop()

# Film selector
film_options = {film["display_name"]: film for film in films}

# Default to Princess Kaguya English if available
default_film = None
for display_name, film in film_options.items():
    if "Princess Kaguya" in display_name and "English" in display_name:
        default_film = display_name
        break

if default_film is None and film_options:
    default_film = list(film_options.keys())[0]

col_selector, col_resolution = st.columns([3, 1])

with col_selector:
    selected_film_display = st.selectbox(
        "🎬 Select Film & Language",
        options=list(film_options.keys()),
        index=list(film_options.keys()).index(default_film) if default_film else 0,
        help="All 174 film-language combinations available",
    )

selected_film = film_options[selected_film_display]
film_slug = selected_film["film_slug"]
selected_language_code = selected_film["language_code"]

with col_resolution:
    data_mode = st.selectbox(
        "📊 Data Resolution",
        options=["Smoothed", "Raw"],
        index=0,
        help="Smoothed: 10-min rolling avg | Raw: dialogue-level",
    )

is_smoothed = data_mode == "Smoothed"

# ============================================================================
# Data Quality Validation Warning
# ============================================================================

validation_data = None
try:
    validation_data = get_validation_status(film_slug, selected_language_code)
except Exception as e:
    logger.warning(f"Failed to fetch validation status: {e}")

render_data_quality_warning(validation_data)

st.divider()

# ============================================================================
# Load Data
# ============================================================================

with st.spinner("Loading emotion data..."):
    try:
        if is_smoothed:
            timeseries_df = get_film_emotion_timeseries(film_slug, selected_language_code)
        else:
            timeseries_df = get_raw_emotion_peaks(film_slug, selected_language_code)

        emotion_summary = get_film_emotion_summary(film_slug, selected_language_code)

        # Compute peaks directly from displayed timeseries to ensure alignment
        # (peaks mart may use different data source for v2 subtitle versions)
        peaks_df = compute_peaks_from_timeseries(timeseries_df, top_n=3)

    except Exception as e:
        logger.error(f"Failed to load emotion data: {e}", exc_info=True)
        st.error(f"Failed to load emotion data: {e}")
        st.stop()

if timeseries_df.empty:
    st.warning(
        f"Emotion data not available for **{selected_film['film_title']}** "
        f"in **{selected_language_code.upper()}**. Please select another option."
    )
    st.stop()

# ============================================================================
# Emotion Timeline + Peak Dialogue Sidebar
# ============================================================================

st.markdown("### 📈 Emotion Timeline")

col_timeline, col_peaks = st.columns([2, 1])

with col_timeline:
    try:
        timeline_fig = plot_emotion_timeline(
            timeseries_df,
            selected_film["film_title"],
            selected_language_code,
            is_smoothed,
            peaks_df=peaks_df,
        )
        st.plotly_chart(timeline_fig, use_container_width=True, config={"displayModeBar": True})
        
        st.caption(
            f"📊 Showing top 5 dominant emotions. "
            f"{'10-minute rolling average applied for clarity.' if is_smoothed else 'Raw dialogue-level scores.'}"
        )
    except Exception as e:
        logger.error(f"Failed to render emotion timeline: {e}", exc_info=True)
        st.error(f"Failed to render emotion timeline: {e}")

with col_peaks:
    st.markdown("#### 🎭 Peak Moments")
    st.markdown("*Dialogue from the most emotionally intense scenes*")

    try:
        top_5_emotions = get_top_n_emotions(timeseries_df, n=5)
        film_slug_base = re.sub(rf"_{selected_language_code}(_v\d+)?$", "", film_slug)

        if film_slug_base:
            all_peak_dialogues = get_peak_dialogues(
                film_slug_base, selected_language_code, peaks_df
            )

            # Filter peaks: only show rank 1 peaks for top 5 emotions
            # (peaks are now computed from displayed data, so they match the chart stars)
            peak_dialogues = [
                peak
                for peak in all_peak_dialogues
                if peak["emotion_type"] in top_5_emotions
                and peak.get("peak_rank", 1) == 1
            ]
            
            if peak_dialogues:
                for peak in peak_dialogues[:5]:  # Show up to 5 (one per top emotion)
                    emotion_emoji = {
                        "joy": "😊", "fear": "😨", "sadness": "😢", "anger": "😠",
                        "love": "💗", "excitement": "🤩", "admiration": "🤩",
                        "amusement": "😄", "caring": "🤗", "gratitude": "🙏",
                        "optimism": "☀️", "pride": "💪", "relief": "😌",
                        "disappointment": "😞", "nervousness": "😰", "annoyance": "😤",
                        "grief": "😭", "disgust": "🤢",
                    }.get(peak["emotion_type"], "✨")

                    with st.expander(
                        f"{emotion_emoji} **{peak['emotion_type'].capitalize()}** ({peak['minute_range']})",
                        expanded=False,
                    ):
                        st.caption(f"Intensity: {peak['intensity']:.3f}")
                        for line in peak["dialogue_lines"]:
                            st.markdown(f"> {line}")
            else:
                st.info("No high-intensity peaks found for the dominant emotions.")
        else:
            st.info("Unable to load peak dialogues.")

    except Exception as e:
        logger.error(f"Failed to load peak dialogues: {e}", exc_info=True)
        st.warning(f"Could not load peak dialogues: {e}")

st.divider()

# ============================================================================
# Emotion Composition
# ============================================================================

st.markdown("### 🎨 Emotion Composition")

st.markdown(
    """
    <p style="color: #94A3B8; margin-bottom: 1rem;">
        All emotions stacked over time. Positive emotions build upward from zero, 
        negative emotions extend downward — visualizing the emotional density of each scene.
    </p>
    """,
    unsafe_allow_html=True,
)

try:
    if is_smoothed:
        composition_df = get_film_emotion_timeseries(film_slug, selected_language_code)
    else:
        composition_df = get_raw_emotion_peaks(film_slug, selected_language_code)

    composition_fig = plot_emotion_composition(composition_df, selected_film["film_title"])
    st.plotly_chart(composition_fig, use_container_width=True, config={"displayModeBar": True})
except Exception as e:
    logger.error(f"Failed to render emotion composition: {e}", exc_info=True)
    st.error(f"Failed to render emotion composition: {e}")

st.divider()

# ============================================================================
# Emotional Fingerprint
# ============================================================================

st.markdown("### 🔮 Emotional Fingerprint")

st.markdown(
    """
    <p style="color: #94A3B8; margin-bottom: 1rem;">
        The film's unique emotional signature across 22 emotion categories. 
        Use comparison mode to see how different films (or languages) shape the same story.
    </p>
    """,
    unsafe_allow_html=True,
)

enable_comparison = st.checkbox(
    "📊 Compare with other films", value=False, key="fingerprint_comparison"
)

if enable_comparison:
    comparison_films = st.multiselect(
        "Select films to compare (up to 4):",
        options=[f["display_name"] for f in films if f["film_slug"] != selected_film["film_slug"]],
        max_selections=4,
        key="comparison_selector",
    )

    main_label = f"({selected_film['language_code'].upper()}) {selected_film['film_title']}"
    emotion_summaries = [(main_label, emotion_summary)]

    if comparison_films:
        for comp_film_display in comparison_films:
            comp_film = film_options[comp_film_display]
            comp_summary = get_film_emotion_summary(
                comp_film["film_slug"], comp_film["language_code"]
            )
            comp_label = f"({comp_film['language_code'].upper()}) {comp_film['film_title']}"
            emotion_summaries.append((comp_label, comp_summary))
else:
    single_label = f"({selected_film['language_code'].upper()}) {selected_film['film_title']}"
    emotion_summaries = [(single_label, emotion_summary)]

try:
    fingerprint_fig = plot_emotional_fingerprint(
        emotion_summaries, comparison_mode=enable_comparison
    )
    st.plotly_chart(fingerprint_fig, use_container_width=True, config={"displayModeBar": True})
    
    st.caption(
        "🎭 Each axis represents one emotion. The filled area shows the film's emotional profile. "
        "Larger shapes indicate more emotionally expressive content."
    )
except Exception as e:
    logger.error(f"Failed to render emotional fingerprint: {e}", exc_info=True)
    st.error(f"Failed to render emotional fingerprint: {e}")

st.divider()

# ============================================================================
# Export Section
# ============================================================================

st.markdown("### 📥 Export Data")

col_export1, col_export2 = st.columns(2)

with col_export1:
    try:
        csv_data = timeseries_df.to_csv(index=False).encode("utf-8")
        film_slug_filename = selected_film["film_title"].lower().replace(" ", "_").replace("'", "").replace(":", "")
        filename = f"{film_slug_filename}_{selected_language_code}_emotions.csv"

        st.download_button(
            label="📊 Download Emotion Timeline (CSV)",
            data=csv_data,
            file_name=filename,
            mime="text/csv",
            help=f"Export emotion data for {selected_film['film_title']}",
        )
    except Exception as e:
        logger.error(f"Failed to generate CSV export: {e}", exc_info=True)
        st.error(f"Failed to generate CSV export: {e}")

with col_export2:
    # Copy-to-clipboard summary
    if emotion_summary:
        top_emotions = sorted(emotion_summary.items(), key=lambda x: x[1], reverse=True)[:5]
        summary_text = f"""{selected_film['film_title']} ({selected_language_code.upper()})
Top 5 Emotions:
"""
        for i, (emotion, score) in enumerate(top_emotions, 1):
            summary_text += f"{i}. {emotion.capitalize()}: {score:.4f}\n"

        st.code(summary_text, language=None)
        st.caption("📋 Copy the summary above to share.")

# ============================================================================
# Footer
# ============================================================================

st.markdown(
    """
    <div style='text-align: center; padding: 2rem 0; color: #64748B; font-size: 0.85rem;'>
        <p>Emotion analysis powered by multilingual transformer model • 
        28 emotion dimensions • Signal processing for clarity</p>
    </div>
    """,
    unsafe_allow_html=True,
)
