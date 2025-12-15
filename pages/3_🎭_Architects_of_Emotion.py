"""
The Architects of Emotion - Director Profiles Page

Compare signature emotion styles across Studio Ghibli directors.
Explore career evolution and film similarity analysis.
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Add parent directory to path for utils imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.config import THEME
from utils.data_loader import (
    get_director_career_evolution,
    get_director_list,
    get_director_profile,
    get_film_similarity_matrix,
)
from utils.theme import apply_custom_css, render_header
from utils.visualization import (
    plot_career_evolution,
    plot_director_emotion_radar,
    plot_film_similarity_heatmap,
)

# Page configuration
st.set_page_config(page_title="The Architects of Emotion", page_icon="🎭", layout="wide")
apply_custom_css()

# ============================================================================
# Page Header
# ============================================================================

render_header(
    "🎭 The Architects of Emotion",
    "Compare the emotional signatures of Studio Ghibli's master storytellers",
)

# Introduction paragraph with gradient background
st.markdown(
    """
    <div style="background: linear-gradient(90deg, rgba(56, 189, 248, 0.1), rgba(245, 158, 11, 0.1));
                padding: 1rem; border-radius: 8px; margin-bottom: 1.5rem;">
        <p style="color: #F1F5F9; margin: 0;">
            Every director has an <b>emotional fingerprint</b> — a signature blend of joy, fear, 
            love, and melancholy that permeates their work. Here, we decode how Miyazaki, Takahata, 
            and other Ghibli masters paint with emotion across decades of filmmaking.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================================
# Methodology Expander
# ============================================================================

with st.expander("📚 How is this calculated?"):
    st.markdown(
        """
        ### Understanding Director Profiles
        
        **Signature Emotion Profile (Radar Chart)**
        - Aggregates emotion scores across ALL of a director's films
        - Each axis represents one of 22 emotion categories
        - Larger filled area = more emotionally expressive overall
        - Comparison mode overlays multiple directors for direct contrast
        
        **Career Evolution**
        - Tracks how a director's emotional style changed over time
        - Each point = one film's average emotion score
        - Look for trends: Did they become more joyful? More melancholic?
        - Films ordered chronologically by release year
        
        **Film Similarity Matrix**
        - Measures how similar any two films are based on emotional patterns
        - Uses distance calculation across 27 emotion dimensions
        - 100% = identical emotional profiles | 0% = completely different
        - Reveals unexpected connections between films
        
        **What's Excluded**
        
        Six emotions are excluded from director comparisons as they're more 
        reflective of dialogue content than directorial style: confusion, 
        curiosity, desire, realization, surprise, and neutral.
        """
    )

st.divider()

# ============================================================================
# Selectors
# ============================================================================

# Load director list
with st.spinner("Loading director profiles..."):
    directors_list = get_director_list()

if not directors_list:
    st.error("No director data found. Please run the data pipeline first.")
    st.stop()

col_director, col_language, col_comparison = st.columns([2, 1, 1])

with col_director:
    director_options = [d["display_name"] for d in directors_list]
    director_names = [d["director"] for d in directors_list]

    selected_display = st.selectbox(
        "🎬 Select Director",
        options=director_options,
        index=0,
        help="Directors sorted by film count",
    )

    selected_director = director_names[director_options.index(selected_display)]

with col_language:
    language_options = ["English", "French", "Spanish", "Dutch", "Arabic"]
    language_codes = ["en", "fr", "es", "nl", "ar"]

    selected_language_name = st.selectbox(
        "🌐 Language",
        options=language_options,
        index=0,
        help="Language for career evolution chart",
    )

    selected_language_code = language_codes[language_options.index(selected_language_name)]

with col_comparison:
    comparison_enabled = st.checkbox(
        "📊 Compare Directors",
        value=False,
        help="Overlay multiple directors on radar chart",
    )

# Comparison director selector
comparison_directors = []
if comparison_enabled:
    available_directors = [d for d in director_names if d != selected_director]
    comparison_directors = st.multiselect(
        "Select additional directors (up to 3)",
        options=available_directors,
        max_selections=3,
        key="comparison_directors",
    )

st.divider()

# ============================================================================
# Load Director Profile Data
# ============================================================================

try:
    primary_profile = get_director_profile(selected_director)

    if not primary_profile:
        st.error(f"No profile data found for {selected_director}")
        st.stop()

    comparison_profiles = []
    if comparison_enabled and comparison_directors:
        for director in comparison_directors:
            profile = get_director_profile(director)
            if profile:
                comparison_profiles.append(profile)

    all_profiles = [primary_profile] + comparison_profiles

except Exception as e:
    st.error(f"Error loading director data: {e}")
    st.stop()

# ============================================================================
# Director Stats Cards
# ============================================================================

st.markdown(f"### 👤 {selected_director}")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(
        f"""
        <div style="background: {THEME['secondary_bg_color']}; padding: 1rem; border-radius: 8px; text-align: center;">
            <p style="color: {THEME['accent_color']}; font-size: 2rem; margin: 0; font-weight: bold;">
                {primary_profile['film_count']}
            </p>
            <p style="color: #94A3B8; margin: 0; font-size: 0.9rem;">Films Directed</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        f"""
        <div style="background: {THEME['secondary_bg_color']}; padding: 1rem; border-radius: 8px; text-align: center;">
            <p style="color: {THEME['primary_color']}; font-size: 2rem; margin: 0; font-weight: bold;">
                {primary_profile['total_minutes_analyzed']:,}
            </p>
            <p style="color: #94A3B8; margin: 0; font-size: 0.9rem;">Minutes Analyzed</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        f"""
        <div style="background: {THEME['secondary_bg_color']}; padding: 1rem; border-radius: 8px; text-align: center;">
            <p style="color: {THEME['text_color']}; font-size: 1.5rem; margin: 0; font-weight: bold;">
                {primary_profile['earliest_film_year']}-{primary_profile['latest_film_year']}
            </p>
            <p style="color: #94A3B8; margin: 0; font-size: 0.9rem;">{primary_profile['career_span_years']} Year Career</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col4:
    st.markdown(
        f"""
        <div style="background: {THEME['secondary_bg_color']}; padding: 1rem; border-radius: 8px; text-align: center;">
            <p style="color: {THEME['accent_color']}; font-size: 1.5rem; margin: 0; font-weight: bold;">
                {primary_profile['top_emotion_1'].capitalize()}
            </p>
            <p style="color: #94A3B8; margin: 0; font-size: 0.9rem;">Signature Emotion</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Top 3 emotions insight
st.info(
    f"💡 **Top 3 Emotions:** "
    f"{primary_profile['top_emotion_1'].capitalize()} ({primary_profile['top_emotion_1_score']:.3f}) • "
    f"{primary_profile['top_emotion_2'].capitalize()} ({primary_profile['top_emotion_2_score']:.3f}) • "
    f"{primary_profile['top_emotion_3'].capitalize()} ({primary_profile['top_emotion_3_score']:.3f})"
)

# ============================================================================
# Comparison Table (if enabled)
# ============================================================================

if comparison_enabled and comparison_profiles:
    st.divider()
    st.markdown("### 📊 Director Comparison")

    comparison_data = []
    for profile in all_profiles:
        comparison_data.append({
            "Director": profile["director"],
            "Films": profile["film_count"],
            "Top Emotion": profile["top_emotion_1"].capitalize(),
            "Score": f"{profile['top_emotion_1_score']:.3f}",
            "Diversity": f"{profile['emotion_diversity']:.4f}",
            "Career": f"{profile['earliest_film_year']}-{profile['latest_film_year']}",
        })

    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)

st.divider()

# ============================================================================
# Signature Emotion Profile (Radar Chart)
# ============================================================================

st.markdown("### 🎯 Signature Emotion Profile")

st.markdown(
    """
    <p style="color: #94A3B8; margin-bottom: 1rem;">
        The director's emotional fingerprint across 22 emotion categories. 
        Larger filled areas indicate more emotionally expressive filmmaking.
    </p>
    """,
    unsafe_allow_html=True,
)

try:
    radar_fig = plot_director_emotion_radar(
        director_profiles=all_profiles, comparison_mode=comparison_enabled
    )
    st.plotly_chart(radar_fig, use_container_width=True, config={"displayModeBar": True})

    st.caption(
        "🎭 Each axis represents one emotion. The shape reveals the director's emotional priorities — "
        "which emotions dominate their storytelling across all films."
    )
except Exception as e:
    st.error(f"Error generating radar chart: {e}")

st.divider()

# ============================================================================
# Career Evolution Timeline
# ============================================================================

st.markdown(f"### 📈 Career Emotion Evolution")

st.markdown(
    f"""
    <p style="color: #94A3B8; margin-bottom: 1rem;">
        How {selected_director}'s emotional storytelling evolved over time.
        Each line tracks one emotion across their filmography (using {selected_language_name} subtitles).
    </p>
    """,
    unsafe_allow_html=True,
)

try:
    career_df = get_director_career_evolution(selected_director, selected_language_code)

    if not career_df.empty:
        all_relevant_emotions = [
            "admiration", "amusement", "anger", "annoyance", "approval", "caring",
            "disappointment", "disapproval", "disgust", "embarrassment", "excitement",
            "fear", "gratitude", "grief", "joy", "love", "nervousness", "optimism",
            "pride", "relief", "remorse", "sadness",
        ]

        evolution_fig = plot_career_evolution(
            df=career_df, director=selected_director, top_emotions=all_relevant_emotions
        )
        st.plotly_chart(evolution_fig, use_container_width=True, config={"displayModeBar": True})

        st.caption(
            "📊 Track emotional trends across decades. Did they lean into melancholy over time? "
            "Find more joy in later works? The data reveals the arc."
        )
    else:
        st.warning(f"No career evolution data available for {selected_director}")
except Exception as e:
    st.error(f"Error generating career evolution chart: {e}")

st.divider()

# ============================================================================
# Film Similarity Heatmap
# ============================================================================

st.markdown("### 🔗 Film Emotional Similarity")

st.markdown(
    f"""
    <p style="color: #94A3B8; margin-bottom: 1rem;">
        Which films share similar emotional DNA? This matrix reveals unexpected connections — 
        films that feel similar even if their plots differ dramatically (using {selected_language_name} subtitles).
    </p>
    """,
    unsafe_allow_html=True,
)

try:
    similarity_df = get_film_similarity_matrix(selected_language_code)

    if not similarity_df.empty:
        heatmap_fig = plot_film_similarity_heatmap(
            df=similarity_df, language_code=selected_language_code
        )
        st.plotly_chart(heatmap_fig, use_container_width=True, config={"responsive": True})

        st.caption(
            "🎬 Blue = similar emotional journeys | Red = different emotional styles. "
            "Hover over cells to see exact similarity percentages."
        )
    else:
        st.warning(f"No similarity data available for {selected_language_name}")
except Exception as e:
    st.error(f"Error generating similarity heatmap: {e}")

st.divider()

# ============================================================================
# Export Section
# ============================================================================

st.markdown("### 📥 Export Data")

col_export1, col_export2, col_export3 = st.columns(3)

with col_export1:
    try:
        profile_df = pd.DataFrame([primary_profile])
        csv_profile = profile_df.to_csv(index=False).encode("utf-8")
        director_slug = selected_director.lower().replace(" ", "_").replace("'", "")
        filename_profile = f"{director_slug}_emotion_profile.csv"

        st.download_button(
            label="👤 Download Director Profile",
            data=csv_profile,
            file_name=filename_profile,
            mime="text/csv",
        )
    except Exception as e:
        st.error(f"Error preparing profile export: {e}")

with col_export2:
    try:
        if not similarity_df.empty:
            csv_similarity = similarity_df.to_csv(index=False).encode("utf-8")
            filename_similarity = f"film_similarity_matrix_{selected_language_code}.csv"

            st.download_button(
                label="🔗 Download Similarity Matrix",
                data=csv_similarity,
                file_name=filename_similarity,
                mime="text/csv",
            )
    except Exception as e:
        st.error(f"Error preparing similarity export: {e}")

with col_export3:
    # Quick stats for copy
    quick_stats = f"""{selected_director}
Films: {primary_profile['film_count']}
Career: {primary_profile['earliest_film_year']}-{primary_profile['latest_film_year']}
Signature: {primary_profile['top_emotion_1'].capitalize()} ({primary_profile['top_emotion_1_score']:.3f})
Diversity: {primary_profile['emotion_diversity']:.4f}"""

    st.code(quick_stats, language=None)
    st.caption("📋 Copy to share.")

# ============================================================================
# Footer
# ============================================================================

st.markdown(
    """
    <div style='text-align: center; padding: 2rem 0; color: #64748B; font-size: 0.85rem;'>
        <p>Director profiles aggregated across all available films and languages • 
        22 emotion categories analyzed</p>
    </div>
    """,
    unsafe_allow_html=True,
)
