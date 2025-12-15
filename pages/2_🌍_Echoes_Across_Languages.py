"""
Echoes Across Languages - Cross-Language Emotion Analysis Page

Discover how emotion patterns shift across subtitle translations.
Showcases translation biases, consistency scores, and language comparisons.

[Source: Story 5.5 - Epic 5: Production-Grade Streamlit Emotion Analysis App]
"""

import numpy as np
import pandas as pd
import streamlit as st

# Add parent directory to path for utils imports
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.config import THEME
from utils.data_loader import (
    LANGUAGE_NAMES,
    get_consistency_matrix,
    get_consistency_score,
    get_emotion_by_language,
    get_film_consistency_ranking,
    get_film_list_for_language_pair,
    get_language_pairs,
    get_translation_biases,
)
from utils.theme import apply_custom_css, render_header
from utils.visualization import (
    EXCLUDED_EMOTIONS,
    plot_consistency_heatmap,
    plot_divergence_bar_chart,
    plot_emotion_by_language,
)

# Page configuration
st.set_page_config(page_title="Echoes Across Languages", page_icon="🌍", layout="wide")
apply_custom_css()

# ============================================================================
# Page Header (AC8)
# ============================================================================

render_header("🌍 Echoes Across Languages", "Discover how emotion patterns shift across subtitle translations")

st.markdown(
    """
    <div style="background: linear-gradient(90deg, rgba(56, 189, 248, 0.1), rgba(245, 158, 11, 0.1));
                padding: 1rem; border-radius: 8px; margin-bottom: 1.5rem;">
        <p style="color: #F1F5F9; margin: 0;">
            This page reveals how <b>translation choices</b> affect emotional interpretation.
            Compare emotion patterns across <b>5 languages</b> to discover translation biases
            and cultural adaptation patterns in Studio Ghibli subtitles.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================================
# Methodology Expander (AC8)
# ============================================================================

with st.expander("📚 How are translation biases calculated?"):
    st.markdown(
        """
        ### Methodology

        **Percent Difference Formula:**
        ```
        Percent Difference = ((Score_B - Score_A) / Score_A) × 100
        ```

        **Significance Threshold:** Differences > **20%** are flagged as significant.
        This threshold was chosen based on Epic 2.5.4 analysis - smaller differences
        are likely noise from subtitle timing variations, sample size, or model uncertainty.

        **Why Do Differences Exist?**
        - **Cultural adaptation**: Humor, idioms, and expressions translate differently
        - **Language structure**: Some emotions are expressed more/less explicitly in different languages
        - **Subtitle quality**: Different source materials, timing, and completeness
        - **Translation style**: Literal vs. creative translation approaches

        **⚠️ Important Caveat:**
        The emotion model was trained primarily on English text. Other languages may show
        baseline biases due to how multilingual BERT handles different scripts and structures.
        Arabic, in particular, may show amplified differences due to right-to-left script
        and different syntactic patterns.

        **Consistency Score:** Measures how similarly emotions are conveyed across translations.
        Calculated as 100% minus the average absolute percent difference.
        - **> 80%**: Highly Consistent (similar emotion patterns)
        - **60-80%**: Moderately Consistent (some translation effects visible)
        - **< 60%**: Divergent (significant language/cultural differences)
        """
    )

# ============================================================================
# Selectors Row (AC1)
# ============================================================================

# Language options for individual selectors
LANGUAGE_OPTIONS = {
    "English": "en",
    "French": "fr", 
    "Spanish": "es",
    "Dutch": "nl",
    "Arabic": "ar",
}
LANGUAGE_DISPLAY = {v: k for k, v in LANGUAGE_OPTIONS.items()}

col1, col2, col3, col4 = st.columns([1.5, 1.5, 2, 3])

with col1:
    source_language_display = st.selectbox(
        "🌐 Source Language",
        options=list(LANGUAGE_OPTIONS.keys()),
        index=0,  # Default: English
        help="Select the source language to compare from",
    )

with col2:
    # Filter out source language from target options
    target_options = [lang for lang in LANGUAGE_OPTIONS.keys() if lang != source_language_display]
    target_language_display = st.selectbox(
        "➡️ Target Language",
        options=target_options,
        index=0,  # Default: French (first after English)
        help="Select the target language to compare to",
    )

# Get language codes
source_code = LANGUAGE_OPTIONS[source_language_display]
target_code = LANGUAGE_OPTIONS[target_language_display]

# Database stores pairs alphabetically (e.g., 'en' < 'fr' means en-fr, not fr-en)
# We need to query with the alphabetically-ordered pair but track if we need to swap display
if source_code < target_code:
    language_a = source_code
    language_b = target_code
    display_a = source_language_display
    display_b = target_language_display
    is_swapped = False  # User order matches DB order
else:
    language_a = target_code
    language_b = source_code
    display_a = target_language_display  # DB language_a
    display_b = source_language_display  # DB language_b
    is_swapped = True  # User order is reversed from DB order

# Load films for selected pair
films = get_film_list_for_language_pair(language_a, language_b)
film_options = {"All Films (Aggregated)": None}
film_options.update({f["film_title"]: f["film_id"] for f in films})

with col3:
    selected_film_display = st.selectbox(
        "🎬 Film",
        options=list(film_options.keys()),
        index=0,
        help="Select a specific film or view aggregated data across all films",
    )

selected_film_id = film_options[selected_film_display]
selected_film_title = selected_film_display if selected_film_id else None

# Emotion filter (22 relevant emotions)
all_relevant_emotions = [
    e for e in [
        "admiration", "amusement", "anger", "annoyance", "approval",
        "caring", "disappointment", "disapproval", "disgust", "embarrassment",
        "excitement", "fear", "gratitude", "grief", "joy", "love",
        "nervousness", "optimism", "pride", "relief", "remorse", "sadness"
    ]
]

with col4:
    selected_emotions = st.multiselect(
        "🎭 Emotion Filter",
        options=[e.capitalize() for e in all_relevant_emotions],
        default=[e.capitalize() for e in all_relevant_emotions],
        help="Filter to specific emotions for focused analysis",
    )

# User-facing display labels (source → target from user's perspective)
user_source_display = source_language_display
user_target_display = target_language_display

# Convert back to lowercase for queries
selected_emotions_lower = [e.lower() for e in selected_emotions]

st.divider()

# ============================================================================
# Main Content: Two Columns
# ============================================================================

left_col, right_col = st.columns([3, 2])

# ============================================================================
# Left Column: Translation Bias Detection (AC2)
# ============================================================================

with left_col:
    st.subheader("📊 Translation Bias Detection")

    # Load bias data
    biases_df = get_translation_biases(language_a, language_b, selected_film_id, top_n=22)

    # Filter to selected emotions
    if selected_emotions_lower:
        biases_df = biases_df[biases_df["emotion_type"].isin(selected_emotions_lower)]

    # Take top 10 for display
    top_biases_df = biases_df.head(10)

    if not top_biases_df.empty:
        # Handle swap: adjust percent_difference to reflect user's source → target perspective
        display_df = top_biases_df.copy()
        if is_swapped:
            # User selected target → source (reverse of DB order), so invert the difference
            display_df["percent_difference"] = -display_df["percent_difference"]
            display_df["difference_score"] = -display_df["difference_score"]
            # Swap score columns for display
            display_df["user_score_source"] = display_df["avg_score_lang_b"]
            display_df["user_score_target"] = display_df["avg_score_lang_a"]
        else:
            display_df["user_score_source"] = display_df["avg_score_lang_a"]
            display_df["user_score_target"] = display_df["avg_score_lang_b"]

        # Insight Card (AC2.3) - using user's perspective
        top_3 = display_df.head(3)
        insights = []
        for _, row in top_3.iterrows():
            emotion = row["emotion_type"]
            pct = row["percent_difference"]
            score_source = row["user_score_source"]
            score_target = row["user_score_target"]
            direction = "more" if pct > 0 else "less"

            insights.append(
                f"**{user_target_display}** subtitles show **{abs(pct):.0f}%** {direction} "
                f"**{emotion}** than {user_source_display} ({score_source:.4f} → {score_target:.4f})"
            )

        st.info("💡 **Top Insights**\n\n" + "\n\n".join(insights))

        # Divergence Bar Chart (AC2.2) - using user's perspective
        fig = plot_divergence_bar_chart(display_df, user_source_display, user_target_display)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True})

        st.caption(
            f"📈 **Reading the chart:** Bars extending **right** (cyan) indicate {user_target_display} "
            f"has higher scores. Bars extending **left** (gold) indicate {user_source_display} has higher scores."
        )

        # Bias Ranking Table (AC2.1)
        with st.expander("📋 View Full Bias Ranking Table"):
            # Prepare table display DataFrame using user's perspective
            table_df = display_df.copy()
            table_df["Emotion"] = table_df["emotion_type"].str.capitalize()
            table_df[f"{user_source_display} Score"] = table_df["user_score_source"].apply(lambda x: f"{x:.4f}")
            table_df[f"{user_target_display} Score"] = table_df["user_score_target"].apply(lambda x: f"{x:.4f}")
            table_df["Difference"] = table_df["difference_score"].apply(lambda x: f"{x:+.4f}")
            table_df["% Change"] = table_df["percent_difference"].apply(lambda x: f"{x:+.1f}%")
            table_df["Significant"] = table_df["is_significant"].apply(lambda x: "⚠️ Yes" if x else "No")

            st.dataframe(
                table_df[["Emotion", f"{user_source_display} Score", f"{user_target_display} Score", "Difference", "% Change", "Significant"]],
                use_container_width=True,
                hide_index=True,
            )
    else:
        st.warning("No translation bias data available for the selected filters.")

# ============================================================================
# Right Column: Consistency Score & Heatmap (AC3)
# ============================================================================

with right_col:
    st.subheader("🎯 Translation Consistency")

    # Calculate consistency score (inverted: 100% = perfectly consistent)
    raw_divergence = get_consistency_score(language_a, language_b, selected_film_id)
    consistency_score = max(0, 100 - raw_divergence)  # Invert so higher = more consistent

    # Determine rating (inverted thresholds)
    if consistency_score > 80:
        rating = "Highly Consistent"
        rating_color = "#34D399"  # Green
        rating_emoji = "✅"
    elif consistency_score > 60:
        rating = "Moderately Consistent"
        rating_color = "#FBBF24"  # Yellow
        rating_emoji = "⚡"
    else:
        rating = "Divergent"
        rating_color = "#EF4444"  # Red
        rating_emoji = "⚠️"

    # Consistency Score Card (AC3.2)
    st.markdown(
        f"""
        <div style="background: {THEME['secondary_bg_color']}; padding: 1.5rem;
                    border-radius: 12px; text-align: center; border: 2px solid {rating_color};">
            <h2 style="color: {rating_color}; margin: 0; font-size: 2.5rem;">{consistency_score:.1f}%</h2>
            <p style="color: {THEME['text_color']}; margin: 0.5rem 0 0 0; font-size: 1.2rem;">
                {rating_emoji} {rating}
            </p>
            <p style="color: #94A3B8; margin: 0.5rem 0 0 0; font-size: 0.9rem;">
                Emotion pattern similarity across {len(selected_emotions_lower)} emotions
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # Consistency Heatmap (AC3.3)
    st.markdown("**Cross-Language Consistency Matrix**")
    consistency_matrix = get_consistency_matrix(selected_film_id)

    # Invert matrix values so higher = more consistent (100 - divergence)
    consistency_matrix_inverted = 100 - consistency_matrix

    fig_heatmap = plot_consistency_heatmap(consistency_matrix_inverted, selected_film_title)
    st.plotly_chart(fig_heatmap, use_container_width=True, config={"displayModeBar": True})

    st.caption(
        "🔍 **Reading the heatmap:** Higher values (green) indicate more consistent emotion patterns "
        "between language pairs. Lower values (red) indicate more divergent translations."
    )

st.divider()

# ============================================================================
# Emotion-Specific Language Comparison (AC4)
# ============================================================================

st.subheader("🔬 Emotion-Specific Language Comparison")

col_emotion_select, col_emotion_chart = st.columns([1, 3])

# Default to top biased emotion if available
default_emotion = "amusement"
if not biases_df.empty:
    default_emotion = biases_df.iloc[0]["emotion_type"]

with col_emotion_select:
    emotion_for_comparison = st.selectbox(
        "Select Emotion",
        options=[e.capitalize() for e in all_relevant_emotions],
        index=all_relevant_emotions.index(default_emotion) if default_emotion in all_relevant_emotions else 0,
    )

    emotion_lower = emotion_for_comparison.lower()

    # Load emotion data across languages
    emotion_by_lang_df = get_emotion_by_language(emotion_lower, selected_film_id)

    if not emotion_by_lang_df.empty:
        # Statistical Summary (AC4.3)
        st.markdown("**Statistical Summary**")

        max_row = emotion_by_lang_df.loc[emotion_by_lang_df["avg_score"].idxmax()]
        min_row = emotion_by_lang_df.loc[emotion_by_lang_df["avg_score"].idxmin()]
        score_range = max_row["avg_score"] - min_row["avg_score"]
        std_dev = emotion_by_lang_df["avg_score"].std()

        st.markdown(
            f"""
            <div style="background: {THEME['secondary_bg_color']}; padding: 1rem;
                        border-radius: 8px; font-size: 0.9rem;">
                <p style="margin: 0.3rem 0;"><b>Highest:</b> {max_row['language_name']} ({max_row['avg_score']:.4f})</p>
                <p style="margin: 0.3rem 0;"><b>Lowest:</b> {min_row['language_name']} ({min_row['avg_score']:.4f})</p>
                <p style="margin: 0.3rem 0;"><b>Range:</b> {score_range:.4f}</p>
                <p style="margin: 0.3rem 0;"><b>Std Dev:</b> {std_dev:.4f}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

with col_emotion_chart:
    if not emotion_by_lang_df.empty:
        # Language Comparison Chart (AC4.2) - highlight user's selected languages
        fig_emotion = plot_emotion_by_language(
            emotion_by_lang_df,
            emotion_lower,
            highlight_languages=(source_code, target_code),
        )
        st.plotly_chart(fig_emotion, use_container_width=True, config={"displayModeBar": True})

        st.caption(
            f"📊 **Highlighted languages** ({user_source_display} and {user_target_display}) are shown in cyan. "
            "Other languages in gold for reference."
        )
    else:
        st.warning(f"No data available for {emotion_for_comparison}.")

st.divider()

# ============================================================================
# Film-Level Translation Consistency (AC5)
# ============================================================================

st.subheader("🎬 Film-Level Translation Consistency")

film_ranking_df = get_film_consistency_ranking(language_a, language_b)

if not film_ranking_df.empty:
    # Invert consistency score (100 - divergence) so higher = more consistent
    film_ranking_df["consistency_score_inverted"] = 100 - film_ranking_df["consistency_score"]

    # Sort by inverted score descending (most consistent first)
    film_ranking_df = film_ranking_df.sort_values("consistency_score_inverted", ascending=False)

    # Identify outliers (<40% consistency = highly divergent)
    film_ranking_df["is_outlier"] = film_ranking_df["consistency_score_inverted"] < 40

    # Prepare display DataFrame
    display_ranking = film_ranking_df.copy()
    display_ranking["Rank"] = range(1, len(display_ranking) + 1)
    display_ranking["Film Title"] = display_ranking["film_title"]
    display_ranking["Consistency Score"] = display_ranking["consistency_score_inverted"].apply(lambda x: f"{x:.1f}%")
    display_ranking["Most Divergent Emotion"] = display_ranking["most_divergent_emotion"].str.capitalize()
    display_ranking["Max % Difference"] = display_ranking["max_percent_difference"].apply(lambda x: f"{x:.1f}%")
    display_ranking["Status"] = display_ranking["is_outlier"].apply(lambda x: "⚠️ Outlier" if x else "✓ Normal")

    # Show table
    st.dataframe(
        display_ranking[["Rank", "Film Title", "Consistency Score", "Most Divergent Emotion", "Max % Difference", "Status"]],
        width="stretch",
        hide_index=True,
        height=400,
    )

    # Outlier explanation
    outlier_count = display_ranking["is_outlier"].sum()
    if outlier_count > 0:
        st.info(
            f"⚠️ **{outlier_count} film(s)** show outlier-level inconsistency (<40% consistency). "
            "This may indicate subtitle quality issues, significant cultural adaptation, or timing discrepancies."
        )

    st.caption(
        "📋 Films are ranked by consistency score (descending). "
        "Higher scores indicate translations that preserve similar emotion patterns."
    )
else:
    st.warning("No film consistency data available for the selected language pair.")

st.divider()

# ============================================================================
# Export Functionality (AC6)
# ============================================================================

st.subheader("📥 Export Data")

export_col1, export_col2, export_col3 = st.columns(3)

with export_col1:
    # CSV Export - Translation Bias Report (AC6.1)
    if not biases_df.empty:
        film_suffix = selected_film_id[:8] if selected_film_id else "all"
        filename_bias = f"translation_bias_{language_a}_{language_b}_{film_suffix}.csv"

        csv_bias = biases_df.to_csv(index=False)
        st.download_button(
            label="📊 Download Bias Report (CSV)",
            data=csv_bias,
            file_name=filename_bias,
            mime="text/csv",
        )
    else:
        st.button("📊 Download Bias Report (CSV)", disabled=True)

with export_col2:
    # CSV Export - Consistency Matrix (AC6.2)
    if not consistency_matrix.empty:
        film_suffix = selected_film_id[:8] if selected_film_id else "all"
        filename_matrix = f"consistency_matrix_{film_suffix}.csv"

        csv_matrix = consistency_matrix.to_csv()
        st.download_button(
            label="🎯 Download Consistency Matrix (CSV)",
            data=csv_matrix,
            file_name=filename_matrix,
            mime="text/csv",
        )
    else:
        st.button("🎯 Download Consistency Matrix (CSV)", disabled=True)

with export_col3:
    # Copy-to-Clipboard - Top Insights (AC6.3)
    if not biases_df.empty:
        # Use display_df which has user's perspective adjustments
        summary_text = f"""Cross-Language Emotion Analysis Summary
Language Pair: {user_source_display} → {user_target_display}
Film: {selected_film_display}
Consistency Score: {consistency_score:.1f}% ({rating})

Top 3 Translation Biases:
"""
        for i, (_, row) in enumerate(display_df.head(3).iterrows(), 1):
            direction = "more" if row["percent_difference"] > 0 else "less"
            summary_text += f"{i}. {row['emotion_type'].capitalize()}: {abs(row['percent_difference']):.1f}% {direction} in {user_target_display}\n"

        st.code(summary_text, language=None)
        st.caption("📋 Copy the text above to share insights.")
    else:
        st.info("No data available for summary export.")

# ============================================================================
# Footer
# ============================================================================

st.markdown(
    """
    <div style="text-align: center; padding: 2rem 0; color: #64748B; font-size: 0.85rem;">
        <p>Data from <b>22 emotions</b> across <b>5 languages</b> • 
        Powered by multilingual GoEmotions BERT model</p>
    </div>
    """,
    unsafe_allow_html=True,
)
