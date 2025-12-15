"""
Plotly visualization helpers for SpiritedData Streamlit app.

Provides chart generation functions following "Spirit World" dark theme.
Follows Epic 3.5 emotion analysis guidelines:
- Exclude neutral emotion from calculations
- Show top 5 emotions (not top 3)
- Display negative emotions with negative intensity (below zero)

[Source: Story 5.2 - AC4, AC3; Story 3.5.2 - Emotion Guidelines]
"""

from typing import List, Optional, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from .config import THEME

# Emotion categorization per Epic 3.5 guidelines
POSITIVE_EMOTIONS = [
    "admiration",
    "amusement",
    "approval",
    "caring",
    "excitement",
    "gratitude",
    "joy",
    "love",
    "optimism",
    "pride",
    "relief",
]

NEGATIVE_EMOTIONS = [
    "anger",
    "annoyance",
    "disappointment",
    "disapproval",
    "disgust",
    "embarrassment",
    "fear",
    "grief",
    "nervousness",
    "remorse",
    "sadness",
]

# Neutral/ambiguous emotions EXCLUDED from analysis per Epic 3.5
EXCLUDED_EMOTIONS = ["confusion", "curiosity", "desire", "realization", "surprise", "neutral"]


def get_top_n_emotions(df: pd.DataFrame, n: int = 5) -> List[str]:
    """
    Calculate top N dominant emotions across entire timeline.

    Excludes neutral/ambiguous emotions per Epic 3.5 guidelines.

    Args:
        df: DataFrame with emotion_* columns
        n: Number of top emotions to return (default: 5 per Epic 3.5)

    Returns:
        List of emotion names (e.g., ['joy', 'fear', 'admiration', 'sadness', 'caring'])

    [Source: Story 5.2 - AC4.5; Story 3.5.2 - Emotion Guidelines]
    """
    # Get all emotion columns EXCLUDING neutral/ambiguous per Epic 3.5
    all_cols = [col for col in df.columns if col.startswith("emotion_")]
    emotion_cols = [col for col in all_cols if col.replace("emotion_", "") not in EXCLUDED_EMOTIONS]

    # Calculate average score for each emotion across timeline
    emotion_averages = df[emotion_cols].mean().sort_values(ascending=False)

    # Get top N emotion names (strip "emotion_" prefix)
    top_emotions = [col.replace("emotion_", "") for col in emotion_averages.head(n).index]

    return top_emotions


def plot_emotion_preview(
    df: pd.DataFrame, film_title: str, language_code: str, top_emotions: Optional[List[str]] = None
) -> go.Figure:
    """
    Create emotion timeline preview chart for film selector.

    Follows Epic 3.5 guidelines:
    - Shows top 5 emotions (excludes neutral/ambiguous)
    - Negative emotions displayed with negative intensity (inverted below zero)

    Args:
        df: DataFrame with minute_offset and emotion_* columns
        film_title: Film title for chart title
        language_code: Language code for chart subtitle
        top_emotions: List of emotions to plot (default: auto-detect top 5)

    Returns:
        Plotly Figure object

    [Source: Story 5.2 - AC4; Story 3.5.2 - Emotion Guidelines]
    """
    if top_emotions is None:
        top_emotions = get_top_n_emotions(df, n=5)

    # Define emotion colors (neon/pastel for dark background)
    emotion_colors = {
        # Positive emotions - warm colors
        "joy": "#FFD700",  # Gold
        "admiration": "#EC4899",  # Pink
        "amusement": "#F59E0B",  # Amber
        "love": "#F472B6",  # Light Pink
        "excitement": "#FBBF24",  # Yellow
        "gratitude": "#34D399",  # Green
        "optimism": "#60A5FA",  # Light Blue
        "caring": "#A78BFA",  # Light Purple
        "approval": "#4ADE80",  # Light Green
        "pride": "#FB923C",  # Orange
        "relief": "#22D3EE",  # Cyan
        # Negative emotions - cool/dark colors
        "fear": "#9333EA",  # Purple
        "sadness": "#3B82F6",  # Blue
        "anger": "#EF4444",  # Red
        "disappointment": "#6366F1",  # Indigo
        "disgust": "#DC2626",  # Dark Red
        "grief": "#1E3A8A",  # Dark Blue
        "embarrassment": "#DB2777",  # Pink-Red
        "nervousness": "#7C3AED",  # Violet
        "annoyance": "#F97316",  # Orange-Red
        "disapproval": "#7C2D12",  # Brown
        "remorse": "#831843",  # Dark Pink
    }

    fig = go.Figure()

    # Track all y-values for dynamic scaling
    all_y_values = []

    # Add line trace for each top emotion
    for emotion in top_emotions:
        col_name = f"emotion_{emotion}"
        if col_name not in df.columns:
            continue

        color = emotion_colors.get(emotion, "#94A3B8")  # Default to slate

        # Invert negative emotions per Epic 3.5 guidelines
        if emotion in NEGATIVE_EMOTIONS:
            y_values = -df[col_name]  # Negative values display below zero
            hover_template = f"<b>{emotion.capitalize()} (negative)</b><br>Minute: %{{x}}<br>Intensity: %{{y:.3f}}<extra></extra>"
        else:
            y_values = df[col_name]
            hover_template = f"<b>{emotion.capitalize()}</b><br>Minute: %{{x}}<br>Intensity: %{{y:.3f}}<extra></extra>"

        all_y_values.extend(y_values.tolist())

        fig.add_trace(
            go.Scatter(
                x=df["minute_offset"],
                y=y_values,
                mode="lines",
                name=emotion.capitalize(),
                line=dict(color=color, width=2),
                hovertemplate=hover_template,
            )
        )

    # Calculate dynamic y-axis range with 20% padding
    if all_y_values:
        y_min = min(all_y_values)
        y_max = max(all_y_values)
        y_range = y_max - y_min

        # Add 20% padding for visual breathing room
        padding = y_range * 0.2
        y_axis_min = y_min - padding
        y_axis_max = y_max + padding

        # Ensure minimum range of 0.02 for very flat data
        if y_range < 0.01:
            y_axis_min = -0.01
            y_axis_max = 0.01
    else:
        # Fallback if no data
        y_axis_min = -0.05
        y_axis_max = 0.05

    # Add zero baseline per Epic 3.5 (separates positive/negative emotions)
    fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1, opacity=0.5)

    # Add positive zone shading (above zero) - use dynamic range
    if y_axis_max > 0:
        fig.add_hrect(
            y0=0, y1=y_axis_max, fillcolor="green", opacity=0.05, layer="below", line_width=0
        )

    # Add negative zone shading (below zero) - use dynamic range
    if y_axis_min < 0:
        fig.add_hrect(
            y0=y_axis_min, y1=0, fillcolor="red", opacity=0.05, layer="below", line_width=0
        )

    # Style chart with "Spirit World" theme
    fig.update_layout(
        title=dict(
            text=f"{film_title} - Emotional Journey",
            font=dict(size=18, color=THEME["text_color"]),
            x=0.5,
            xanchor="center",
        ),
        xaxis_title="Runtime (minutes)",
        yaxis_title="Emotion Intensity (+ positive / - negative)",
        xaxis=dict(gridcolor="#1E293B", color=THEME["text_color"]),
        yaxis=dict(
            gridcolor="#1E293B",
            color=THEME["text_color"],
            range=[y_axis_min, y_axis_max],  # Dynamic range based on actual data
            zeroline=True,
            zerolinecolor="gray",
            zerolinewidth=1,
            tickformat=".3f",  # Show 3 decimal places for small values
        ),
        plot_bgcolor=THEME["background_color"],
        paper_bgcolor=THEME["background_color"],
        font=dict(color=THEME["text_color"], family=THEME["font_body"]),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(30, 41, 59, 0.8)",  # Semi-transparent slate
            bordercolor=THEME["primary_color"],
            borderwidth=1,
        ),
        hovermode="x unified",
        height=400,
    )

    return fig


def plot_emotion_bar(emotion_score: float, max_score: float = 1.0) -> go.Figure:
    """
    Create small horizontal bar chart for insight cards.

    Args:
        emotion_score: Emotion score value (0-1)
        max_score: Maximum score for scale (default: 1.0)

    Returns:
        Plotly Figure object

    [Source: Story 5.2 - AC3]
    """
    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=[emotion_score],
            y=[""],
            orientation="h",
            marker=dict(
                color=THEME["primary_color"], line=dict(color=THEME["accent_color"], width=1)
            ),
            hoverinfo="skip",
        )
    )

    fig.update_layout(
        xaxis=dict(range=[0, max_score], showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(showticklabels=False),
        plot_bgcolor=THEME["background_color"],
        paper_bgcolor=THEME["background_color"],
        margin=dict(l=0, r=0, t=0, b=0),
        height=30,
        showlegend=False,
    )

    return fig


# ============================================================================
# Epic 5.3: Film Explorer (The Spirit Archives) Visualizations
# ============================================================================


def plot_emotion_timeline(
    df: pd.DataFrame,
    film_title: str,
    language_code: str,
    is_smoothed: bool = True,
    peaks_df: Optional[pd.DataFrame] = None,
) -> go.Figure:
    """
    Create emotion timeline with zone shading and top 5 dominant emotions.

    Follows Epic 3.5 guidelines:
    - Shows top 5 emotions (excludes neutral/ambiguous)
    - Negative emotions displayed with negative intensity (inverted below zero)
    - Zero baseline with green/red zone shading

    Args:
        df: DataFrame with minute_offset and emotion_* columns
        film_title: Film title for chart title
        language_code: Language code for display
        is_smoothed: Whether data is smoothed (10-min rolling avg) or raw

    Returns:
        Plotly Figure object

    [Source: Story 5.3 - Task 2.1, AC2]
    """
    if df.empty:
        # Return empty figure with message
        fig = go.Figure()
        fig.add_annotation(
            text="No emotion data available for this film/language combination",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=16, color=THEME["text_color"]),
        )
        fig.update_layout(
            plot_bgcolor=THEME["background_color"],
            paper_bgcolor=THEME["background_color"],
            height=500,
        )
        return fig

    # Get top 5 emotions (excluding neutral/ambiguous per Epic 3.5)
    top_emotions = get_top_n_emotions(df, n=5)

    # Define emotion colors
    emotion_colors = {
        "joy": "#FFD700",
        "admiration": "#EC4899",
        "amusement": "#F59E0B",
        "love": "#F472B6",
        "excitement": "#FBBF24",
        "gratitude": "#34D399",
        "optimism": "#60A5FA",
        "caring": "#A78BFA",
        "approval": "#4ADE80",
        "pride": "#FB923C",
        "relief": "#22D3EE",
        "fear": "#9333EA",
        "sadness": "#3B82F6",
        "anger": "#EF4444",
        "disappointment": "#6366F1",
        "disgust": "#DC2626",
        "grief": "#1E3A8A",
        "embarrassment": "#DB2777",
        "nervousness": "#7C3AED",
        "annoyance": "#F97316",
        "disapproval": "#7C2D12",
        "remorse": "#831843",
    }

    fig = go.Figure()

    # Track all y-values for dynamic scaling
    all_y_values = []

    # Add line trace for each top emotion
    for emotion in top_emotions:
        col_name = f"emotion_{emotion}"
        if col_name not in df.columns:
            continue

        color = emotion_colors.get(emotion, "#94A3B8")

        # Invert negative emotions per Epic 3.5 guidelines
        if emotion in NEGATIVE_EMOTIONS:
            y_values = -df[col_name]
            hover_template = f"<b>{emotion.capitalize()} (negative)</b><br>Minute: %{{x}}<br>Intensity: %{{y:.3f}}<extra></extra>"
        else:
            y_values = df[col_name]
            hover_template = f"<b>{emotion.capitalize()}</b><br>Minute: %{{x}}<br>Intensity: %{{y:.3f}}<extra></extra>"

        all_y_values.extend(y_values.tolist())

        fig.add_trace(
            go.Scatter(
                x=df["minute_offset"],
                y=y_values,
                mode="lines",
                name=emotion.capitalize(),
                line=dict(color=color, width=2),
                hovertemplate=hover_template,
            )
        )

    # Calculate dynamic y-axis range with 20% padding
    if all_y_values:
        y_min = min(all_y_values)
        y_max = max(all_y_values)
        y_range = y_max - y_min
        padding = y_range * 0.2
        y_axis_min = y_min - padding
        y_axis_max = y_max + padding

        if y_range < 0.01:
            y_axis_min = -0.01
            y_axis_max = 0.01
    else:
        y_axis_min = -0.05
        y_axis_max = 0.05

    # Add positive zone shading (green, above zero)
    if y_axis_max > 0:
        fig.add_hrect(
            y0=0, y1=y_axis_max, fillcolor="green", opacity=0.05, layer="below", line_width=0
        )

    # Add negative zone shading (red, below zero)
    if y_axis_min < 0:
        fig.add_hrect(
            y0=y_axis_min, y1=0, fillcolor="red", opacity=0.05, layer="below", line_width=0
        )

    # Add zero baseline
    fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1, opacity=0.5)

    # Add peak markers with emotion-matched colors if provided
    # Peaks are now computed directly from displayed timeseries, so they match the chart
    if peaks_df is not None and not peaks_df.empty:
        for _, peak in peaks_df.iterrows():
            emotion_label = peak["emotion_type"]
            minute = peak["peak_minute_offset"]
            peak_rank = peak.get("peak_rank", 1)

            # Only annotate top peaks (rank 1) for top 5 emotions
            if emotion_label in top_emotions and peak_rank == 1:
                # Determine if negative emotion for y-position
                y_val = df[df["minute_offset"] == minute][f"emotion_{emotion_label}"]
                if not y_val.empty:
                    y_val = y_val.iloc[0]
                    if emotion_label in NEGATIVE_EMOTIONS:
                        y_val = -y_val

                    # Use same color as emotion line for star marker
                    star_color = emotion_colors.get(emotion_label, "#94A3B8")

                    # Add star marker (no confusing tooltips - just visual markers)
                    fig.add_trace(
                        go.Scatter(
                            x=[minute],
                            y=[y_val],
                            mode="markers",
                            marker=dict(
                                size=12,
                                color=star_color,
                                symbol="star",
                                line=dict(color="white", width=1),
                            ),
                            showlegend=False,
                            hoverinfo="skip",  # Disable hover to avoid confusion with line tooltips
                        )
                    )

    # Style chart
    data_type = "Smoothed (10-min avg)" if is_smoothed else "Raw (dialogue-level)"
    language_display = {
        "en": "English",
        "fr": "French",
        "es": "Spanish",
        "nl": "Dutch",
        "ar": "Arabic",
    }.get(language_code, language_code.upper())

    fig.update_layout(
        title=dict(
            text=f"{film_title} - Emotion Timeline ({language_display}, {data_type})",
            font=dict(size=18, color=THEME["text_color"]),
            x=0.5,
            xanchor="center",
        ),
        xaxis_title="Runtime (minutes)",
        yaxis_title="Emotion Intensity (+ positive / - negative)",
        xaxis=dict(gridcolor="#1E293B", color=THEME["text_color"]),
        yaxis=dict(
            gridcolor="#1E293B",
            color=THEME["text_color"],
            range=[y_axis_min, y_axis_max],
            zeroline=True,
            zerolinecolor="gray",
            zerolinewidth=1,
            tickformat=".3f",
        ),
        plot_bgcolor=THEME["background_color"],
        paper_bgcolor=THEME["background_color"],
        font=dict(color=THEME["text_color"], family=THEME["font_body"]),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(30, 41, 59, 0.8)",
            bordercolor=THEME["primary_color"],
            borderwidth=1,
        ),
        hovermode="x unified",
        height=500,
    )

    return fig


def plot_emotion_composition(df: pd.DataFrame, film_title: str) -> go.Figure:
    """
    Create stacked area chart showing emotion intensity over time.
    Negative emotions displayed below zero axis, positive above.
    Shows ALL emotions except neutral (not just top 7).

    Args:
        df: DataFrame with minute_offset and emotion_* columns
        film_title: Film title for chart title

    Returns:
        Plotly Figure object

    [Source: Story 5.3 - Task 2.2, AC3 + Story 3.5 styling]
    """
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No emotion data available",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=16, color=THEME["text_color"]),
        )
        fig.update_layout(
            plot_bgcolor=THEME["background_color"],
            paper_bgcolor=THEME["background_color"],
            height=400,
        )
        return fig

    # Get ALL emotions (excluding neutral/ambiguous per Epic 3.5 guidelines)
    all_cols = [col for col in df.columns if col.startswith("emotion_")]
    all_emotions = [
        col.replace("emotion_", "")
        for col in all_cols
        if col.replace("emotion_", "") not in EXCLUDED_EMOTIONS
    ]

    # Separate positive and negative emotions
    positive_emotions = [e for e in all_emotions if e not in NEGATIVE_EMOTIONS]
    negative_emotions = [e for e in all_emotions if e in NEGATIVE_EMOTIONS]

    # Use same emotion colors as timeline for consistency
    emotion_colors = {
        "joy": "#FFD700",
        "admiration": "#EC4899",
        "amusement": "#F59E0B",
        "love": "#F472B6",
        "excitement": "#FBBF24",
        "gratitude": "#34D399",
        "optimism": "#60A5FA",
        "caring": "#A78BFA",
        "approval": "#4ADE80",
        "pride": "#FB923C",
        "relief": "#22D3EE",
        "fear": "#9333EA",
        "sadness": "#3B82F6",
        "anger": "#EF4444",
        "disappointment": "#6366F1",
        "disgust": "#DC2626",
        "grief": "#1E3A8A",
        "embarrassment": "#DB2777",
        "nervousness": "#7C3AED",
        "annoyance": "#F97316",
        "disapproval": "#7C2D12",
        "remorse": "#831843",
    }

    # Create stacked area chart
    fig = go.Figure()

    # Add positive emotions (stacked above zero) - use stackgroup for proper cumulative stacking
    for i, emotion in enumerate(positive_emotions):
        col_name = f"emotion_{emotion}"
        color = emotion_colors.get(emotion, "#94A3B8")

        fig.add_trace(
            go.Scatter(
                x=df["minute_offset"],
                y=df[col_name],
                mode="lines",
                name=emotion.capitalize(),
                stackgroup="positive",  # This ensures proper cumulative stacking
                line=dict(width=0.5, color=color),
                fillcolor=color,
                hovertemplate=f"<b>{emotion.capitalize()}</b><br>Minute: %{{x}}<br>Intensity: %{{y:.3f}}<extra></extra>",
            )
        )

    # Add negative emotions (stacked below zero) - invert values
    for i, emotion in enumerate(negative_emotions):
        col_name = f"emotion_{emotion}"
        color = emotion_colors.get(emotion, "#94A3B8")

        # Invert for negative display
        y_values = -df[col_name]

        fig.add_trace(
            go.Scatter(
                x=df["minute_offset"],
                y=y_values,
                mode="lines",
                name=f"{emotion.capitalize()} (negative)",
                stackgroup="negative",  # Separate stackgroup for negative emotions
                line=dict(width=0.5, color=color),
                fillcolor=color,
                hovertemplate=f"<b>{emotion.capitalize()} (negative)</b><br>Minute: %{{x}}<br>Intensity: %{{y:.3f}}<extra></extra>",
            )
        )

    # Add zero baseline
    fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1, opacity=0.5)

    # Calculate dynamic y-axis range from ACTUAL stacked data
    # Calculate row-wise sums to get true stacked maximums
    if positive_emotions:
        positive_cols = [f"emotion_{e}" for e in positive_emotions if f"emotion_{e}" in df.columns]
        max_positive = df[positive_cols].sum(axis=1).max() if positive_cols else 0.1
    else:
        max_positive = 0.1

    if negative_emotions:
        negative_cols = [f"emotion_{e}" for e in negative_emotions if f"emotion_{e}" in df.columns]
        max_negative = df[negative_cols].sum(axis=1).max() if negative_cols else 0.1
    else:
        max_negative = 0.1

    # Add minimal padding (5%) for tighter fit
    y_max = max_positive * 1.05
    y_min = -max_negative * 1.05

    # Add positive zone shading (green, above zero)
    fig.add_hrect(y0=0, y1=y_max, fillcolor="green", opacity=0.05, layer="below", line_width=0)

    # Add negative zone shading (red, below zero)
    fig.add_hrect(y0=y_min, y1=0, fillcolor="red", opacity=0.05, layer="below", line_width=0)

    fig.update_layout(
        title=dict(
            text=f"{film_title} - Emotion Composition",
            font=dict(size=18, color=THEME["text_color"]),
            x=0.5,
            xanchor="center",
        ),
        xaxis_title="Runtime (minutes)",
        yaxis_title="Emotion Intensity (+ positive / - negative)",
        xaxis=dict(gridcolor="#1E293B", color=THEME["text_color"]),
        yaxis=dict(
            gridcolor="#1E293B",
            color=THEME["text_color"],
            range=[y_min, y_max],  # Dynamic range based on data
            zeroline=True,
            zerolinecolor="gray",
            zerolinewidth=1,
        ),
        plot_bgcolor=THEME["background_color"],
        paper_bgcolor=THEME["background_color"],
        font=dict(color=THEME["text_color"], family=THEME["font_body"]),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.2,  # Position below the chart
            xanchor="center",
            x=0.5,  # Center horizontally
            bgcolor="rgba(30, 41, 59, 0.8)",
            bordercolor=THEME["primary_color"],
            borderwidth=1,
        ),
        hovermode="x unified",
        height=500,  # Increased to accommodate legend below
        margin=dict(b=120),  # Add bottom margin for legend
    )

    return fig


def plot_emotional_fingerprint(
    emotion_summaries: List[Tuple[str, dict]], comparison_mode: bool = False
) -> go.Figure:
    """
    Create radar chart showing overall emotion profile with optional multi-film comparison.

    Uses FIXED axes (all 22 relevant emotions) so polygon shape stays consistent when comparing films.

    Args:
        emotion_summaries: List of tuples (film_title, emotion_summary_dict)
        comparison_mode: If True, overlay multiple films for comparison

    Returns:
        Plotly Figure object

    [Source: Story 5.3 - Task 2.3, AC4 + Enhancement]
    """
    if not emotion_summaries:
        fig = go.Figure()
        fig.add_annotation(
            text="No emotion summary available",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=16, color=THEME["text_color"]),
        )
        fig.update_layout(
            plot_bgcolor=THEME["background_color"],
            paper_bgcolor=THEME["background_color"],
            height=600,
        )
        return fig

    # FIXED emotion axes - all 22 relevant emotions (excludes neutral/ambiguous per Epic 3.5)
    # This ensures polygon shape stays consistent when comparing films
    ALL_EMOTIONS = POSITIVE_EMOTIONS + NEGATIVE_EMOTIONS
    fixed_labels = [e.capitalize() for e in ALL_EMOTIONS]

    # Color palette for multiple films
    colors = ["#38BDF8", "#F59E0B", "#EC4899", "#34D399", "#A78BFA"]

    fig = go.Figure()
    all_values = []

    for idx, (film_title, emotion_summary) in enumerate(emotion_summaries):
        if not emotion_summary:
            continue

        # Get values for ALL emotions (in fixed order), defaulting to 0 if not present
        values = [emotion_summary.get(emotion, 0) for emotion in ALL_EMOTIONS]
        all_values.extend(values)

        color = colors[idx % len(colors)]

        fig.add_trace(
            go.Scatterpolar(
                r=values,
                theta=fixed_labels,
                fill="toself",
                fillcolor=color,
                opacity=0.2 if comparison_mode else 0.3,
                line=dict(color=color, width=2),
                name=film_title,
            )
        )

    # Title based on mode
    if comparison_mode and len(emotion_summaries) > 1:
        title_text = "Emotional Fingerprint Comparison"
    else:
        title_text = f"{emotion_summaries[0][0]} - Emotional Fingerprint"

    fig.update_layout(
        title=dict(
            text=title_text, font=dict(size=20, color=THEME["text_color"]), x=0.5, xanchor="center"
        ),
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max(all_values) * 1.2] if all_values else [0, 1],
                gridcolor="#1E293B",
                color=THEME["text_color"],
            ),
            angularaxis=dict(gridcolor="#1E293B", color=THEME["text_color"], rotation=90),
            bgcolor=THEME["background_color"],
        ),
        plot_bgcolor=THEME["background_color"],
        paper_bgcolor=THEME["background_color"],
        font=dict(color=THEME["text_color"], family=THEME["font_body"]),
        height=600,  # Increased from 400
        showlegend=comparison_mode and len(emotion_summaries) > 1,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.15,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(30, 41, 59, 0.8)",
            bordercolor=THEME["primary_color"],
            borderwidth=1,
        ),
    )

    return fig


# ============================================================================
# Epic 5.4: Director Profiles & Cross-Film Analysis Visualizations
# ============================================================================


def plot_director_emotion_radar(
    director_profiles: List[dict], comparison_mode: bool = False
) -> go.Figure:
    """
    Create radar chart showing director emotion profiles with optional multi-director overlay.

    Follows Epic 3.5 guidelines: Excludes 6 neutral/ambiguous emotions (confusion, curiosity,
    desire, realization, surprise, neutral), showing only 22 relevant emotions.

    Args:
        director_profiles: List of director profile dicts from get_director_profile()
        comparison_mode: If True, overlay multiple directors with distinct colors

    Returns:
        Plotly Figure object with radar chart

    [Source: Story 5.4 - Task 2.1, AC2]
    """
    # Define 22 relevant emotions (alphabetically sorted for consistent axes)
    relevant_emotions = [
        "admiration",
        "amusement",
        "anger",
        "annoyance",
        "approval",
        "caring",
        "disappointment",
        "disapproval",
        "disgust",
        "embarrassment",
        "excitement",
        "fear",
        "gratitude",
        "grief",
        "joy",
        "love",
        "nervousness",
        "optimism",
        "pride",
        "relief",
        "remorse",
        "sadness",
    ]

    # Color palette for multi-director comparison
    color_palette = px.colors.qualitative.Plotly  # Plotly default palette

    fig = go.Figure()

    # Collect all r_values to determine dynamic range
    all_r_values = []

    # Add trace for each director
    for idx, profile in enumerate(director_profiles):
        # Extract emotion values in fixed order
        r_values = [profile.get(f"avg_emotion_{emotion}", 0) for emotion in relevant_emotions]
        all_r_values.extend(r_values)

        # Determine styling based on primary vs comparison directors
        is_primary = idx == 0
        color = color_palette[idx % len(color_palette)]

        fig.add_trace(
            go.Scatterpolar(
                r=r_values,
                theta=[e.capitalize() for e in relevant_emotions],  # Capitalize for display
                name=profile["director"],
                fill="toself" if is_primary else "none",  # Only fill primary director
                fillcolor=(
                    f"rgba{tuple(list(px.colors.hex_to_rgb(color)) + [0.4])}"
                    if is_primary
                    else None
                ),
                line=dict(color=color, width=2),
                marker=dict(size=4, color=color),
            )
        )

    # Calculate dynamic range based on actual data with 20% buffer
    max_value = max(all_r_values) if all_r_values else 0.1
    range_max = max_value * 1.2  # Add 20% buffer above max value

    # Layout configuration
    fig.update_layout(
        title=dict(
            text=(
                "Director Signature Emotion Profile"
                if not comparison_mode
                else "Director Emotion Comparison"
            ),
            font=dict(size=20, color=THEME["text_color"], family=THEME["font_headers"]),
        ),
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, range_max],  # Dynamic range based on actual data
                gridcolor="#1E293B",
                color=THEME["text_color"],
            ),
            angularaxis=dict(gridcolor="#1E293B", color=THEME["text_color"]),
            bgcolor=THEME["background_color"],
        ),
        plot_bgcolor=THEME["background_color"],
        paper_bgcolor=THEME["background_color"],
        font=dict(color=THEME["text_color"], family=THEME["font_body"]),
        height=700,
        showlegend=comparison_mode and len(director_profiles) > 1,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.15,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(30, 41, 59, 0.8)",
            bordercolor=THEME["primary_color"],
            borderwidth=1,
        ),
    )

    return fig


def plot_career_evolution(df: pd.DataFrame, director: str, top_emotions: List[str]) -> go.Figure:
    """
    Create line chart showing emotion trends across director's career.

    Args:
        df: DataFrame with columns: film_title, release_year, emotion_* (from get_director_career_evolution())
        director: Director name for chart title
        top_emotions: List of emotions to plot (typically top 5 from director profile)

    Returns:
        Plotly Figure object with line chart

    [Source: Story 5.4 - Task 2.2, AC3]
    """
    # Emotion colors (reuse from plot_emotion_preview)
    emotion_colors = {
        "joy": "#FFD700",
        "admiration": "#EC4899",
        "amusement": "#F59E0B",
        "love": "#F472B6",
        "excitement": "#FBBF24",
        "gratitude": "#34D399",
        "optimism": "#60A5FA",
        "caring": "#A78BFA",
        "approval": "#4ADE80",
        "pride": "#FB923C",
        "relief": "#22D3EE",
        "fear": "#9333EA",
        "sadness": "#3B82F6",
        "anger": "#EF4444",
        "disappointment": "#6366F1",
        "disgust": "#DC2626",
        "grief": "#1E3A8A",
        "embarrassment": "#DB2777",
        "nervousness": "#7C3AED",
        "annoyance": "#F97316",
        "disapproval": "#7C2D12",
        "remorse": "#831843",
    }

    fig = go.Figure()

    # Add line trace for each emotion
    for emotion in top_emotions:
        col_name = f"emotion_{emotion}"
        if col_name not in df.columns:
            continue

        color = emotion_colors.get(emotion, "#94A3B8")

        fig.add_trace(
            go.Scatter(
                x=df["release_year"],
                y=df[col_name],
                mode="lines+markers",
                name=emotion.capitalize(),
                line=dict(color=color, width=3),
                marker=dict(
                    size=8, color=color, line=dict(width=1, color=THEME["background_color"])
                ),
                hovertemplate=f"<b>{emotion.capitalize()}</b><br>Year: %{{x}}<br>Score: %{{y:.3f}}<br>Film: %{{customdata}}<extra></extra>",
                customdata=df["film_title"],
            )
        )

    # Layout configuration
    fig.update_layout(
        title=dict(
            text=f"{director}'s Career Emotion Evolution",
            font=dict(size=20, color=THEME["text_color"], family=THEME["font_headers"]),
        ),
        xaxis=dict(
            title="Release Year",
            gridcolor="#1E293B",
            color=THEME["text_color"],
            showline=True,
            linecolor="#1E293B",
        ),
        yaxis=dict(
            title="Emotion Score",
            gridcolor="#1E293B",
            color=THEME["text_color"],
            showline=True,
            linecolor="#1E293B",
            range=[
                0,
                max(
                    0.15,
                    df[[f"emotion_{e}" for e in top_emotions if f"emotion_{e}" in df.columns]]
                    .max()
                    .max()
                    * 1.1,
                ),
            ],
        ),
        plot_bgcolor=THEME["background_color"],
        paper_bgcolor=THEME["background_color"],
        font=dict(color=THEME["text_color"], family=THEME["font_body"]),
        height=500,
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
            bgcolor="rgba(30, 41, 59, 0.8)",
            bordercolor=THEME["primary_color"],
            borderwidth=1,
        ),
        hovermode="closest",
    )

    return fig


def plot_film_similarity_heatmap(df: pd.DataFrame, language_code: str) -> go.Figure:
    """
    Create heatmap showing emotional similarity between all films.

    Args:
        df: DataFrame with columns: film_title_a, film_title_b, similarity_score
            (from get_film_similarity_matrix())
        language_code: Language code for chart subtitle

    Returns:
        Plotly Figure object with heatmap

    [Source: Story 5.4 - Task 2.3, AC4]
    """
    # Get unique films and sort alphabetically (filter out None values)
    films_a = df["film_title_a"].unique()
    films_b = df["film_title_b"].unique()
    all_films_raw = set(list(films_a) + list(films_b))
    # Filter out None/NaN values before sorting
    all_films = sorted([f for f in all_films_raw if f is not None and pd.notna(f)])

    # Truncate long titles more aggressively for readability
    film_labels = []
    for title in all_films:
        if len(title) > 25:
            # Truncate and add ellipsis
            film_labels.append(title[:25] + "...")
        else:
            film_labels.append(title)

    # Create NxN matrix with similarity scores
    # Initialize with zeros
    n = len(all_films)
    matrix = [[0.0 for _ in range(n)] for _ in range(n)]

    # Fill matrix with similarity scores
    film_to_idx = {film: idx for idx, film in enumerate(all_films)}

    for _, row in df.iterrows():
        title_a = row["film_title_a"]
        title_b = row["film_title_b"]
        score = row["similarity_score"]

        idx_a = film_to_idx.get(title_a)
        idx_b = film_to_idx.get(title_b)

        if idx_a is not None and idx_b is not None:
            # Symmetric matrix (similarity is bidirectional)
            matrix[idx_a][idx_b] = score
            matrix[idx_b][idx_a] = score

    # Fill diagonal with 1.0 (self-similarity)
    for i in range(n):
        matrix[i][i] = 1.0

    # Create heatmap matching Epic 3.5 style with better text contrast
    # Use custom colorscale that provides better contrast with text
    custom_colorscale = [
        [0.0, "#8B0000"],  # Dark red for 0%
        [0.25, "#CD5C5C"],  # Light red for 25%
        [0.5, "#FFFF99"],  # Light yellow for 50% (mid-point)
        [0.75, "#6495ED"],  # Light blue for 75%
        [1.0, "#00008B"],  # Dark blue for 100%
    ]

    fig = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=film_labels,
            y=film_labels,
            colorscale=custom_colorscale,  # Custom scale with better text contrast
            zmid=0.5,  # Center color scale at 50%
            zmin=0,
            zmax=1,
            text=[[f"{val*100:.0f}%" for val in row] for row in matrix],  # Show percentage in cells
            texttemplate="%{text}",
            textfont={"size": 10, "color": "black"},  # Black text for maximum contrast
            hovertemplate="<b>%{y}</b> vs <b>%{x}</b><br>Similarity: %{z:.1%}<extra></extra>",
            colorbar=dict(
                title=dict(
                    text="Emotional<br>Similarity",
                    font=dict(size=14, color=THEME["text_color"]),
                    side="right",
                ),
                tickformat=".0%",
                tickfont=dict(color=THEME["text_color"], size=12),
                thickness=20,
                len=0.7,
            ),
        )
    )

    # Layout configuration matching Epic 3.5 style
    fig.update_layout(
        title=dict(
            text=f"Film Emotional Similarity Matrix - {language_code.upper()}<br><sub>(Neutral emotion excluded for clarity)</sub>",
            x=0.5,
            xanchor="center",
            font=dict(size=20, color=THEME["text_color"], family=THEME["font_headers"]),
        ),
        xaxis=dict(
            title="",
            tickangle=-45,
            side="bottom",
            color=THEME["text_color"],
            tickfont=dict(size=10),
            showgrid=False,
        ),
        yaxis=dict(title="", color=THEME["text_color"], tickfont=dict(size=10), showgrid=False),
        plot_bgcolor=THEME["background_color"],
        paper_bgcolor=THEME["background_color"],
        font=dict(color=THEME["text_color"], family=THEME["font_body"]),
        height=700,  # Fixed height like Epic 3.5
        width=900,  # Fixed width for better readability
        margin=dict(l=150, r=100, t=120, b=150),
    )

    return fig


# ============================================================================
# Epic 5.5: Cross-Language Insights Visualizations
# ============================================================================


def plot_divergence_bar_chart(
    df: pd.DataFrame, language_a_name: str, language_b_name: str
) -> go.Figure:
    """
    Create horizontal bar chart showing translation biases.

    Positive bars (Lang B > Lang A): Extend right, cyan color
    Negative bars (Lang A > Lang B): Extend left, gold color

    Args:
        df: DataFrame with columns: emotion_type, percent_difference, avg_score_lang_a, avg_score_lang_b
        language_a_name: Display name for language A (e.g., "Arabic")
        language_b_name: Display name for language B (e.g., "English")

    Returns:
        Plotly Figure object

    [Source: Story 5.5 - Task 2.1, AC2]
    """
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No translation bias data available",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=16, color=THEME["text_color"]),
        )
        fig.update_layout(
            plot_bgcolor=THEME["background_color"],
            paper_bgcolor=THEME["background_color"],
            height=400,
        )
        return fig

    # Sort by absolute percent difference (largest first)
    df_sorted = df.sort_values("percent_difference", key=abs, ascending=True)

    # Color based on sign: cyan for positive (Lang B > Lang A), gold for negative
    colors = [
        THEME["primary_color"] if x > 0 else THEME["accent_color"]
        for x in df_sorted["percent_difference"]
    ]

    fig = go.Figure(
        go.Bar(
            y=df_sorted["emotion_type"].str.capitalize(),
            x=df_sorted["percent_difference"],
            orientation="h",
            marker=dict(color=colors, line=dict(color="white", width=1)),
            hovertemplate=(
                "<b>%{y}</b><br>"
                + f"{language_a_name}: %{{customdata[0]:.4f}}<br>"
                + f"{language_b_name}: %{{customdata[1]:.4f}}<br>"
                + "Difference: %{x:.1f}%<br>"
                + "<extra></extra>"
            ),
            customdata=df_sorted[["avg_score_lang_a", "avg_score_lang_b"]].values,
        )
    )

    # Add vertical zero line
    fig.add_vline(x=0, line_dash="dash", line_color="white", line_width=2)

    # Calculate x-axis range symmetrically
    max_abs = max(abs(df_sorted["percent_difference"].max()), abs(df_sorted["percent_difference"].min()))
    x_range = [-max_abs * 1.1, max_abs * 1.1]

    fig.update_layout(
        title=dict(
            text=f"Translation Bias: {language_a_name} → {language_b_name}",
            font=dict(size=18, color=THEME["text_color"]),
            x=0.5,
            xanchor="center",
        ),
        xaxis=dict(
            title="Percent Difference (%)",
            gridcolor="#1E293B",
            color=THEME["text_color"],
            zeroline=True,
            zerolinewidth=2,
            zerolinecolor="white",
            range=x_range,
        ),
        yaxis=dict(
            title="",
            gridcolor="#1E293B",
            color=THEME["text_color"],
        ),
        plot_bgcolor=THEME["background_color"],
        paper_bgcolor=THEME["background_color"],
        font=dict(color=THEME["text_color"], family=THEME["font_body"]),
        height=max(400, len(df_sorted) * 40),  # Dynamic height based on rows
        showlegend=False,
        margin=dict(l=120),  # Extra left margin for emotion labels
    )

    return fig


def plot_consistency_heatmap(matrix_df: pd.DataFrame, film_title: Optional[str] = None) -> go.Figure:
    """
    Create 5x5 heatmap of consistency scores across language pairs.

    Higher scores (green): More consistent translations (similar emotion patterns)
    Lower scores (red): More divergent translations (different emotion patterns)

    Args:
        matrix_df: DataFrame with languages as index/columns, values are consistency scores (0-100%)
        film_title: Optional film title for chart subtitle (None = "All Films")

    Returns:
        Plotly Figure object

    [Source: Story 5.5 - Task 2.2, AC3]
    """
    if matrix_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No consistency data available",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=16, color=THEME["text_color"]),
        )
        fig.update_layout(
            plot_bgcolor=THEME["background_color"],
            paper_bgcolor=THEME["background_color"],
            height=500,
        )
        return fig

    # Get language labels from DataFrame index
    languages = list(matrix_df.index)

    # Create heatmap with RdYlGn (green = high/consistent, red = low/divergent)
    fig = go.Figure(
        data=go.Heatmap(
            z=matrix_df.values,
            x=languages,
            y=languages,
            colorscale="RdYlGn",  # Green (high) = consistent, Red (low) = divergent
            zmin=0,
            zmax=100,  # Percent scale
            text=[[f"{val:.1f}%" for val in row] for row in matrix_df.values],
            texttemplate="%{text}",
            textfont={"size": 14, "color": "black"},  # Black text for better readability on RdYlGn
            hovertemplate="<b>%{y}</b> vs <b>%{x}</b><br>Consistency: %{z:.1f}%<extra></extra>",
            colorbar=dict(
                title=dict(text="Consistency<br>Score", font=dict(size=12, color=THEME["text_color"])),
                ticksuffix="%",
                tickfont=dict(color=THEME["text_color"]),
            ),
        )
    )

    # Title with optional film name
    subtitle = f" ({film_title})" if film_title else " (All Films)"

    fig.update_layout(
        title=dict(
            text=f"Cross-Language Consistency Matrix{subtitle}",
            font=dict(size=18, color=THEME["text_color"]),
            x=0.5,
            xanchor="center",
        ),
        xaxis=dict(
            title="Language",
            side="bottom",
            color=THEME["text_color"],
            tickfont=dict(size=12),
        ),
        yaxis=dict(
            title="Language",
            color=THEME["text_color"],
            tickfont=dict(size=12),
        ),
        plot_bgcolor=THEME["background_color"],
        paper_bgcolor=THEME["background_color"],
        font=dict(color=THEME["text_color"], family=THEME["font_body"]),
        height=500,
        width=600,
    )

    return fig


# ============================================================================
# Epic 5.6: Methodology & Data Quality Page Visualizations
# ============================================================================


def plot_raw_vs_smoothed_comparison(
    raw_df: pd.DataFrame,
    smoothed_df: pd.DataFrame,
    film_title: str,
    window_size: int = 10,
) -> go.Figure:
    """
    Create side-by-side comparison of raw vs smoothed emotion curves.

    Args:
        raw_df: DataFrame with minute_offset, emotion_score (raw data)
        smoothed_df: DataFrame with minute_offset, emotion_score (smoothed)
        film_title: Film title for chart title
        window_size: Window size used for smoothing (for subtitle)

    Returns:
        Plotly Figure with 2 subplots (side-by-side)

    [Source: Story 5.6 - Task 2.1, AC2]
    """
    from plotly.subplots import make_subplots

    # Handle empty data
    if raw_df.empty and smoothed_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No emotion data available",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=16, color=THEME["text_color"]),
        )
        fig.update_layout(
            plot_bgcolor=THEME["background_color"],
            paper_bgcolor=THEME["background_color"],
            height=500,
        )
        return fig

    # Create subplots
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Raw 1-Minute Data (Noisy)", f"Smoothed ({window_size}-min Rolling Avg)"),
        shared_yaxes=True,
        horizontal_spacing=0.08,
    )

    # Left subplot: Raw data (gold/amber - noisy)
    if not raw_df.empty:
        fig.add_trace(
            go.Scatter(
                x=raw_df["minute_offset"],
                y=raw_df["emotion_score"],
                mode="lines+markers",
                name="Raw",
                line=dict(color=THEME["accent_color"], width=1),  # Gold
                marker=dict(size=3, color=THEME["accent_color"]),
                hovertemplate="<b>Raw</b><br>Minute: %{x}<br>Score: %{y:.4f}<extra></extra>",
            ),
            row=1,
            col=1,
        )

    # Right subplot: Smoothed data (cyan - clean)
    if not smoothed_df.empty:
        fig.add_trace(
            go.Scatter(
                x=smoothed_df["minute_offset"],
                y=smoothed_df["emotion_score"],
                mode="lines",
                name="Smoothed",
                line=dict(color=THEME["primary_color"], width=2),  # Cyan
                hovertemplate=f"<b>Smoothed ({window_size}min)</b><br>Minute: %{{x}}<br>Score: %{{y:.4f}}<extra></extra>",
            ),
            row=1,
            col=2,
        )

    # Calculate shared y-axis range
    all_values = []
    if not raw_df.empty:
        all_values.extend(raw_df["emotion_score"].tolist())
    if not smoothed_df.empty:
        all_values.extend(smoothed_df["emotion_score"].tolist())

    if all_values:
        y_min = min(all_values)
        y_max = max(all_values)
        y_padding = (y_max - y_min) * 0.1
        y_range = [max(0, y_min - y_padding), y_max + y_padding]
    else:
        y_range = [0, 0.1]

    # Update layout
    fig.update_xaxes(title_text="Runtime (minutes)", gridcolor="#1E293B", color=THEME["text_color"])
    fig.update_yaxes(
        title_text="Joy Score",
        gridcolor="#1E293B",
        color=THEME["text_color"],
        range=y_range,
        row=1,
        col=1,
    )

    fig.update_layout(
        title=dict(
            text=f"{film_title} - Signal Processing Comparison",
            font=dict(size=18, color=THEME["text_color"]),
            x=0.5,
            xanchor="center",
        ),
        plot_bgcolor=THEME["background_color"],
        paper_bgcolor=THEME["background_color"],
        font=dict(color=THEME["text_color"], family=THEME["font_body"]),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.05,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(30, 41, 59, 0.8)",
        ),
        height=450,
        margin=dict(t=100),
    )

    # Style subplot titles
    for annotation in fig["layout"]["annotations"]:
        annotation["font"] = dict(size=14, color=THEME["text_color"])

    return fig


def plot_peak_count_comparison(df: pd.DataFrame, current_window: int = 10) -> go.Figure:
    """
    Create bar chart showing peak counts across different window sizes.

    Args:
        df: DataFrame with window_size_minutes, peak_count columns
        current_window: Currently selected window size (for highlighting)

    Returns:
        Plotly Figure with bar chart

    [Source: Story 5.6 - Task 2.2, AC3]
    """
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No methodology data available",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=16, color=THEME["text_color"]),
        )
        fig.update_layout(
            plot_bgcolor=THEME["background_color"],
            paper_bgcolor=THEME["background_color"],
            height=400,
        )
        return fig

    # Color bars: cyan for selected, gold for recommended (10min), gray for others
    colors = []
    for _, row in df.iterrows():
        ws = int(row["window_size_minutes"])
        if ws == current_window:
            colors.append(THEME["primary_color"])  # Selected - cyan
        elif row.get("is_recommended", False):
            colors.append("#10B981")  # Recommended - green
        else:
            colors.append("#64748B")  # Others - slate

    fig = go.Figure(
        go.Bar(
            x=df["window_size_minutes"].astype(str) + " min",
            y=df["peak_count"],
            marker=dict(color=colors, line=dict(color="white", width=1)),
            text=df["peak_count"].astype(int),
            textposition="outside",
            textfont=dict(color=THEME["text_color"], size=14),
            hovertemplate="<b>%{x}</b><br>Peaks Detected: %{y:.0f}<extra></extra>",
        )
    )

    fig.update_layout(
        title=dict(
            text="Peak Detection by Window Size",
            font=dict(size=18, color=THEME["text_color"]),
            x=0.5,
            xanchor="center",
        ),
        xaxis=dict(
            title="Window Size",
            gridcolor="#1E293B",
            color=THEME["text_color"],
        ),
        yaxis=dict(
            title="Number of Peaks Detected",
            gridcolor="#1E293B",
            color=THEME["text_color"],
            range=[0, df["peak_count"].max() * 1.2],
        ),
        plot_bgcolor=THEME["background_color"],
        paper_bgcolor=THEME["background_color"],
        font=dict(color=THEME["text_color"], family=THEME["font_body"]),
        height=400,
        showlegend=False,
    )

    return fig


def plot_language_coverage_heatmap(matrix_df: pd.DataFrame) -> go.Figure:
    """
    Create heatmap showing language coverage across films.

    Args:
        matrix_df: DataFrame with films as index, languages as columns,
                   values: 1 (available), 0 (missing)

    Returns:
        Plotly Figure with heatmap

    [Source: Story 5.6 - Task 2.3, AC5]
    """
    if matrix_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No coverage data available",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=16, color=THEME["text_color"]),
        )
        fig.update_layout(
            plot_bgcolor=THEME["background_color"],
            paper_bgcolor=THEME["background_color"],
            height=600,
        )
        return fig

    # Language display names
    lang_names = {
        "en": "English",
        "fr": "French",
        "es": "Spanish",
        "nl": "Dutch",
        "ar": "Arabic",
    }
    display_cols = [lang_names.get(c, c) for c in matrix_df.columns]

    # Create annotations (✓ or ✗)
    annotations = [[" ✓" if val == 1 else "✗" for val in row] for row in matrix_df.values]

    # Custom colorscale: green for available, red for missing
    colorscale = [[0, "#DC2626"], [1, "#10B981"]]  # Red to Green

    fig = go.Figure(
        data=go.Heatmap(
            z=matrix_df.values,
            x=display_cols,
            y=matrix_df.index.tolist(),
            colorscale=colorscale,
            showscale=False,
            text=annotations,
            texttemplate="%{text}",
            textfont={"size": 14, "color": "white"},  # White on red/green for checkmarks
            hovertemplate="<b>%{y}</b><br>%{x}: %{text}<extra></extra>",
        )
    )

    fig.update_layout(
        title=dict(
            text="Language Coverage by Film",
            font=dict(size=18, color=THEME["text_color"]),
            x=0.5,
            xanchor="center",
        ),
        xaxis=dict(
            title="Language",
            side="bottom",
            color=THEME["text_color"],
            tickfont=dict(size=12),
        ),
        yaxis=dict(
            title="",
            color=THEME["text_color"],
            tickfont=dict(size=10),
            autorange="reversed",  # Films at top
        ),
        plot_bgcolor=THEME["background_color"],
        paper_bgcolor=THEME["background_color"],
        font=dict(color=THEME["text_color"], family=THEME["font_body"]),
        height=max(400, len(matrix_df) * 25),  # Dynamic height
        margin=dict(l=200),  # Room for film titles
    )

    return fig


def plot_methodology_consistency_matrix(matrix_df: pd.DataFrame) -> go.Figure:
    """
    Create 5x5 heatmap of cross-language consistency scores.

    Higher scores (green) = more consistent (similar emotion patterns)
    Lower scores (red) = more divergent (different emotion patterns)

    Args:
        matrix_df: DataFrame with languages as index/columns, consistency scores as values (0-100%)

    Returns:
        Plotly Figure with heatmap

    [Source: Story 5.6 - Task 2.4, AC6]
    """
    if matrix_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No consistency data available",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=16, color=THEME["text_color"]),
        )
        fig.update_layout(
            plot_bgcolor=THEME["background_color"],
            paper_bgcolor=THEME["background_color"],
            height=500,
        )
        return fig

    # Create heatmap with RdYlGn (green=high/consistent, red=low/divergent)
    fig = go.Figure(
        data=go.Heatmap(
            z=matrix_df.values,
            x=matrix_df.columns.tolist(),
            y=matrix_df.index.tolist(),
            colorscale="RdYlGn",
            zmin=0,
            zmax=100,
            text=[[f"{val:.1f}%" for val in row] for row in matrix_df.values],
            texttemplate="%{text}",
            textfont={"size": 14, "color": "black"},  # Black text for better readability on RdYlGn
            hovertemplate="<b>%{y}</b> vs <b>%{x}</b><br>Consistency: %{z:.1f}%<extra></extra>",
            colorbar=dict(
                title=dict(text="Consistency<br>Score", font=dict(size=12, color=THEME["text_color"])),
                ticksuffix="%",
                tickfont=dict(color=THEME["text_color"]),
            ),
        )
    )

    fig.update_layout(
        title=dict(
            text="Cross-Language Consistency Matrix",
            font=dict(size=18, color=THEME["text_color"]),
            x=0.5,
            xanchor="center",
        ),
        xaxis=dict(
            title="Language",
            side="bottom",
            color=THEME["text_color"],
            tickfont=dict(size=12),
        ),
        yaxis=dict(
            title="Language",
            color=THEME["text_color"],
            tickfont=dict(size=12),
        ),
        plot_bgcolor=THEME["background_color"],
        paper_bgcolor=THEME["background_color"],
        font=dict(color=THEME["text_color"], family=THEME["font_body"]),
        height=500,
        width=600,
    )

    return fig


def plot_emotion_by_language(
    df: pd.DataFrame,
    emotion_type: str,
    highlight_languages: Optional[Tuple[str, str]] = None,
) -> go.Figure:
    """
    Create vertical bar chart comparing emotion scores across 5 languages.

    Args:
        df: DataFrame with columns: language_code, language_name, avg_score
        emotion_type: Emotion being compared (for title)
        highlight_languages: Optional tuple of (lang_a, lang_b) to highlight

    Returns:
        Plotly Figure object

    [Source: Story 5.5 - Task 2.3, AC4]
    """
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No emotion data available",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=16, color=THEME["text_color"]),
        )
        fig.update_layout(
            plot_bgcolor=THEME["background_color"],
            paper_bgcolor=THEME["background_color"],
            height=400,
        )
        return fig

    # Sort by language name for consistent ordering
    df_sorted = df.sort_values("language_name")

    # Determine colors - highlight selected languages
    colors = []
    for _, row in df_sorted.iterrows():
        if highlight_languages and row["language_code"] in highlight_languages:
            colors.append(THEME["primary_color"])  # Highlighted - cyan
        else:
            colors.append(THEME["accent_color"])  # Default - gold

    fig = go.Figure(
        go.Bar(
            x=df_sorted["language_name"],
            y=df_sorted["avg_score"],
            marker=dict(
                color=colors,
                line=dict(color="white", width=1),
            ),
            text=[f"{v:.4f}" for v in df_sorted["avg_score"]],
            textposition="outside",
            textfont=dict(color=THEME["text_color"], size=12),
            hovertemplate="<b>%{x}</b><br>Score: %{y:.4f}<extra></extra>",
        )
    )

    # Calculate y-axis range with padding for text
    y_max = df_sorted["avg_score"].max() * 1.2

    fig.update_layout(
        title=dict(
            text=f"{emotion_type.capitalize()} Score by Language",
            font=dict(size=18, color=THEME["text_color"]),
            x=0.5,
            xanchor="center",
        ),
        xaxis=dict(
            title="Language",
            gridcolor="#1E293B",
            color=THEME["text_color"],
        ),
        yaxis=dict(
            title="Average Score",
            gridcolor="#1E293B",
            color=THEME["text_color"],
            range=[0, y_max],
        ),
        plot_bgcolor=THEME["background_color"],
        paper_bgcolor=THEME["background_color"],
        font=dict(color=THEME["text_color"], family=THEME["font_body"]),
        height=400,
        showlegend=False,
    )

    return fig
