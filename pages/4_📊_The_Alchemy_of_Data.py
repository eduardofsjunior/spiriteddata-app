"""
The Alchemy of Data - Methodology & Data Quality Page

Demonstrates signal processing decisions, methodology transparency, and data quality metrics.
This is the CRITICAL portfolio differentiator showing understanding of:
- Rolling window smoothing (noise reduction vs temporal precision trade-offs)
- Peak detection algorithms (smoothed vs raw resolution)
- Data validation processes
- Medallion architecture (Bronze/Silver/Gold data layers)
- ETLT hybrid pipeline (Python for ML, dbt for analytics)

[Source: Story 5.6 - Epic 5: Production-Grade Streamlit Emotion Analysis App]
"""

import streamlit as st
import pandas as pd

# Add parent directory to path for utils imports
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.theme import apply_custom_css, render_header
from utils.config import THEME, DATA_STATS
from utils.data_loader import (
    get_methodology_metrics,
    get_film_list_for_methodology,
    get_emotion_timeseries_for_methodology,
    get_smoothed_timeseries_for_methodology,
    get_dual_resolution_peaks,
    get_validation_summary,
    get_language_coverage_matrix,
    get_cross_language_consistency_summary,
    get_consistency_matrix,
    get_film_list_with_metadata,
    LANGUAGE_NAMES,
)
from utils.visualization import (
    plot_raw_vs_smoothed_comparison,
    plot_peak_count_comparison,
    plot_language_coverage_heatmap,
    plot_methodology_consistency_matrix,
)

# Page configuration
st.set_page_config(
    page_title="The Alchemy of Data",
    page_icon="📊",
    layout="wide",
)
apply_custom_css()

# ============================================================================
# Page Header & Overview (AC1)
# ============================================================================

render_header("📊 The Alchemy of Data", "Methodology Transparency & Data Quality")

# Portfolio callout - why this page matters
st.success(
    """
    **🎯 Portfolio Differentiator**: Most data projects hide their limitations. 
    This page embraces **transparency over perfection** — demonstrating signal processing 
    knowledge, methodology choices, and data quality awareness that separates 
    professional data engineering from hobbyist projects.
    """
)

# Overview card
st.info(
    """
    **What You'll Find Here:**
    - 📈 **Signal Processing**: Interactive comparison of raw vs smoothed emotion curves
    - 🔍 **Data Quality**: Validation metrics, language coverage, cross-language consistency
    - 📚 **Methodology Notes**: Detailed explanations of algorithms and architectural decisions
    
    *"Understanding: Why smoothing? What do we lose? How do we validate quality?"*
    """
)

# ============================================================================
# Navigation Tabs (AC1)
# ============================================================================

tab1, tab2, tab3 = st.tabs([
    "📈 Signal Processing",
    "🔍 Data Quality",
    "📚 Methodology Notes",
])

# ============================================================================
# Tab 1: Signal Processing (AC2, AC3, AC4)
# ============================================================================

with tab1:
    st.markdown("### Rolling Window Analysis")
    st.markdown(
        """
        Emotion scores from dialogue are inherently **noisy** — a single powerful line 
        creates spikes that can obscure narrative-level patterns. Rolling window smoothing 
        reduces this noise, but at the cost of **temporal precision**.
        """
    )

    # Film selector and window size controls
    col_controls1, col_controls2 = st.columns([2, 1])

    with col_controls1:
        # Get films available in methodology mart
        films_list = get_film_list_with_metadata()
        
        # Default to a film with good data
        default_idx = 0
        for i, f in enumerate(films_list):
            if "spirited away" in f["film_title"].lower() and f["language_code"] == "en":
                default_idx = i
                break

        selected_film = st.selectbox(
            "Select Film for Demonstration",
            options=films_list,
            format_func=lambda x: x["display_name"],
            index=default_idx,
            key="methodology_film_select",
        )

    with col_controls2:
        window_size = st.slider(
            "Window Size (minutes)",
            min_value=1,
            max_value=20,
            value=10,
            step=1,
            help="Adjust to see real-time effect on smoothing. Recommended: 10 min.",
        )

    # Extract film_slug and language_code
    film_slug = selected_film["film_slug"]
    language_code = selected_film["language_code"]
    film_title = selected_film["film_title"]

    # Fetch raw and smoothed timeseries
    raw_df = get_emotion_timeseries_for_methodology(film_slug, language_code, "joy")
    smoothed_df = get_smoothed_timeseries_for_methodology(
        film_slug, language_code, "joy", window_size
    )

    # Side-by-side comparison chart (AC2)
    if not raw_df.empty or not smoothed_df.empty:
        fig_comparison = plot_raw_vs_smoothed_comparison(
            raw_df, smoothed_df, film_title, window_size
        )
        st.plotly_chart(fig_comparison, use_container_width=True, config={"displayModeBar": True})
    else:
        st.warning("No emotion data available for this film/language combination.")

    # Metrics display (AC2)
    st.markdown("### Signal Processing Metrics")
    
    # Get methodology metrics for current film
    methodology_df = get_methodology_metrics()  # Aggregated across all films
    
    if not methodology_df.empty:
        # Find metrics for current window size
        current_metrics = methodology_df[
            methodology_df["window_size_minutes"] == window_size
        ]
        
        if not current_metrics.empty:
            metrics_row = current_metrics.iloc[0]
            
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            
            with col_m1:
                is_recommended = metrics_row.get("is_recommended", False)
                st.metric(
                    label="🎚️ Noise Level",
                    value=f"{metrics_row['noise_level']:.6f}",
                    delta="Lower = Smoother" if window_size > 5 else None,
                    delta_color="inverse",
                    help="Standard deviation of second derivative (measures 'jaggedness')",
                )
            
            with col_m2:
                st.metric(
                    label="📍 Peak Count",
                    value=f"{int(metrics_row['peak_count'])}",
                    delta="Fewer false peaks" if window_size > 5 else None,
                    help="Number of local maxima detected (fewer = less noise)",
                )
            
            with col_m3:
                st.metric(
                    label="⏱️ Temporal Loss",
                    value=f"{metrics_row['temporal_precision_loss_pct']:.1f}%",
                    delta="Higher with larger windows",
                    help="Percentage of film duration consumed by window",
                )
            
            with col_m4:
                if is_recommended:
                    st.success("⭐ **Recommended Window**")
                else:
                    st.info(f"Window: {window_size} min")

    # Window Size Comparison Table (AC3)
    st.markdown("### Window Size Comparison")
    
    if not methodology_df.empty:
        # Format for display
        display_df = methodology_df.copy()
        display_df["Recommendation"] = display_df["is_recommended"].apply(
            lambda x: "⭐ Recommended" if x else ""
        )
        display_df["Window Size"] = display_df["window_size_minutes"].astype(str) + " min"
        display_df["Noise Level"] = display_df["noise_level"].apply(lambda x: f"{x:.6f}")
        display_df["Peak Count"] = display_df["peak_count"].astype(int)
        display_df["Temporal Loss %"] = display_df["temporal_precision_loss_pct"].apply(
            lambda x: f"{x:.1f}%"
        )
        
        # Select and reorder columns for display
        display_cols = ["Window Size", "Noise Level", "Peak Count", "Temporal Loss %", "Recommendation"]
        
        st.dataframe(
            display_df[display_cols],
            use_container_width=True,
            hide_index=True,
        )
        
        # Trade-off explanation
        st.markdown(
            """
            **Understanding the Trade-offs:**
            - **Smaller windows (3-5 min)**: More peaks detected, noisier curves, but preserves dialogue-level detail
            - **Larger windows (15-20 min)**: Smoother curves, fewer peaks, but loses temporal precision
            - **Recommended 10-min window**: Balances noise reduction with narrative-level granularity
            """
        )
        
        # Peak count bar chart
        fig_peaks = plot_peak_count_comparison(methodology_df, window_size)
        st.plotly_chart(fig_peaks, use_container_width=True, config={"displayModeBar": True})

    # Dual-Resolution Peaks Explanation (AC4)
    with st.expander("🔍 Why Two Peak Resolutions?"):
        st.markdown(
            """
            ### Smoothed vs Raw Peaks
            
            This project provides **both** smoothed and raw peak catalogs because they serve different purposes:
            
            | Resolution | Window | Best For | Example |
            |------------|--------|----------|---------|
            | **Smoothed Peaks** | 10-min rolling avg | Narrative-level climaxes | "The final battle scene" |
            | **Raw Peaks** | 1-min buckets | Dialogue-level spikes | "A single powerful line" |
            
            **Portfolio Value**: Providing both demonstrates understanding that **scale matters** in time-series analysis.
            The "right" resolution depends on your analysis goal.
            """
        )
        
        # Show example peaks for selected film
        peaks_data = get_dual_resolution_peaks(film_slug, language_code, "joy", 3)
        
        if peaks_data["smoothed"] or peaks_data["raw"]:
            col_peaks1, col_peaks2 = st.columns(2)
            
            with col_peaks1:
                st.markdown("**Smoothed Peaks (Narrative)**")
                if peaks_data["smoothed"]:
                    peaks_smooth_df = pd.DataFrame(peaks_data["smoothed"])
                    peaks_smooth_df["Type"] = "Smoothed"
                    st.dataframe(peaks_smooth_df, hide_index=True)
                else:
                    st.caption("No smoothed peaks available")
            
            with col_peaks2:
                st.markdown("**Raw Peaks (Dialogue)**")
                if peaks_data["raw"]:
                    peaks_raw_df = pd.DataFrame(peaks_data["raw"])
                    peaks_raw_df["Type"] = "Raw"
                    st.dataframe(peaks_raw_df, hide_index=True)
                else:
                    st.caption("No raw peaks available")

# ============================================================================
# Tab 2: Data Quality (AC5, AC6)
# ============================================================================

with tab2:
    st.markdown("### Subtitle Validation Dashboard")
    st.markdown(
        """
        Validation checks whether subtitle timestamps extend beyond expected film runtime 
        (+ 10-minute buffer). This identifies **timing drift** from wrong subtitle versions.
        
        *Only preferred subtitle versions (v2 when available) are validated and shown in the app.*
        """
    )
    
    # Get validation summary
    validation = get_validation_summary()
    
    # Overall metrics (AC5)
    col_v1, col_v2, col_v3, col_v4 = st.columns(4)
    
    with col_v1:
        st.metric(
            label="📊 Films Validated",
            value=validation["total_films"],
            help="Preferred film-language combinations (v2 when available)",
        )
    
    with col_v2:
        st.metric(
            label="✅ Pass Rate",
            value=f"{validation['pass_rate']}%",
            delta=f"{validation['pass_count']} passed",
            delta_color="normal",
        )
    
    with col_v3:
        st.metric(
            label="⚠️ Failures",
            value=validation["fail_count"],
            delta="Timing drift detected",
            delta_color="inverse",
        )
    
    with col_v4:
        st.metric(
            label="🌍 Languages",
            value=len(DATA_STATS["languages"]),
            help="EN, FR, ES, NL, AR",
        )

    # Validation details
    if validation["failures"]:
        st.markdown("### ⚠️ Failed Validations")
        st.markdown(
            """
            These film-language combinations have **timing drift** where subtitle timestamps 
            extend beyond expected film runtime + 10-minute buffer. This typically indicates 
            subtitle files from wrong film versions (theatrical vs extended cut).
            """
        )
        
        failures_df = pd.DataFrame(validation["failures"])
        failures_df["Language"] = failures_df["language_code"].map(LANGUAGE_NAMES)
        failures_df["Overrun"] = failures_df["overrun_minutes"].apply(lambda x: f"{x:.0f} min")
        
        display_fail_df = failures_df[["film_title", "Language", "Overrun", "validation_status"]]
        display_fail_df.columns = ["Film", "Language", "Timing Overrun", "Status"]
        
        st.dataframe(display_fail_df, hide_index=True, use_container_width=True)

    # Language Coverage Heatmap (AC5)
    st.markdown("### Language Coverage Matrix")
    st.markdown(
        """
        Shows which films have emotion data in each of the 5 supported languages.
        ✓ = Available, ✗ = Missing
        """
    )
    
    coverage_matrix = get_language_coverage_matrix()
    
    if not coverage_matrix.empty:
        fig_coverage = plot_language_coverage_heatmap(coverage_matrix)
        st.plotly_chart(fig_coverage, use_container_width=True, config={"displayModeBar": True})
    else:
        st.warning("Language coverage data not available.")

    # Cross-Language Consistency (AC6)
    st.markdown("### Cross-Language Consistency")
    st.markdown(
        """
        Measures how similar emotion patterns are across different subtitle translations.
        **Higher scores = more consistent** (translations preserve emotional content).
        **Lower scores = more divergent** (translation choices affect emotion detection).
        """
    )

    consistency_matrix = get_consistency_matrix()

    if not consistency_matrix.empty:
        # Invert matrix values so higher = more consistent (100 - divergence)
        consistency_matrix_inverted = 100 - consistency_matrix
        fig_consistency = plot_methodology_consistency_matrix(consistency_matrix_inverted)
        st.plotly_chart(fig_consistency, use_container_width=True, config={"displayModeBar": True})

        # Interpretation guide
        st.markdown(
            """
            **Interpreting Consistency Scores:**
            - 🟢 **> 80%**: Highly consistent — emotion patterns very similar across translations
            - 🟡 **60-80%**: Moderately consistent — some translation effects visible
            - 🔴 **< 60%**: Divergent — significant differences (translation style, cultural adaptation, or quality issues)

            **Note**: The emotion model is trained primarily on English, which may introduce baseline bias for non-English languages.
            """
        )
    else:
        st.warning("Consistency data not available.")

# ============================================================================
# Tab 3: Methodology Notes (AC7)
# ============================================================================

with tab3:
    st.markdown("### Technical Documentation")
    st.markdown(
        """
        Deep-dive explanations for technical interviews and portfolio review.
        Each section answers common questions about methodology and architecture.
        """
    )

    # Rolling Window Smoothing (AC7.1)
    with st.expander("📈 Rolling Window Smoothing Explained"):
        st.markdown(
            """
            ### What is Rolling Window Smoothing?
            
            A **moving average** applied over a time window to reduce noise in time-series data.
            
            **Formula:**
            ```
            smoothed_value[t] = AVG(values[t - n : t + n])
            ```
            Where `n` = half-window size (for 10-min window, n = 5)
            
            ### Why Is It Needed?
            
            Raw emotion scores from dialogue analysis are inherently **noisy**:
            - A single powerful line creates a spike
            - Technical dialogue (names, actions) has low emotional content
            - Short scenes create rapid fluctuations
            
            ### Trade-offs
            
            | Benefit | Cost |
            |---------|------|
            | Smoother, cleaner curves | Temporal precision loss |
            | Easier to identify narrative patterns | Blurs exact timing of emotional moments |
            | Reduces false peak detection | May merge distinct emotional events |
            
            ### Implementation (dbt SQL)
            
            ```sql
            AVG(emotion_joy) OVER (
                PARTITION BY film_id, language_code
                ORDER BY minute_offset
                ROWS BETWEEN 4 PRECEDING AND 5 FOLLOWING  -- 10-minute window
            ) AS smoothed_joy
            ```
            """
        )

    # Peak Detection (AC7.2)
    with st.expander("📍 How Peaks Are Detected"):
        st.markdown(
            """
            ### Peak Detection Algorithm
            
            **Definition**: A local maximum where `value > previous AND value > next`
            
            **Why Peaks Matter:**
            - Identify narrative climaxes (emotional high points)
            - Find memorable moments for close reading
            - Compare emotional intensity across films
            
            ### False Positives Problem
            
            Raw (unsmoothed) data produces **many noise-induced peaks**:
            - Random fluctuations appear as peaks
            - Makes it hard to distinguish true emotional climaxes from noise
            
            ### Solution: Smoothed Peak Detection
            
            1. Apply rolling window smoothing (10-min window)
            2. Detect local maxima on smoothed curve
            3. Result: Fewer, more meaningful peaks representing narrative-level climaxes
            
            **Expected Pattern:**
            - Raw data: 15-25 peaks per film
            - Smoothed data: 5-10 peaks per film (2-3x reduction in false positives)
            """
        )

    # Subtitle Validation (AC7.3)
    with st.expander("✅ How We Validate Data Quality"):
        st.markdown(
            """
            ### Validation Process
            
            **1. Timing Drift Detection**
            - Compare max subtitle timestamp to expected film runtime
            - Buffer: 10 minutes (accounts for credits, variations)
            - FAIL if: `max_minute_offset > expected_duration + 10`
            
            **2. Cross-Language Consistency**
            - Calculate emotion pattern similarity across translations
            - Flag significant divergence (<60% consistency)
            
            **3. Manual Spot Checks**
            - Sample dialogues for obvious quality issues
            - Check encoding problems, missing sections
            
            ### Quality Discovery & Improvement
            
            During initial validation, we discovered that many subtitle files from 
            public sources had **timing drift** — timestamps extending far beyond 
            the actual film runtime. This indicated wrong film versions (theatrical 
            vs extended cuts) or incorrectly synchronized subtitles.
            
            **Solution Implemented:**
            - Developed runtime-based validation using official film durations
            - Identified priority films requiring improved subtitle sources
            - Acquired higher-quality subtitle files (v2 versions)
            - Re-ran emotion analysis pipeline on corrected data
            
            **Result:** Pass rate improved from ~40% to 85.5% after targeted improvements,
            with clear documentation of which film-language combinations have validated data.
            """
        )

    # Medallion Architecture & ETLT Pipeline (AC7.4 - NEW)
    with st.expander("🏗️ Architecture: Medallion + ETLT Pipeline"):
        st.markdown(
            """
            ### Medallion Architecture (Bronze → Silver → Gold)
            
            This project uses industry-standard **Medallion Architecture** for data layering:
            
            | Layer | Schema | Purpose | Tool |
            |-------|--------|---------|------|
            | **🥉 Bronze** | `raw.*` | Raw extracted data | Python |
            | **🥈 Silver** | `main_staging.*` | Cleaned, typed, standardized | dbt |
            | **🥇 Gold** | `main_marts.*` | Analytics-ready aggregations | dbt |
            
            ### ETLT Pipeline Flow
            
            This project uses a **hybrid ETLT** (Extract-Transform-Load-Transform) approach:
            
            ```
            ┌─────────────────────────────────────────────────────────────┐
            │  Python (E+T₁)           │  DuckDB (L)  │  dbt (T₂+L)      │
            ├─────────────────────────────────────────────────────────────┤
            │  1. Parse .srt files     │              │                  │
            │  2. Run HuggingFace      │  3. Write to │  4. Staging      │
            │     emotion model        │     raw.*    │     models       │
            │  3. Aggregate to         │              │  5. Intermediate │
            │     1-min buckets        │              │  6. Mart models  │
            └─────────────────────────────────────────────────────────────┘
            ```
            
            ### Why This Hybrid Approach?
            
            | Task | Best Tool | Reason |
            |------|-----------|--------|
            | ML model inference | **Python** | HuggingFace transformers, GPU/CPU inference |
            | Subtitle parsing | **Python** | pysrt library, complex text handling |
            | Rolling windows | **dbt (SQL)** | Window functions, declarative, testable |
            | Aggregations | **dbt (SQL)** | GROUP BY, HAVING, built-in testing |
            | Peak detection | **dbt (SQL)** | LAG/LEAD functions, row numbering |
            
            ### Code Example: Rolling Window in dbt
            
            ```sql
            -- From mart_film_emotion_timeseries.sql
            SELECT
                film_slug,
                minute_offset,
                AVG(emotion_joy) OVER (
                    PARTITION BY film_slug, language_code
                    ORDER BY minute_offset
                    ROWS BETWEEN 4 PRECEDING AND 5 FOLLOWING
                ) AS emotion_joy
            FROM {{ ref('stg_film_emotions') }}
            ```
            
            ### Portfolio Value
            
            This architecture demonstrates understanding of:
            - ✅ **Medallion layering** (Bronze/Silver/Gold) — industry standard for data lakes
            - ✅ **ELT vs ETL trade-offs** — transform where it makes sense
            - ✅ **dbt best practices** — modular SQL, built-in testing, documentation
            - ✅ **Tool-appropriate allocation** — right tool for the right job
            """
        )

    # Model Limitations (AC7.5)
    with st.expander("⚠️ Known Limitations"):
        st.markdown(
            """
            ### Current Limitations
            
            1. **Model Training Bias**
               - Emotion model (`multilingual_go_emotions`) trained on English-heavy corpus
               - May have reduced accuracy for non-English languages
               - No anime/film-specific training data
            
            2. **Subtitle Quality Variance**
               - Quality varies significantly across films and languages
               - Some films only available in 2-3 languages (not all 5)
               - Timing drift in some subtitle files (wrong film versions)
            
            3. **No Ground Truth**
               - No human-annotated emotion labels for Ghibli films
               - Cannot measure objective accuracy
               - Validation is relative (consistency) not absolute (correctness)
            
            4. **Temporal Resolution**
               - 1-minute buckets aggregate multiple dialogues
               - Loses sub-minute emotional variation
               - Smoothing further reduces temporal precision
            
            These limitations are acknowledged transparently rather than hidden — 
            a key differentiator for professional data engineering work.
            """
        )

# ============================================================================
# Footer
# ============================================================================

st.markdown("---")
st.caption(
    """
    *"Transparency over perfection — showing the work behind the magic."*
    """
)
