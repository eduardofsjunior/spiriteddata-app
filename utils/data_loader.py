"""
DuckDB connection and emotion mart data loaders.

Provides cached connections and query functions for Streamlit app.

[Source: architecture/12-security-and-performance.md#1-caching]
"""

from typing import Any, Dict, List, Optional

import duckdb
import pandas as pd
import streamlit as st

from .config import DUCKDB_PATH, EMOTION_LABELS, EMOTION_MARTS

# Language display name mapping [Source: Story 3.6.5.1 - AC1]
LANGUAGE_NAMES = {"en": "English", "fr": "French", "es": "Spanish", "nl": "Dutch", "ar": "Arabic"}


@st.cache_resource
def get_duckdb_connection() -> duckdb.DuckDBPyConnection:
    """
    Create persistent DuckDB connection (cached at app startup).

    Returns:
        DuckDB connection object

    Raises:
        FileNotFoundError: If database file does not exist

    [Source: architecture/12-security-and-performance.md#1-caching]
    """
    if not DUCKDB_PATH.exists():
        raise FileNotFoundError(
            f"DuckDB database not found at {DUCKDB_PATH}. "
            "Run data pipeline first: dbt run && python src/nlp/analyze_emotions.py"
        )

    return duckdb.connect(str(DUCKDB_PATH), read_only=True)


@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_emotion_mart(mart_name: str) -> pd.DataFrame:
    """
    Load emotion mart table from DuckDB.

    Args:
        mart_name: Mart key from config.EMOTION_MARTS

    Returns:
        DataFrame with mart data

    Raises:
        ValueError: If mart_name is not recognized

    [Source: architecture/database-schema.md#marts-schema-dbt-tables]
    """
    conn = get_duckdb_connection()

    if mart_name not in EMOTION_MARTS:
        raise ValueError(f"Unknown mart: {mart_name}. Valid options: {list(EMOTION_MARTS.keys())}")

    table_name = EMOTION_MARTS[mart_name]

    query = f"SELECT * FROM {table_name}"

    return conn.execute(query).fetch_df()


@st.cache_data(ttl=3600)
def get_film_list() -> pd.DataFrame:
    """
    Get list of all films with metadata.

    Returns:
        DataFrame with columns: film_id, title, director, release_year, rt_score

    [Source: architecture/database-schema.md#staging-schema-dbt-views]
    """
    conn = get_duckdb_connection()

    query = """
    SELECT
        id as film_id,
        title,
        director,
        release_year,
        rt_score
    FROM main_staging.stg_films
    ORDER BY release_year DESC
    """

    return conn.execute(query).fetch_df()


@st.cache_data(ttl=3600)
def get_film_emotions(film_slug: str, language_code: str) -> pd.DataFrame:
    """
    Get minute-level emotion data for a specific film and language.

    MIGRATION (2025-11-29): Changed to accept film_slug directly instead of film_id.
    This enables emotion queries for v2 subtitle versions (which have NULL film_id).

    Args:
        film_slug: Film slug identifier (e.g., "spirited_away_en", "the_wind_rises_fr_v2")
        language_code: Language code (en, fr, es, nl, ar)

    Returns:
        DataFrame with columns: minute_offset, emotion_* (28 emotions)

    Raises:
        ValueError: If language_code is not in supported languages

    [Source: Story 3.6.5.1 - AC3, Modified from Epic 1]
    """
    # Input validation to prevent SQL injection
    from .config import DATA_STATS

    if language_code not in DATA_STATS["language_codes"]:
        raise ValueError(
            f"Invalid language_code: {language_code}. "
            f"Valid options: {DATA_STATS['language_codes']}"
        )

    conn = get_duckdb_connection()

    # Build dynamic column list from config
    emotion_cols = ", ".join([f"emotion_{label}" for label in EMOTION_LABELS])

    # Use film_slug instead of film_id
    query = f"""
    SELECT
        minute_offset,
        {emotion_cols}
    FROM raw.film_emotions
    WHERE film_slug = ?
      AND language_code = ?
    ORDER BY minute_offset
    """

    return conn.execute(query, [film_slug, language_code]).fetch_df()


# ============================================================================
# Epic 5.2: Home Page Data Functions
# ============================================================================


@st.cache_data(ttl=3600)
def get_hero_stats() -> Dict[str, Any]:
    """
    Get hero statistics for home page dashboard.

    Returns:
        Dict with keys: film_count, emotion_data_points, languages_count, dialogue_entries

    [Source: Story 5.2 - AC2]
    """
    conn = get_duckdb_connection()

    query = """
    SELECT
        (SELECT COUNT(DISTINCT id) FROM main_staging.stg_films) as film_count,
        (SELECT COUNT(*) FROM main_marts.mart_film_emotion_timeseries) as emotion_data_points,
        (SELECT COUNT(DISTINCT language_code) FROM main_marts.mart_film_emotion_timeseries) as languages_count,
        (SELECT SUM(dialogue_count) FROM main_marts.mart_film_emotion_timeseries) as dialogue_entries
    """

    result = conn.execute(query).fetch_df().iloc[0]

    return {
        "film_count": int(result["film_count"]),
        "emotion_data_points": int(result["emotion_data_points"]),
        "languages_count": int(result["languages_count"]),
        "dialogue_entries": int(result["dialogue_entries"]),
    }


@st.cache_data(ttl=3600)
def get_top_joyful_film() -> Dict[str, Any]:
    """
    Get film with highest average joy score.

    Returns:
        Dict with keys: film_title, joy_score

    [Source: Story 5.2 - AC3]
    """
    conn = get_duckdb_connection()

    query = """
    SELECT
        film_title,
        ROUND(emotion_joy, 4) as joy_score
    FROM main_marts.mart_film_emotion_summary
    WHERE language_code = 'en'
    ORDER BY emotion_joy DESC
    LIMIT 1
    """

    result = conn.execute(query).fetch_df().iloc[0]

    return {"film_title": result["film_title"], "joy_score": float(result["joy_score"])}


@st.cache_data(ttl=3600)
def get_top_fearful_film() -> Dict[str, Any]:
    """
    Get film with highest average fear score.

    Returns:
        Dict with keys: film_title, fear_score

    [Source: Story 5.2 - AC3]
    """
    conn = get_duckdb_connection()

    query = """
    SELECT
        film_title,
        ROUND(emotion_fear, 4) as fear_score
    FROM main_marts.mart_film_emotion_summary
    WHERE language_code = 'en'
    ORDER BY emotion_fear DESC
    LIMIT 1
    """

    result = conn.execute(query).fetch_df().iloc[0]

    return {"film_title": result["film_title"], "fear_score": float(result["fear_score"])}


@st.cache_data(ttl=3600)
def get_director_comparison() -> Dict[str, Dict[str, Any]]:
    """
    Get emotional style comparison between Miyazaki and Takahata.

    Returns:
        Dict with keys: miyazaki, takahata
        Each containing: diversity, joy, sadness, style_label

    [Source: Story 5.2 - AC3]
    """
    conn = get_duckdb_connection()

    query = """
    SELECT
        director,
        emotion_diversity,
        avg_emotion_joy,
        avg_emotion_sadness,
        film_count
    FROM main_marts.mart_director_emotion_profile
    WHERE director IN ('Hayao Miyazaki', 'Isao Takahata')
    """

    df = conn.execute(query).fetch_df()

    result = {}
    for _, row in df.iterrows():
        key = "miyazaki" if row["director"] == "Hayao Miyazaki" else "takahata"

        # Determine style based on metrics
        diversity = float(row["emotion_diversity"])
        joy = float(row["avg_emotion_joy"])
        sadness = float(row["avg_emotion_sadness"])

        # Miyazaki: Higher diversity, more adventurous
        # Takahata: Higher sadness/joy, more emotional depth
        if key == "miyazaki":
            style_label = "Adventurous & Diverse"
        else:
            style_label = "Emotionally Intense"

        result[key] = {
            "diversity": diversity,
            "joy": joy,
            "sadness": sadness,
            "film_count": int(row["film_count"]),
            "style_label": style_label,
        }

    return result


@st.cache_data(ttl=3600)
def get_film_emotion_timeseries_by_title(film_title: str, language_code: str) -> pd.DataFrame:
    """
    Get complete emotion timeline for a film by title in a specific language.

    LEGACY FUNCTION: This function queries by film_title instead of film_slug.
    Used by Home.py. Consider migrating to get_film_emotion_timeseries() which uses
    film_slug for better accuracy (film_title is not unique across language versions).

    Args:
        film_title: Film title (e.g., "Spirited Away")
        language_code: Language code (en, fr, es, nl, ar)

    Returns:
        DataFrame with columns: minute_offset, emotion_* (28 emotions)

    [Source: Story 5.2 - AC4]
    """
    from .config import DATA_STATS

    if language_code not in DATA_STATS["language_codes"]:
        raise ValueError(
            f"Invalid language_code: {language_code}. "
            f"Valid options: {DATA_STATS['language_codes']}"
        )

    conn = get_duckdb_connection()

    # Build dynamic column list from config
    emotion_cols = ", ".join([f"emotion_{label}" for label in EMOTION_LABELS])

    query = f"""
    SELECT
        minute_offset,
        {emotion_cols}
    FROM main_marts.mart_film_emotion_timeseries
    WHERE film_title = ?
      AND language_code = ?
    ORDER BY minute_offset
    """

    return conn.execute(query, [film_title, language_code]).fetch_df()


# ============================================================================
# Epic 5.3: Film Explorer (The Spirit Archives) Data Functions
# ============================================================================


@st.cache_data(ttl=3600)
def get_film_list_with_metadata() -> List[Dict[str, Any]]:
    """
    Get list of all film-language combinations with metadata for film selector.

    MIGRATION (2025-11-29): Changed from film_id to film_slug + language_code as primary
    identifier to expose all 174 film-language combinations (including v2 subtitle versions).

    Previous: 97 combinations (56%) - only films with film_id
    Current: 174 combinations (100%) - all films with emotion data

    Returns:
        List of dicts with keys:
        - film_slug: Primary identifier (e.g., "spirited_away", "the_wind_rises_fr_v2")
        - language_code: ISO 639-1 code (en, fr, es, nl, ar)
        - language_name: Display name (English, French, Spanish, Dutch, Arabic)
        - film_title: Film title (e.g., "Spirited Away")
        - film_id: UUID (for backward compatibility, None for v2 films)
        - display_name: Formatted string for UI selector
        - release_year: Year released
        - director: Director name

    [Source: Story 3.6.5.1 - AC1, Fixes Epic 3.6 blocker]
    """
    conn = get_duckdb_connection()

    # Query all film-language combinations from raw.film_emotions
    # CRITICAL: Uses film_slug instead of film_id to expose v2 subtitle versions
    # MIGRATION (Story 3.6.5.1): Added fallback JOIN using parsed film title from slug
    # for v2 films that have NULL film_id
    # PREFERENCE: Show only v2 versions when both v1 and v2 exist (v2 has better subtitles)
    query = """
    WITH parsed_slugs AS (
        SELECT DISTINCT
            film_slug,
            language_code,
            film_id,
            -- Parse film title from slug: "the_wind_rises_fr_v2" -> "the wind rises"
            LOWER(REGEXP_REPLACE(film_slug, '_[a-z]{2}(_v\\d+)?$', '')) as slug_title,
            -- Check if this is a v2 version
            CASE WHEN film_slug LIKE '%_v2' THEN 2 ELSE 1 END as version_priority
        FROM raw.film_emotions
    ),
    -- For each film-language combo, prefer v2 over v1
    preferred_versions AS (
        SELECT
            slug_title,
            language_code,
            MAX(version_priority) as max_version
        FROM parsed_slugs
        GROUP BY slug_title, language_code
    )
    SELECT DISTINCT
        p.film_slug,
        p.language_code,
        p.film_id,
        COALESCE(k.title, f.title, k2.title, f2.title) as film_title,
        COALESCE(k.release_year, f.release_year, k2.release_year, f2.release_year) as release_year,
        COALESCE(k.director, f.director, k2.director, f2.director) as director
    FROM parsed_slugs p
    INNER JOIN preferred_versions pv
        ON p.slug_title = pv.slug_title
        AND p.language_code = pv.language_code
        AND p.version_priority = pv.max_version
    LEFT JOIN main_staging.stg_kaggle_films k ON p.film_id = k.film_id
    LEFT JOIN main_staging.stg_films f ON p.film_id = f.id
    -- Fallback: match by parsed slug title for v2 films with NULL film_id
    LEFT JOIN main_staging.stg_kaggle_films k2
        ON p.film_id IS NULL
        AND LOWER(REPLACE(k2.title, ' ', '_')) = p.slug_title
    LEFT JOIN main_staging.stg_films f2
        ON p.film_id IS NULL
        AND LOWER(REPLACE(f2.title, ' ', '_')) = p.slug_title
    ORDER BY
        COALESCE(k.release_year, f.release_year, k2.release_year, f2.release_year) DESC,
        COALESCE(k.title, f.title, k2.title, f2.title),
        p.language_code
    """

    df = conn.execute(query).fetch_df()

    # Format display names for selector
    films = []
    for _, row in df.iterrows():
        language_name = LANGUAGE_NAMES.get(row["language_code"], row["language_code"])

        # v2 subtitle versions are internal - no need to show in UI
        language_display = language_name

        # Handle null values for release_year and director
        release_year = int(row["release_year"]) if pd.notna(row["release_year"]) else 0
        director = row["director"] if pd.notna(row["director"]) else "Unknown"

        # Extract film title from film_slug if NULL (for v2 films)
        film_title = row["film_title"]
        if pd.isna(film_title):
            # Parse film_slug: e.g., "the_wind_rises_fr_v2" -> "The Wind Rises"
            # Remove language code and _v2 suffix, then convert to title case
            import re

            slug_base = re.sub(r"_[a-z]{2}(_v\d+)?$", "", row["film_slug"])
            film_title = slug_base.replace("_", " ").title()

        films.append(
            {
                "film_slug": row["film_slug"],
                "language_code": row["language_code"],
                "language_name": language_display,
                "film_title": film_title,
                "film_id": str(row["film_id"]) if pd.notna(row["film_id"]) else None,
                "display_name": f"{film_title} ({release_year}) - {language_display} - {director}",
                "release_year": release_year,
                "director": director,
            }
        )

    return films


@st.cache_data(ttl=3600)
def get_film_emotion_timeseries(film_slug: str, language_code: str) -> pd.DataFrame:
    """
    Get smoothed emotion timeline for a film in a specific language.

    MIGRATION (2025-11-29): Renamed from get_film_emotion_timeseries_by_id and changed
    to accept film_slug directly. This enables timeseries queries for v2 subtitle
    versions (which have NULL film_id).

    NOTE: film_slug is the composite identifier that includes language (e.g.,
    "spirited_away_en", "the_wind_rises_fr_v2"), so language_code parameter is
    technically redundant but kept for consistency and validation.

    Args:
        film_slug: Film slug identifier (e.g., "spirited_away_en", "the_wind_rises_fr_v2")
        language_code: Language code (en, fr, es, nl, ar) - used for validation

    Returns:
        DataFrame with columns: minute_offset, dialogue_count, emotion_* (28 emotions)

    [Source: Story 3.6.5.1 - AC3, Renamed from Story 5.3]
    """
    from .config import DATA_STATS

    if language_code not in DATA_STATS["language_codes"]:
        raise ValueError(
            f"Invalid language_code: {language_code}. "
            f"Valid options: {DATA_STATS['language_codes']}"
        )

    conn = get_duckdb_connection()

    # Build dynamic column list from config
    emotion_cols = ", ".join([f"emotion_{label}" for label in EMOTION_LABELS])

    # Query using film_slug directly from mart table (now includes v2 films)
    # MIGRATION (Story 3.6.5.1): Simplified query - no JOIN needed since mart has film_slug
    query = f"""
    SELECT
        minute_offset,
        dialogue_count,
        {emotion_cols}
    FROM main_marts.mart_film_emotion_timeseries
    WHERE film_slug = ?
      AND language_code = ?
    ORDER BY minute_offset
    """

    return conn.execute(query, [film_slug, language_code]).fetch_df()


@st.cache_data(ttl=3600)
def get_raw_emotion_peaks(film_slug: str, language_code: str) -> pd.DataFrame:
    """
    Get raw dialogue-level emotion data for a film (non-smoothed).

    MIGRATION (2025-11-29): Changed to accept film_slug directly instead of film_id.
    This enables raw emotion queries for v2 subtitle versions (which have NULL film_id).

    Args:
        film_slug: Film slug identifier (e.g., "spirited_away_en", "the_wind_rises_fr_v2")
        language_code: Language code (en, fr, es, nl, ar)

    Returns:
        DataFrame with columns: minute_offset, dialogue_count, emotion_* (28 emotions)

    [Source: Story 3.6.5.1 - AC3, Modified from Story 5.3]
    """
    from .config import DATA_STATS

    if language_code not in DATA_STATS["language_codes"]:
        raise ValueError(
            f"Invalid language_code: {language_code}. "
            f"Valid options: {DATA_STATS['language_codes']}"
        )

    conn = get_duckdb_connection()

    # Build dynamic column list from config
    emotion_cols = ", ".join([f"emotion_{label}" for label in EMOTION_LABELS])

    # Use raw.film_emotions table with film_slug (composite identifier)
    query = f"""
    SELECT
        minute_offset,
        dialogue_count,
        {emotion_cols}
    FROM raw.film_emotions
    WHERE film_slug = ?
      AND language_code = ?
    ORDER BY minute_offset
    """

    return conn.execute(query, [film_slug, language_code]).fetch_df()


@st.cache_data(ttl=3600)
def get_film_emotion_summary(film_slug: str, language_code: str) -> Dict[str, float]:
    """
    Get aggregated emotion summary for a film.

    MIGRATION (2025-11-29): Renamed from get_film_emotion_summary_by_id and changed
    to accept film_slug directly. This enables summary queries for v2 subtitle
    versions (which have NULL film_id).

    Args:
        film_slug: Film slug identifier (e.g., "spirited_away_en", "the_wind_rises_fr_v2")
        language_code: Language code (en, fr, es, nl, ar)

    Returns:
        Dict with emotion labels as keys and average scores as values

    [Source: Story 3.6.5.1 - AC3, Renamed from Story 5.3]
    """
    from .config import DATA_STATS

    if language_code not in DATA_STATS["language_codes"]:
        raise ValueError(
            f"Invalid language_code: {language_code}. "
            f"Valid options: {DATA_STATS['language_codes']}"
        )

    conn = get_duckdb_connection()

    # Build dynamic column list from config
    emotion_cols = ", ".join([f"emotion_{label}" for label in EMOTION_LABELS])

    # Query using film_slug directly from mart table (now includes v2 films)
    # MIGRATION (Story 3.6.5.1): Simplified query - no JOIN needed since mart has film_slug
    query = f"""
    SELECT
        {emotion_cols}
    FROM main_marts.mart_film_emotion_summary
    WHERE film_slug = ?
      AND language_code = ?
    LIMIT 1
    """

    df = conn.execute(query, [film_slug, language_code]).fetch_df()

    if df.empty:
        return {}

    # Convert to dict format
    row = df.iloc[0]
    return {label: float(row[f"emotion_{label}"]) for label in EMOTION_LABELS}


@st.cache_data(ttl=3600)
def get_emotion_peaks_with_scenes(film_slug: str, language_code: str) -> pd.DataFrame:
    """
    Get emotion peaks with scene descriptions for marker annotations.

    MIGRATION (2025-11-29): Changed to accept film_slug directly instead of film_id.
    This enables peak emotion queries for v2 subtitle versions (which have NULL film_id).

    Args:
        film_slug: Film slug identifier (e.g., "spirited_away_en", "the_wind_rises_fr_v2")
        language_code: Language code (en, fr, es, nl, ar)

    Returns:
        DataFrame with columns: emotion_type, peak_minute_offset, intensity_score, scene_description

    [Source: Story 3.6.5.1 - AC3, Modified from Story 5.3]
    """
    from .config import DATA_STATS

    if language_code not in DATA_STATS["language_codes"]:
        raise ValueError(
            f"Invalid language_code: {language_code}. "
            f"Valid options: {DATA_STATS['language_codes']}"
        )

    conn = get_duckdb_connection()

    # MIGRATION (Story 3.6.5.1 continuation): Get film_title from film_slug to query peaks
    # Peaks table uses film_title, not film_slug, so we need to extract it
    # film_slug format: "the_wind_rises_en_v2" -> film_title: "The Wind Rises"

    # First, try to get the film_title from staging tables using film_id
    title_query = """
    SELECT DISTINCT
        COALESCE(k.title, f.title) as film_title
    FROM raw.film_emotions e
    LEFT JOIN main_staging.stg_kaggle_films k ON e.film_id = k.film_id
    LEFT JOIN main_staging.stg_films f ON e.film_id = f.id
    WHERE e.film_slug = ?
      AND e.language_code = ?
    LIMIT 1
    """

    title_result = conn.execute(title_query, [film_slug, language_code]).fetchone()

    if title_result and title_result[0]:
        film_title = title_result[0]
    else:
        # Fallback for v2 versions with NULL film_id: derive title from film_slug
        # film_slug format: "the_wind_rises_en_v2" -> "The Wind Rises"
        # Strip language code and version suffix, then title-case
        import re

        base_slug = re.sub(rf"_{language_code}(_v\d+)?$", "", film_slug)
        film_title = base_slug.replace("_", " ").title()

    if not film_title:
        # Return empty DataFrame if film title cannot be determined
        return pd.DataFrame(
            columns=[
                "emotion_type",
                "peak_minute_offset",
                "intensity_score",
                "scene_description",
                "peak_rank",
            ]
        )

    # Now query peaks table directly using film_title and language_code
    # Use LOWER() for case-insensitive comparison (handles "The Tale of The Princess Kaguya" vs "the")
    query = """
    SELECT
        emotion_type,
        peak_minute_offset,
        intensity_score,
        scene_description,
        peak_rank
    FROM main_marts.mart_emotion_peaks_smoothed
    WHERE LOWER(film_title) = LOWER(?)
      AND language_code = ?
      AND peak_rank <= 3
    ORDER BY emotion_type, peak_rank
    """

    return conn.execute(query, [film_title, language_code]).fetch_df()


@st.cache_data(ttl=3600)
def get_film_slug_from_id(film_id: str, language_code: str) -> Optional[str]:
    """
    Get film slug (base name without language suffix) from film ID.

    Args:
        film_id: Film ID (UUID string) from stg_kaggle_films
        language_code: Language code (en, fr, es, nl, ar)

    Returns:
        Base film slug (e.g., "spirited_away") or None if not found

    [Source: Story 5.3 - QA Fix CODE-001: Extract hardcoded query]
    """
    import logging

    import duckdb

    from .config import DATA_STATS, DUCKDB_PATH

    logger = logging.getLogger(__name__)

    if language_code not in DATA_STATS["language_codes"]:
        logger.warning(f"Invalid language_code: {language_code}")
        return None

    try:
        conn = duckdb.connect(str(DUCKDB_PATH), read_only=True)
        result = conn.execute(
            "SELECT DISTINCT film_slug FROM raw.film_emotions WHERE film_id = ? AND language_code = ? LIMIT 1",
            [film_id, language_code],
        ).fetchone()

        if result:
            # film_slug includes language (e.g., "spirited_away_en"), strip language suffix
            film_slug_with_lang = result[0]
            film_slug_base = film_slug_with_lang.rsplit("_", 1)[0]
            return film_slug_base
        else:
            logger.info(f"No film_slug found for film_id={film_id}, language={language_code}")
            return None

    except Exception as e:
        logger.error(f"Failed to get film_slug for film_id={film_id}: {e}", exc_info=True)
        return None


def compute_peaks_from_timeseries(
    timeseries_df: pd.DataFrame, top_n: int = 3
) -> pd.DataFrame:
    """
    Compute emotion peaks directly from timeseries data.

    This ensures peaks always match what's displayed on the chart, avoiding
    misalignment when timeseries and peaks marts use different data sources
    (e.g., v2 subtitles vs original).

    Args:
        timeseries_df: DataFrame from get_film_emotion_timeseries with emotion columns
        top_n: Number of top peaks per emotion to return (default: 3)

    Returns:
        DataFrame with columns: emotion_type, peak_minute_offset, intensity_score, peak_rank
    """
    if timeseries_df.empty:
        return pd.DataFrame(
            columns=["emotion_type", "peak_minute_offset", "intensity_score", "peak_rank"]
        )

    # Get all emotion columns (format: emotion_xxx)
    emotion_cols = [c for c in timeseries_df.columns if c.startswith("emotion_")]

    peaks_list = []
    for col in emotion_cols:
        emotion_type = col.replace("emotion_", "")

        # Skip neutral - not informative for peaks
        if emotion_type == "neutral":
            continue

        # Get top N peaks for this emotion
        sorted_df = timeseries_df.nlargest(top_n, col)

        for rank, (_, row) in enumerate(sorted_df.iterrows(), 1):
            peaks_list.append({
                "emotion_type": emotion_type,
                "peak_minute_offset": int(row["minute_offset"]),
                "intensity_score": float(row[col]),
                "peak_rank": rank,
            })

    return pd.DataFrame(peaks_list)


@st.cache_data(ttl=3600)
def get_peak_dialogues(film_slug: str, language_code: str, peaks_df: pd.DataFrame) -> List[Dict]:
    """
    Load actual dialogue text from parsed subtitle files for emotion peaks.
    Groups consecutive peaks of same emotion into time ranges for diversity.

    IMPROVEMENT (2025-11-23): Uses ±15 second window centered on highest intensity peak
    with proximity-based dialogue ranking for better contextual accuracy.

    Args:
        film_slug: Film slug (e.g., "spirited_away")
        language_code: Language code (en, fr, es, nl, ar)
        peaks_df: DataFrame with peak_minute_offset column

    Returns:
        List of dicts with keys: emotion_type, minute_range, dialogue_lines (list of strings)
        dialogue_lines contains top 3 dialogues closest to peak center (sorted by proximity)

    """
    import json

    # Use DATA_DIR from config for correct path resolution regardless of cwd
    from .config import DATA_DIR

    # Construct path to parsed subtitle file (try v2 improved first, then original)
    parsed_file_v2 = DATA_DIR / "processed" / "subtitles_improved" / f"{film_slug}_{language_code}_v2_parsed.json"
    parsed_file_v1 = DATA_DIR / "processed" / "subtitles" / f"{film_slug}_{language_code}_parsed.json"

    if parsed_file_v2.exists():
        parsed_file = parsed_file_v2
    elif parsed_file_v1.exists():
        parsed_file = parsed_file_v1
    else:
        # Fallback: return empty list if neither file found
        return []

    # Load parsed subtitles
    with open(parsed_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    subtitles = data.get("subtitles", [])

    # Group peaks by emotion type (exclude neutral - not informative)
    emotion_peaks = {}
    for _, peak_row in peaks_df.iterrows():
        emotion_type = peak_row["emotion_type"]

        # Skip neutral emotion
        if emotion_type == "neutral":
            continue

        if emotion_type not in emotion_peaks:
            emotion_peaks[emotion_type] = []
        emotion_peaks[emotion_type].append(
            {
                "minute": int(peak_row["peak_minute_offset"]),
                "intensity": float(peak_row["intensity_score"]),
                "rank": int(peak_row["peak_rank"]),
            }
        )

    # Process each emotion's peaks - group consecutive minutes
    peak_dialogues = []

    for emotion_type, peaks in emotion_peaks.items():
        # Sort by minute
        peaks_sorted = sorted(peaks, key=lambda x: x["minute"])

        # Group consecutive minutes (within 5 minutes = likely same scene due to rolling avg)
        grouped_peaks = []
        current_group = [peaks_sorted[0]]

        for peak in peaks_sorted[1:]:
            if peak["minute"] - current_group[-1]["minute"] <= 5:
                # Consecutive peak - add to current group
                current_group.append(peak)
            else:
                # New peak cluster - save current group and start new one
                grouped_peaks.append(current_group)
                current_group = [peak]

        # Don't forget the last group
        grouped_peaks.append(current_group)

        # For each group, create a single entry with time range
        for group in grouped_peaks:
            # Find highest intensity peak in group
            max_intensity_peak = max(group, key=lambda x: x["intensity"])

            # Calculate time range for display
            min_minute = min(p["minute"] for p in group)
            max_minute = max(p["minute"] for p in group)

            # IMPROVED: Use tighter window centered on highest intensity peak (±15 seconds)
            # This reduces contextually inaccurate dialogue compared to old ±30s from edges
            peak_center_sec = max_intensity_peak["minute"] * 60
            peak_start_sec = peak_center_sec - 15
            peak_end_sec = peak_center_sec + 15

            # Extract dialogues from this time window with proximity scoring
            dialogue_candidates = []
            for sub in subtitles:
                if peak_start_sec <= sub["start_time"] <= peak_end_sec:
                    # Calculate proximity score (closer to peak center = better)
                    distance_from_peak = abs(sub["start_time"] - peak_center_sec)
                    dialogue_candidates.append(
                        {"text": sub["dialogue_text"], "distance": distance_from_peak}
                    )

            # Sort by proximity to peak and select best representative lines
            dialogue_candidates.sort(key=lambda x: x["distance"])
            best_dialogues = [d["text"] for d in dialogue_candidates[:3]]  # Top 3 closest

            # Only include if we found dialogues
            if best_dialogues:
                # Format time range display
                if min_minute == max_minute:
                    minute_display = min_minute
                    minute_range = f"min {min_minute}"
                else:
                    minute_display = min_minute  # For sorting
                    minute_range = f"min {min_minute}-{max_minute}"

                peak_dialogues.append(
                    {
                        "emotion_type": emotion_type,
                        "minute": minute_display,  # For sorting
                        "minute_range": minute_range,  # For display
                        "intensity": max_intensity_peak["intensity"],
                        "peak_rank": max_intensity_peak["rank"],
                        "dialogue_lines": best_dialogues,  # Best representative lines
                    }
                )

    # Sort by minute (chronological order to follow film's narrative)
    peak_dialogues.sort(key=lambda x: x["minute"])

    return peak_dialogues


# ============================================================================
# Epic 3.6: Data Quality Validation Functions
# ============================================================================


@st.cache_data(ttl=3600)
def get_validation_status(film_slug: str, language_code: str) -> Optional[Dict[str, any]]:
    """
    Get data quality validation status for a specific film-language combination.

    MIGRATION (2025-11-29): Changed to accept film_slug directly instead of film_id,
    removing unnecessary intermediate lookup query. This enables validation status
    checks for v2 subtitle versions (which have NULL film_id).

    Checks against dbt validation model to determine if subtitle data extends
    beyond film runtime + 10-minute buffer (indicating data quality issues).

    Args:
        film_slug: Film slug identifier (e.g., "spirited_away", "the_wind_rises_fr_v2")
        language_code: Language code (en, fr, es, nl, ar)

    Returns:
        Dict with keys: validation_status ('PASS'/'FAIL'/'UNKNOWN'),
                       overrun_minutes (float), film_title (str),
                       max_minute_offset (float), expected_duration_minutes (float)
        None if no validation data found

    [Source: Story 3.6.5.1 - AC2, Fixes Epic 3.6 blocker]
    """
    from .config import DATA_STATS

    if language_code not in DATA_STATS["language_codes"]:
        return None

    conn = get_duckdb_connection()

    # Query validation model directly using film_slug (no intermediate lookup needed)
    query = """
    SELECT
        validation_status,
        overrun_minutes,
        film_title,
        max_minute_offset,
        expected_duration_minutes
    FROM main_intermediate.int_emotion_data_quality_checks
    WHERE film_slug = ?
      AND language_code = ?
      AND film_slug != '_SUMMARY_'
    """

    result = conn.execute(query, [film_slug, language_code]).fetch_df()

    if result.empty:
        return None

    row = result.iloc[0]
    return {
        "validation_status": row["validation_status"],
        "overrun_minutes": (
            float(row["overrun_minutes"]) if pd.notna(row["overrun_minutes"]) else None
        ),
        "film_title": row["film_title"],
        "max_minute_offset": (
            float(row["max_minute_offset"]) if pd.notna(row["max_minute_offset"]) else None
        ),
        "expected_duration_minutes": (
            float(row["expected_duration_minutes"])
            if pd.notna(row["expected_duration_minutes"])
            else None
        ),
    }


# ============================================================================
# Epic 5.4: Director Profiles & Cross-Film Analysis Data Functions
# ============================================================================


@st.cache_data(ttl=3600)
def get_director_list() -> List[Dict[str, Any]]:
    """
    Get list of all directors with film counts for director selector.

    Returns:
        List of dicts with keys:
        - director: Director name
        - film_count: Number of films directed
        - display_name: Formatted string for UI selector (e.g., "Hayao Miyazaki (7 films)")

    [Source: Story 5.4 - Task 1.1, AC1]
    """
    conn = get_duckdb_connection()

    query = """
    SELECT
        director,
        film_count
    FROM main_marts.mart_director_emotion_profile
    ORDER BY film_count DESC, director
    """

    df = conn.execute(query).fetch_df()

    directors = []
    for _, row in df.iterrows():
        directors.append(
            {
                "director": row["director"],
                "film_count": int(row["film_count"]),
                "display_name": f"{row['director']} ({int(row['film_count'])} films)",
            }
        )

    return directors


@st.cache_data(ttl=3600)
def get_director_profile(director: str) -> Dict[str, Any]:
    """
    Get complete emotion profile for a specific director.

    Args:
        director: Director name (e.g., "Hayao Miyazaki")

    Returns:
        Dict with keys:
        - director: Director name
        - film_count: Number of films
        - total_minutes_analyzed: Total minutes of emotion data
        - avg_emotion_* : Average scores for all 28 emotions (0-1 range)
        - top_emotion_1/2/3: Top 3 emotions
        - top_emotion_1/2/3_score: Scores for top 3 emotions
        - emotion_diversity: Standard deviation across emotions
        - career_span_years: Career length
        - earliest_film_year: First film year
        - latest_film_year: Most recent film year

    [Source: Story 5.4 - Task 1.2, AC2]
    """
    conn = get_duckdb_connection()

    # Build dynamic column list for all 28 emotions
    emotion_cols = ", ".join([f"avg_emotion_{label}" for label in EMOTION_LABELS])

    query = f"""
    SELECT
        director,
        film_count,
        total_minutes_analyzed,
        {emotion_cols},
        top_emotion_1,
        top_emotion_1_score,
        top_emotion_2,
        top_emotion_2_score,
        top_emotion_3,
        top_emotion_3_score,
        emotion_diversity,
        career_span_years,
        earliest_film_year,
        latest_film_year
    FROM main_marts.mart_director_emotion_profile
    WHERE director = ?
    """

    result = conn.execute(query, [director]).fetch_df()

    if result.empty:
        return {}

    row = result.iloc[0]

    # Convert to dict with proper types
    profile = {
        "director": row["director"],
        "film_count": int(row["film_count"]),
        "total_minutes_analyzed": int(row["total_minutes_analyzed"]),
        "top_emotion_1": row["top_emotion_1"],
        "top_emotion_1_score": float(row["top_emotion_1_score"]),
        "top_emotion_2": row["top_emotion_2"],
        "top_emotion_2_score": float(row["top_emotion_2_score"]),
        "top_emotion_3": row["top_emotion_3"],
        "top_emotion_3_score": float(row["top_emotion_3_score"]),
        "emotion_diversity": float(row["emotion_diversity"]),
        "career_span_years": int(row["career_span_years"]),
        "earliest_film_year": int(row["earliest_film_year"]),
        "latest_film_year": int(row["latest_film_year"]),
    }

    # Add all 28 emotion averages
    for label in EMOTION_LABELS:
        profile[f"avg_emotion_{label}"] = float(row[f"avg_emotion_{label}"])

    return profile


@st.cache_data(ttl=3600)
def get_director_career_evolution(director: str, language_code: str) -> pd.DataFrame:
    """
    Get per-film emotion data for director's career timeline visualization.

    Args:
        director: Director name (e.g., "Hayao Miyazaki")
        language_code: Language code (en, fr, es, nl, ar) to filter results

    Returns:
        DataFrame with columns:
        - film_title: Film title
        - release_year: Year released
        - emotion_* : Average scores for all 28 emotions (0-1 range)

    [Source: Story 5.4 - Task 1.3, AC3]
    """
    from .config import DATA_STATS

    if language_code not in DATA_STATS["language_codes"]:
        raise ValueError(
            f"Invalid language_code: {language_code}. "
            f"Valid options: {DATA_STATS['language_codes']}"
        )

    conn = get_duckdb_connection()

    # Build dynamic column list for all 28 emotions - prefixed for CTE and unprefixed for output
    emotion_cols_cte = ", ".join([f"fs.emotion_{label}" for label in EMOTION_LABELS])
    emotion_cols_select = ", ".join([f"emotion_{label}" for label in EMOTION_LABELS])

    # Join film_emotion_summary with stg_films to get director and release_year
    # Use film_slug to match across tables (handles v2 versions correctly)
    # Filter by language_code to show only selected language
    # PREFER v2 versions when both v1 and v2 exist (to avoid duplicates)
    query = f"""
    WITH version_priority AS (
        SELECT
            fs.film_title,
            f.release_year,
            {emotion_cols_cte},
            -- Assign priority: v2=2, v1=1
            CASE WHEN fs.film_slug LIKE '%_v2' THEN 2 ELSE 1 END as version_priority,
            ROW_NUMBER() OVER (
                PARTITION BY fs.film_title, f.release_year
                ORDER BY CASE WHEN fs.film_slug LIKE '%_v2' THEN 2 ELSE 1 END DESC
            ) as rn
        FROM main_marts.mart_film_emotion_summary fs
        LEFT JOIN main_staging.stg_films f
            ON LOWER(REPLACE(f.title, ' ', '_')) = REGEXP_REPLACE(fs.film_slug, '_[a-z]{{2}}(_v\\d+)?$', '')
        WHERE f.director = ?
          AND fs.language_code = ?
          AND f.release_year IS NOT NULL
    )
    SELECT
        film_title,
        release_year,
        {emotion_cols_select}
    FROM version_priority
    WHERE rn = 1  -- Only take highest priority version (v2 if exists, else v1)
    ORDER BY release_year ASC, film_title
    """

    return conn.execute(query, [director, language_code]).fetch_df()


# ============================================================================
# Epic 5.5: Cross-Language Insights Data Functions
# ============================================================================

# 22 relevant emotions (excludes neutral/ambiguous per Epic 3.5)
EXCLUDED_EMOTIONS = ["confusion", "curiosity", "desire", "realization", "surprise", "neutral"]

# All 10 language pairs stored in mart (alphabetically ordered: lang_a < lang_b)
LANGUAGE_PAIRS_DB = [
    ("ar", "en"),
    ("ar", "es"),
    ("ar", "fr"),
    ("ar", "nl"),
    ("en", "es"),
    ("en", "fr"),
    ("en", "nl"),
    ("es", "fr"),
    ("es", "nl"),
    ("fr", "nl"),
]


@st.cache_data(ttl=3600)
def get_language_pairs() -> List[Dict[str, str]]:
    """
    Get all language pairs available for cross-language comparison.

    Language pairs in DB are stored alphabetically (e.g., ar-en not en-ar).
    This function provides user-friendly display names and handles the mapping.

    Returns:
        List of dicts with keys:
        - language_a: DB column language_a (alphabetically first)
        - language_b: DB column language_b (alphabetically second)
        - display_name: User-friendly format (e.g., "English → Arabic")
        - display_a: Display name for language_a
        - display_b: Display name for language_b

    [Source: Story 5.5 - Task 1.1, AC1]
    """
    pairs = []

    for lang_a, lang_b in LANGUAGE_PAIRS_DB:
        display_a = LANGUAGE_NAMES.get(lang_a, lang_a.upper())
        display_b = LANGUAGE_NAMES.get(lang_b, lang_b.upper())

        pairs.append(
            {
                "language_a": lang_a,
                "language_b": lang_b,
                "display_name": f"{display_a} → {display_b}",
                "display_a": display_a,
                "display_b": display_b,
            }
        )

    return pairs


@st.cache_data(ttl=3600)
def get_film_list_for_language_pair(language_a: str, language_b: str) -> List[Dict[str, Any]]:
    """
    Get list of films available for a specific language pair.

    Args:
        language_a: First language code (as stored in DB)
        language_b: Second language code (as stored in DB)

    Returns:
        List of dicts with keys: film_id, film_title

    [Source: Story 5.5 - Task 1.2, AC1]
    """
    conn = get_duckdb_connection()

    query = """
    SELECT DISTINCT
        film_id,
        film_title
    FROM main_marts.mart_cross_language_emotion_comparison
    WHERE language_a = ?
      AND language_b = ?
    ORDER BY film_title
    """

    df = conn.execute(query, [language_a, language_b]).fetch_df()

    films = []
    for _, row in df.iterrows():
        films.append({"film_id": row["film_id"], "film_title": row["film_title"]})

    return films


@st.cache_data(ttl=3600)
def get_translation_biases(
    language_a: str, language_b: str, film_id: Optional[str] = None, top_n: int = 10
) -> pd.DataFrame:
    """
    Get top N emotions with largest percent differences for language pair.

    Args:
        language_a: First language code (as stored in DB)
        language_b: Second language code (as stored in DB)
        film_id: Optional film ID to filter (None = all films aggregated)
        top_n: Number of top biases to return (default: 10)

    Returns:
        DataFrame with columns: emotion_type, avg_score_lang_a, avg_score_lang_b,
                               difference_score, percent_difference, is_significant

    [Source: Story 5.5 - Task 1.3, AC2]
    """
    conn = get_duckdb_connection()

    # Build emotion exclusion list for SQL
    excluded_list = ", ".join([f"'{e}'" for e in EXCLUDED_EMOTIONS])

    if film_id is not None:
        # Specific film
        query = f"""
        SELECT
            emotion_type,
            avg_score_lang_a,
            avg_score_lang_b,
            difference_score,
            percent_difference,
            is_significant
        FROM main_marts.mart_cross_language_emotion_comparison
        WHERE language_a = ?
          AND language_b = ?
          AND film_id = ?
          AND emotion_type NOT IN ({excluded_list})
        ORDER BY ABS(percent_difference) DESC
        LIMIT ?
        """
        params = [language_a, language_b, film_id, top_n]
    else:
        # All films aggregated - calculate average across films
        query = f"""
        SELECT
            emotion_type,
            AVG(avg_score_lang_a) as avg_score_lang_a,
            AVG(avg_score_lang_b) as avg_score_lang_b,
            AVG(difference_score) as difference_score,
            AVG(percent_difference) as percent_difference,
            BOOL_OR(is_significant) as is_significant
        FROM main_marts.mart_cross_language_emotion_comparison
        WHERE language_a = ?
          AND language_b = ?
          AND emotion_type NOT IN ({excluded_list})
        GROUP BY emotion_type
        ORDER BY ABS(AVG(percent_difference)) DESC
        LIMIT ?
        """
        params = [language_a, language_b, top_n]

    return conn.execute(query, params).fetch_df()


@st.cache_data(ttl=3600)
def get_consistency_score(
    language_a: str, language_b: str, film_id: Optional[str] = None
) -> float:
    """
    Calculate consistency score (avg absolute percent difference) for language pair.

    Lower score = more consistent translations.

    Args:
        language_a: First language code (as stored in DB)
        language_b: Second language code (as stored in DB)
        film_id: Optional film ID to filter (None = all films)

    Returns:
        Consistency score as float (0-100+ range)

    [Source: Story 5.5 - Task 1.4, AC3]
    """
    conn = get_duckdb_connection()

    # Build emotion exclusion list for SQL
    excluded_list = ", ".join([f"'{e}'" for e in EXCLUDED_EMOTIONS])

    if film_id is not None:
        query = f"""
        SELECT AVG(ABS(percent_difference)) as consistency_score
        FROM main_marts.mart_cross_language_emotion_comparison
        WHERE language_a = ?
          AND language_b = ?
          AND film_id = ?
          AND emotion_type NOT IN ({excluded_list})
        """
        params = [language_a, language_b, film_id]
    else:
        query = f"""
        SELECT AVG(ABS(percent_difference)) as consistency_score
        FROM main_marts.mart_cross_language_emotion_comparison
        WHERE language_a = ?
          AND language_b = ?
          AND emotion_type NOT IN ({excluded_list})
        """
        params = [language_a, language_b]

    result = conn.execute(query, params).fetchone()

    return float(result[0]) if result and result[0] is not None else 0.0


@st.cache_data(ttl=3600)
def get_consistency_matrix(film_id: Optional[str] = None) -> pd.DataFrame:
    """
    Get consistency scores for all 10 language pairs as a matrix.

    Args:
        film_id: Optional film ID to filter (None = all films aggregated)

    Returns:
        DataFrame with 5x5 matrix (languages as rows/cols), values are consistency scores

    [Source: Story 5.5 - Task 1.5, AC3]
    """
    conn = get_duckdb_connection()

    # Build emotion exclusion list for SQL
    excluded_list = ", ".join([f"'{e}'" for e in EXCLUDED_EMOTIONS])

    if film_id is not None:
        query = f"""
        SELECT
            language_a,
            language_b,
            AVG(ABS(percent_difference)) as consistency_score
        FROM main_marts.mart_cross_language_emotion_comparison
        WHERE film_id = ?
          AND emotion_type NOT IN ({excluded_list})
        GROUP BY language_a, language_b
        """
        params = [film_id]
    else:
        query = f"""
        SELECT
            language_a,
            language_b,
            AVG(ABS(percent_difference)) as consistency_score
        FROM main_marts.mart_cross_language_emotion_comparison
        WHERE emotion_type NOT IN ({excluded_list})
        GROUP BY language_a, language_b
        """
        params = []

    df = conn.execute(query, params).fetch_df()

    # Create 5x5 matrix with language display names
    languages = ["ar", "en", "es", "fr", "nl"]
    display_names = ["Arabic", "English", "Spanish", "French", "Dutch"]

    # Initialize matrix with zeros (diagonal = 0, self-consistency)
    matrix = pd.DataFrame(0.0, index=display_names, columns=display_names)

    # Fill matrix with consistency scores
    for _, row in df.iterrows():
        lang_a = row["language_a"]
        lang_b = row["language_b"]
        score = row["consistency_score"]

        # Map to display names
        idx_a = languages.index(lang_a) if lang_a in languages else -1
        idx_b = languages.index(lang_b) if lang_b in languages else -1

        if idx_a >= 0 and idx_b >= 0:
            # Symmetric matrix
            matrix.iloc[idx_a, idx_b] = score
            matrix.iloc[idx_b, idx_a] = score

    return matrix


@st.cache_data(ttl=3600)
def get_emotion_by_language(emotion_type: str, film_id: Optional[str] = None) -> pd.DataFrame:
    """
    Get specific emotion scores across all 5 languages.

    Args:
        emotion_type: Emotion name (e.g., "amusement")
        film_id: Optional film ID to filter (None = all films aggregated)

    Returns:
        DataFrame with columns: language_code, language_name, avg_score

    [Source: Story 5.5 - Task 1.6, AC4]
    """
    conn = get_duckdb_connection()

    # Query scores from both language_a and language_b columns to get all 5 languages
    if film_id is not None:
        query = """
        WITH lang_scores AS (
            -- Get scores for language_a
            SELECT language_a as language_code, avg_score_lang_a as avg_score
            FROM main_marts.mart_cross_language_emotion_comparison
            WHERE emotion_type = ? AND film_id = ?
            UNION ALL
            -- Get scores for language_b
            SELECT language_b as language_code, avg_score_lang_b as avg_score
            FROM main_marts.mart_cross_language_emotion_comparison
            WHERE emotion_type = ? AND film_id = ?
        )
        SELECT
            language_code,
            AVG(avg_score) as avg_score
        FROM lang_scores
        GROUP BY language_code
        ORDER BY language_code
        """
        params = [emotion_type, film_id, emotion_type, film_id]
    else:
        query = """
        WITH lang_scores AS (
            SELECT language_a as language_code, avg_score_lang_a as avg_score
            FROM main_marts.mart_cross_language_emotion_comparison
            WHERE emotion_type = ?
            UNION ALL
            SELECT language_b as language_code, avg_score_lang_b as avg_score
            FROM main_marts.mart_cross_language_emotion_comparison
            WHERE emotion_type = ?
        )
        SELECT
            language_code,
            AVG(avg_score) as avg_score
        FROM lang_scores
        GROUP BY language_code
        ORDER BY language_code
        """
        params = [emotion_type, emotion_type]

    df = conn.execute(query, params).fetch_df()

    # Add display names
    df["language_name"] = df["language_code"].map(LANGUAGE_NAMES)

    return df


@st.cache_data(ttl=3600)
def get_film_consistency_ranking(language_a: str, language_b: str) -> pd.DataFrame:
    """
    Get all films ranked by consistency score for a language pair.

    Args:
        language_a: First language code (as stored in DB)
        language_b: Second language code (as stored in DB)

    Returns:
        DataFrame with columns: film_id, film_title, consistency_score,
                               most_divergent_emotion, max_percent_difference

    [Source: Story 5.5 - Task 1.7, AC5]
    """
    conn = get_duckdb_connection()

    # Build emotion exclusion list for SQL
    excluded_list = ", ".join([f"'{e}'" for e in EXCLUDED_EMOTIONS])

    query = f"""
    WITH film_scores AS (
        SELECT
            film_id,
            film_title,
            AVG(ABS(percent_difference)) as consistency_score,
            MAX(ABS(percent_difference)) as max_percent_difference
        FROM main_marts.mart_cross_language_emotion_comparison
        WHERE language_a = ?
          AND language_b = ?
          AND emotion_type NOT IN ({excluded_list})
        GROUP BY film_id, film_title
    ),
    most_divergent AS (
        SELECT DISTINCT ON (film_id)
            film_id,
            emotion_type as most_divergent_emotion
        FROM main_marts.mart_cross_language_emotion_comparison
        WHERE language_a = ?
          AND language_b = ?
          AND emotion_type NOT IN ({excluded_list})
        ORDER BY film_id, ABS(percent_difference) DESC
    )
    SELECT
        fs.film_id,
        fs.film_title,
        fs.consistency_score,
        md.most_divergent_emotion,
        fs.max_percent_difference
    FROM film_scores fs
    LEFT JOIN most_divergent md ON fs.film_id = md.film_id
    ORDER BY fs.consistency_score ASC
    """

    return conn.execute(query, [language_a, language_b, language_a, language_b]).fetch_df()


# ============================================================================
# Epic 5.6: Methodology & Data Quality Page Data Functions
# ============================================================================


@st.cache_data(ttl=3600)
def get_methodology_metrics(film_id: Optional[str] = None) -> pd.DataFrame:
    """
    Load methodology metrics from mart for signal processing demonstration.

    Args:
        film_id: Optional film ID filter (None = aggregate across all films)

    Returns:
        DataFrame with columns: window_size_minutes, noise_level, peak_count,
        avg_intensity, temporal_precision_loss_pct, is_recommended

    [Source: Story 5.6 - Task 1.1, AC2-AC3]
    """
    conn = get_duckdb_connection()

    if film_id:
        query = """
        SELECT
            window_size_minutes,
            noise_level,
            peak_count,
            avg_intensity,
            temporal_precision_loss_pct,
            is_recommended
        FROM main_marts.mart_emotion_methodology_metrics
        WHERE film_id = ?
        ORDER BY window_size_minutes
        """
        df = conn.execute(query, [film_id]).fetch_df()
    else:
        # Aggregate across all films
        query = """
        SELECT
            window_size_minutes,
            ROUND(AVG(noise_level), 6) AS noise_level,
            ROUND(AVG(peak_count), 1) AS peak_count,
            ROUND(AVG(avg_intensity), 6) AS avg_intensity,
            ROUND(AVG(temporal_precision_loss_pct), 2) AS temporal_precision_loss_pct,
            MAX(is_recommended) AS is_recommended
        FROM main_marts.mart_emotion_methodology_metrics
        GROUP BY window_size_minutes
        ORDER BY window_size_minutes
        """
        df = conn.execute(query).fetch_df()

    return df


@st.cache_data(ttl=3600)
def get_film_list_for_methodology() -> List[Dict[str, Any]]:
    """
    Get list of films available in methodology metrics mart.

    Returns:
        List of dicts with keys: film_id, film_title

    [Source: Story 5.6 - Task 1.2, AC2]
    """
    conn = get_duckdb_connection()

    query = """
    SELECT DISTINCT
        film_id,
        film_title
    FROM main_marts.mart_emotion_methodology_metrics
    WHERE film_id IS NOT NULL
    ORDER BY film_title
    """

    df = conn.execute(query).fetch_df()

    return [
        {"film_id": row["film_id"], "film_title": row["film_title"]}
        for _, row in df.iterrows()
    ]


@st.cache_data(ttl=3600)
def get_emotion_timeseries_for_methodology(
    film_slug: str, language_code: str, emotion: str = "joy"
) -> pd.DataFrame:
    """
    Get raw minute-by-minute emotion data for methodology comparison.

    Prefers v2 subtitle versions when available (better quality).

    Args:
        film_slug: Film slug identifier (e.g., "spirited_away_en")
        language_code: Language code (en, fr, es, nl, ar)
        emotion: Emotion column name without prefix (default: "joy")

    Returns:
        DataFrame with columns: minute_offset, emotion_score (raw)

    [Source: Story 5.6 - Task 1.3, AC2]
    """
    from .config import DATA_STATS

    if language_code not in DATA_STATS["language_codes"]:
        raise ValueError(
            f"Invalid language_code: {language_code}. "
            f"Valid options: {DATA_STATS['language_codes']}"
        )

    conn = get_duckdb_connection()

    # Query raw data - prefers v2 versions
    query = f"""
    SELECT
        minute_offset,
        emotion_{emotion} as emotion_score
    FROM raw.film_emotions
    WHERE film_slug = ?
      AND language_code = ?
    ORDER BY minute_offset
    """

    return conn.execute(query, [film_slug, language_code]).fetch_df()


@st.cache_data(ttl=3600)
def get_smoothed_timeseries_for_methodology(
    film_slug: str, language_code: str, emotion: str = "joy", window_size: int = 10
) -> pd.DataFrame:
    """
    Get smoothed emotion timeseries with configurable window size.

    Computes rolling average on the fly for interactive window size exploration.

    Args:
        film_slug: Film slug identifier (e.g., "spirited_away_en")
        language_code: Language code (en, fr, es, nl, ar)
        emotion: Emotion column name without prefix (default: "joy")
        window_size: Rolling window size in minutes (default: 10)

    Returns:
        DataFrame with columns: minute_offset, emotion_score (smoothed)

    [Source: Story 5.6 - Task 1.4, AC2]
    """
    from .config import DATA_STATS

    if language_code not in DATA_STATS["language_codes"]:
        raise ValueError(
            f"Invalid language_code: {language_code}. "
            f"Valid options: {DATA_STATS['language_codes']}"
        )

    conn = get_duckdb_connection()

    # Compute rolling average with specified window size
    half_window = window_size // 2
    query = f"""
    SELECT
        minute_offset,
        AVG(emotion_{emotion}) OVER (
            ORDER BY minute_offset
            ROWS BETWEEN {half_window} PRECEDING AND {half_window} FOLLOWING
        ) as emotion_score
    FROM raw.film_emotions
    WHERE film_slug = ?
      AND language_code = ?
    ORDER BY minute_offset
    """

    return conn.execute(query, [film_slug, language_code]).fetch_df()


@st.cache_data(ttl=3600)
def get_dual_resolution_peaks(
    film_slug: str, language_code: str, emotion: str = "joy", top_n: int = 3
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Get both smoothed and raw peaks for dual-resolution comparison.

    Args:
        film_slug: Film slug identifier
        language_code: Language code (en, fr, es, nl, ar)
        emotion: Emotion type to get peaks for (default: "joy")
        top_n: Number of top peaks to return (default: 3)

    Returns:
        Dict with keys "smoothed" and "raw", each containing list of dicts:
        - minute_offset: Peak timestamp
        - peak_score: Peak intensity
        - peak_rank: Rank (1 = highest)

    [Source: Story 5.6 - Task 1.5, AC4]
    """
    conn = get_duckdb_connection()

    # Get film_title from film_slug for peaks query
    import re

    base_slug = re.sub(rf"_{language_code}(_v\d+)?$", "", film_slug)
    film_title = base_slug.replace("_", " ").title()

    # Query smoothed peaks (uses peak_minute_offset column)
    smoothed_query = """
    SELECT
        peak_minute_offset as minute_offset,
        intensity_score as peak_score,
        peak_rank
    FROM main_marts.mart_emotion_peaks_smoothed
    WHERE LOWER(film_title) = LOWER(?)
      AND language_code = ?
      AND emotion_type = ?
      AND peak_rank <= ?
    ORDER BY peak_rank
    """

    smoothed_df = conn.execute(
        smoothed_query, [film_title, language_code, emotion, top_n]
    ).fetch_df()

    # Query raw peaks (uses minute_offset column - different from smoothed!)
    raw_query = """
    SELECT
        minute_offset,
        intensity_score as peak_score,
        peak_rank
    FROM main_marts.mart_emotion_peaks_raw
    WHERE LOWER(film_title) = LOWER(?)
      AND language_code = ?
      AND emotion_type = ?
      AND peak_rank <= ?
    ORDER BY peak_rank
    """

    raw_df = conn.execute(raw_query, [film_title, language_code, emotion, top_n]).fetch_df()

    return {
        "smoothed": smoothed_df.to_dict("records") if not smoothed_df.empty else [],
        "raw": raw_df.to_dict("records") if not raw_df.empty else [],
    }


@st.cache_data(ttl=3600)
def get_validation_summary() -> Dict[str, Any]:
    """
    Get aggregate validation summary for data quality dashboard.

    FIXED (Story 5.6): Uses title-based matching instead of film_id to avoid
    UNKNOWN status for v2 films and handles apostrophe differences in film names
    (e.g., "Howl's Moving Castle" vs "howls_moving_castle").

    Returns:
        Dict with keys:
        - total_films: Total film-language combinations validated (preferred versions only)
        - pass_count: Number passing validation
        - fail_count: Number failing validation
        - pass_rate: Pass rate as percentage (PASS / (PASS + FAIL))
        - failures: List of failed films with details

    [Source: Story 5.6 - Task 1.6, AC5]
    """
    conn = get_duckdb_connection()

    # Calculate validation status using title-based matching with apostrophe handling
    # This query:
    # 1. Gets film durations from stg_films, normalizing titles (remove apostrophes)
    # 2. Extracts base film slug from emotion data
    # 3. Selects only PREFERRED versions (v2 if available, else v1)
    # 4. Calculates PASS/FAIL based on max_minute_offset vs expected_duration + 10min buffer
    validation_query = r"""
    WITH film_durations AS (
        SELECT
            LOWER(REPLACE(REPLACE(title, '''', ''), ' ', '_')) as title_slug,
            title as film_title,
            running_time as expected_duration_minutes
        FROM main_staging.stg_films
        WHERE running_time IS NOT NULL
    ),
    emotion_stats AS (
        SELECT
            film_slug,
            language_code,
            -- Normalize: remove language suffix and version, strip apostrophes
            REPLACE(REGEXP_REPLACE(film_slug, '_[a-z]{2}(_v\d+)?$', ''), '''', '') as base_film_slug,
            CASE WHEN film_slug LIKE '%_v2' THEN 2 ELSE 1 END as version_priority,
            MAX(minute_offset) as max_minute_offset
        FROM raw.film_emotions
        WHERE film_slug NOT LIKE '%film1%'
          AND film_slug != '_SUMMARY_'
        GROUP BY film_slug, language_code
    ),
    preferred_versions AS (
        SELECT
            base_film_slug,
            language_code,
            MAX(version_priority) as max_version
        FROM emotion_stats
        GROUP BY base_film_slug, language_code
    ),
    app_films AS (
        SELECT
            e.film_slug,
            e.language_code,
            e.base_film_slug,
            e.max_minute_offset
        FROM emotion_stats e
        INNER JOIN preferred_versions pv
            ON e.base_film_slug = pv.base_film_slug
            AND e.language_code = pv.language_code
            AND e.version_priority = pv.max_version
    )
    SELECT
        a.film_slug,
        COALESCE(d.film_title, REPLACE(a.base_film_slug, '_', ' ')) as film_title,
        a.language_code,
        a.max_minute_offset,
        d.expected_duration_minutes,
        CASE
            WHEN d.expected_duration_minutes IS NULL THEN 'UNKNOWN'
            WHEN a.max_minute_offset <= d.expected_duration_minutes + 10 THEN 'PASS'
            ELSE 'FAIL'
        END as validation_status,
        CASE
            WHEN d.expected_duration_minutes IS NOT NULL
                AND a.max_minute_offset > d.expected_duration_minutes + 10
            THEN a.max_minute_offset - d.expected_duration_minutes
            ELSE NULL
        END as overrun_minutes
    FROM app_films a
    LEFT JOIN film_durations d ON a.base_film_slug = d.title_slug
    ORDER BY a.film_slug
    """

    validation_df = conn.execute(validation_query).fetch_df()

    # Filter out UNKNOWN (should be none with correct matching, but safety check)
    validated_df = validation_df[validation_df["validation_status"] != "UNKNOWN"]

    # Calculate summary stats
    pass_count = len(validated_df[validated_df["validation_status"] == "PASS"])
    fail_count = len(validated_df[validated_df["validation_status"] == "FAIL"])
    total = pass_count + fail_count

    # Get failure details
    failures_df = validated_df[validated_df["validation_status"] == "FAIL"].copy()

    return {
        "total_films": total,
        "pass_count": pass_count,
        "fail_count": fail_count,
        "pass_rate": round(pass_count / total * 100, 1) if total > 0 else 0,
        "failures": failures_df.to_dict("records") if not failures_df.empty else [],
    }


@st.cache_data(ttl=3600)
def get_language_coverage_matrix() -> pd.DataFrame:
    """
    Get language coverage matrix for preferred film versions only.

    FIXED (Story 5.6): Uses title-based matching and prefers v2 versions,
    consistent with validation summary.

    Returns:
        DataFrame with film_title as index, language codes as columns,
        values: 1 (available), 0 (missing)

    [Source: Story 5.6 - Task 1.7, AC5]
    """
    conn = get_duckdb_connection()

    # Get preferred versions (v2 if available) with proper title matching
    query = r"""
    WITH film_titles AS (
        SELECT
            LOWER(REPLACE(REPLACE(title, '''', ''), ' ', '_')) as title_slug,
            title as film_title
        FROM main_staging.stg_films
    ),
    emotion_stats AS (
        SELECT DISTINCT
            film_slug,
            language_code,
            REPLACE(REGEXP_REPLACE(film_slug, '_[a-z]{2}(_v\d+)?$', ''), '''', '') as base_film_slug,
            CASE WHEN film_slug LIKE '%_v2' THEN 2 ELSE 1 END as version_priority
        FROM raw.film_emotions
        WHERE film_slug NOT LIKE '%film1%'
          AND film_slug != '_SUMMARY_'
    ),
    preferred_versions AS (
        SELECT
            base_film_slug,
            language_code,
            MAX(version_priority) as max_version
        FROM emotion_stats
        GROUP BY base_film_slug, language_code
    ),
    app_films AS (
        SELECT
            e.film_slug,
            e.language_code,
            e.base_film_slug
        FROM emotion_stats e
        INNER JOIN preferred_versions pv
            ON e.base_film_slug = pv.base_film_slug
            AND e.language_code = pv.language_code
            AND e.version_priority = pv.max_version
    )
    SELECT DISTINCT
        COALESCE(f.film_title, REPLACE(a.base_film_slug, '_', ' ')) as film_title,
        a.language_code,
        1 as available
    FROM app_films a
    LEFT JOIN film_titles f ON a.base_film_slug = f.title_slug
    WHERE COALESCE(f.film_title, a.base_film_slug) IS NOT NULL
    ORDER BY film_title, language_code
    """

    df = conn.execute(query).fetch_df()

    if df.empty:
        return pd.DataFrame()

    # Pivot to matrix format
    matrix = df.pivot_table(
        index="film_title", columns="language_code", values="available", fill_value=0
    )

    # Ensure all 5 languages are columns
    for lang in ["ar", "en", "es", "fr", "nl"]:
        if lang not in matrix.columns:
            matrix[lang] = 0

    # Reorder columns
    matrix = matrix[["en", "fr", "es", "nl", "ar"]]

    return matrix


@st.cache_data(ttl=3600)
def get_cross_language_consistency_summary() -> pd.DataFrame:
    """
    Get consistency scores for all language pairs.

    Returns:
        DataFrame with columns: language_a, language_b, consistency_score
        Lower score = more consistent (similar emotion patterns)

    [Source: Story 5.6 - Task 1.8, AC6]
    """
    conn = get_duckdb_connection()

    # Build emotion exclusion list
    excluded_emotions = ["confusion", "curiosity", "desire", "realization", "surprise", "neutral"]
    excluded_list = ", ".join([f"'{e}'" for e in excluded_emotions])

    query = f"""
    SELECT
        language_a,
        language_b,
        ROUND(AVG(ABS(percent_difference)), 2) as consistency_score
    FROM main_marts.mart_cross_language_emotion_comparison
    WHERE emotion_type NOT IN ({excluded_list})
    GROUP BY language_a, language_b
    ORDER BY consistency_score ASC
    """

    return conn.execute(query).fetch_df()


@st.cache_data(ttl=3600)
def get_film_similarity_matrix(language_code: str) -> pd.DataFrame:
    """
    Get pre-computed film similarity scores for heatmap visualization.

    Args:
        language_code: Language code (en, fr, es, nl, ar)

    Returns:
        DataFrame with columns:
        - film_title_a: First film title
        - film_title_b: Second film title
        - similarity_score: Cosine similarity (0-1 range, 1 = identical)

    Raises:
        ValueError: If language_code is not in supported languages

    [Source: Story 5.4 - Task 1.4, AC4]
    """
    from .config import DATA_STATS

    if language_code not in DATA_STATS["language_codes"]:
        raise ValueError(
            f"Invalid language_code: {language_code}. "
            f"Valid options: {DATA_STATS['language_codes']}"
        )

    conn = get_duckdb_connection()

    # Query similarity matrix for selected language
    # Note: Matrix is pre-computed in Story 2.5.3 with cosine similarity on 27 emotions
    query = """
    SELECT
        film_title_a,
        film_title_b,
        similarity_score
    FROM main_marts.mart_film_similarity_matrix
    WHERE language_code = ?
    ORDER BY film_title_a, film_title_b
    """

    return conn.execute(query, [language_code]).fetch_df()


# ============================================================================
# Story 5.8: Memories of Sora - RAG Artifact Data Functions
# ============================================================================


@st.cache_data(ttl=3600)
def load_conversation_log(log_filename: str) -> Dict[str, Any]:
    """
    Load and parse a RAG conversation log JSON file.

    Args:
        log_filename: Name of the log file (e.g., "rag_conversation_2025-11-10_18-38-37.json")

    Returns:
        Dict with keys:
        - metadata: Dict with start_time, end_time, total_queries, total_tokens, total_cost
        - queries: List of user queries
        - responses: List of assistant responses
        - exchanges: List of dicts with query/response pairs
        - statistics: Dict with session stats

    [Source: Story 5.8 - Task 1.1, AC5]
    """
    import json
    from pathlib import Path

    # Use DATA_DIR from config for correct path resolution
    from .config import DATA_DIR

    logs_dir = DATA_DIR.parent / "logs"
    log_path = logs_dir / log_filename

    if not log_path.exists():
        return {
            "metadata": {},
            "queries": [],
            "responses": [],
            "exchanges": [],
            "statistics": {},
            "error": f"Log file not found: {log_filename}",
        }

    with open(log_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Parse conversation history into structured format
    history = data.get("history", [])
    queries = []
    responses = []
    exchanges = []

    current_query = None
    for msg in history:
        if msg.get("role") == "user":
            current_query = msg.get("content", "")
            queries.append(current_query)
        elif msg.get("role") == "assistant":
            response = msg.get("content", "")
            responses.append(response)
            if current_query is not None:
                exchanges.append({"query": current_query, "response": response})
                current_query = None

    return {
        "metadata": data.get("metadata", {}),
        "queries": queries,
        "responses": responses,
        "exchanges": exchanges,
        "statistics": data.get("statistics", {}),
    }


@st.cache_data(ttl=3600)
def get_available_conversation_logs() -> List[Dict[str, Any]]:
    """
    List all available RAG conversation log files with metadata.

    Returns:
        List of dicts with keys:
        - filename: Log file name
        - timestamp: Parsed datetime string (YYYY-MM-DD HH:MM:SS)
        - query_count: Number of queries in the log
        - total_cost: Total API cost for the session

    [Source: Story 5.8 - Task 1.2, AC5]
    """
    import json
    import re
    from pathlib import Path

    from .config import DATA_DIR

    logs_dir = DATA_DIR.parent / "logs"
    log_files = []

    if not logs_dir.exists():
        return []

    for log_path in sorted(logs_dir.glob("rag_conversation_*.json"), reverse=True):
        # Parse timestamp from filename: rag_conversation_2025-11-10_18-38-37.json
        match = re.search(r"rag_conversation_(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})\.json", log_path.name)
        if match:
            date_str = match.group(1)
            time_str = match.group(2).replace("-", ":")
            timestamp = f"{date_str} {time_str}"
        else:
            timestamp = "Unknown"

        # Load metadata for query count and cost
        try:
            with open(log_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            metadata = data.get("metadata", {})
            query_count = metadata.get("total_queries", 0)
            total_cost = metadata.get("total_cost", 0.0)
        except (json.JSONDecodeError, IOError):
            query_count = 0
            total_cost = 0.0

        log_files.append({
            "filename": log_path.name,
            "timestamp": timestamp,
            "query_count": query_count,
            "total_cost": total_cost,
        })

    return log_files


@st.cache_data(ttl=3600)
def load_validation_report_summary() -> Dict[str, Any]:
    """
    Parse key metrics from the RAG validation report markdown file.

    Returns:
        Dict with keys:
        - total_queries: Total number of test queries
        - pass_rate: Percentage of queries passed (0-100)
        - validation_score: Overall validation score (0-100)
        - total_cost: Total API cost in USD
        - avg_response_time: Average response time in seconds
        - category_breakdown: Dict mapping category name to (passed, total, percentage)
        - error: Error message if file not found or parsing failed

    [Source: Story 5.8 - Task 1.3, AC3/AC5]
    """
    import re
    from pathlib import Path

    from .config import DATA_DIR

    report_path = DATA_DIR.parent / "docs" / "rag_validation_report.md"

    if not report_path.exists():
        return {
            "total_queries": 0,
            "pass_rate": 0.0,
            "validation_score": 0.0,
            "total_cost": 0.0,
            "avg_response_time": 0.0,
            "category_breakdown": {},
            "error": "Validation report not found",
        }

    try:
        with open(report_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Parse executive summary metrics
        total_queries_match = re.search(r"\*\*Total Queries Tested\*\*:\s*(\d+)", content)
        pass_rate_match = re.search(r"\*\*Queries Passed\*\*:\s*\d+/\d+\s*\((\d+\.?\d*)%\)", content)
        validation_score_match = re.search(r"\*\*Overall Validation Score\*\*:\s*(\d+\.?\d*)%", content)
        total_cost_match = re.search(r"\*\*Total API Cost\*\*:\s*\$(\d+\.?\d*)", content)
        avg_time_match = re.search(r"\*\*Average Response Time\*\*:\s*(\d+\.?\d*)\s*seconds", content)

        total_queries = int(total_queries_match.group(1)) if total_queries_match else 0
        pass_rate = float(pass_rate_match.group(1)) if pass_rate_match else 0.0
        validation_score = float(validation_score_match.group(1)) if validation_score_match else 0.0
        total_cost = float(total_cost_match.group(1)) if total_cost_match else 0.0
        avg_response_time = float(avg_time_match.group(1)) if avg_time_match else 0.0

        # Parse category breakdown
        # Format: - **Category Name**: X/Y passed (Z%)
        category_breakdown = {}
        category_pattern = re.compile(
            r"-\s*\*\*([^*]+)\*\*:\s*(\d+)/(\d+)\s*passed\s*\((\d+\.?\d*)%\)"
        )
        for match in category_pattern.finditer(content):
            category_name = match.group(1).strip()
            passed = int(match.group(2))
            total = int(match.group(3))
            percentage = float(match.group(4))
            category_breakdown[category_name] = {
                "passed": passed,
                "total": total,
                "percentage": percentage,
            }

        return {
            "total_queries": total_queries,
            "pass_rate": pass_rate,
            "validation_score": validation_score,
            "total_cost": total_cost,
            "avg_response_time": avg_response_time,
            "category_breakdown": category_breakdown,
        }

    except Exception as e:
        return {
            "total_queries": 0,
            "pass_rate": 0.0,
            "validation_score": 0.0,
            "total_cost": 0.0,
            "avg_response_time": 0.0,
            "category_breakdown": {},
            "error": f"Failed to parse validation report: {str(e)}",
        }
