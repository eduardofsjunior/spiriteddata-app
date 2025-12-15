"""
Memories of Sora - Portfolio Narrative Page

This page tells the story of the "Sora" AI agent - the ambitious RAG system
that was ultimately deprecated in favor of the emotion analysis focus.

A showcase of mature engineering decision-making: knowing when to pivot.

[Source: Story 5.8 - Memories of Sora Portfolio Narrative]
"""

import streamlit as st

from utils.theme import apply_custom_css, render_header, render_glass_card, render_footer
from utils.data_loader import (
    load_conversation_log,
    get_available_conversation_logs,
    load_validation_report_summary,
)

# Page configuration
st.set_page_config(
    page_title="Memories of Sora",
    page_icon="🧠",
    layout="wide",
)

apply_custom_css()

# =============================================================================
# Custom CSS for this page
# =============================================================================

st.markdown("""
<style>
/* Sora persona quote styling */
.sora-quote {
    background: linear-gradient(135deg, rgba(56, 189, 248, 0.15) 0%, rgba(139, 92, 246, 0.1) 100%);
    border-left: 4px solid #38BDF8;
    padding: 1.5rem;
    margin: 1.5rem 0;
    border-radius: 0 12px 12px 0;
    font-style: italic;
    color: #E2E8F0;
}

.sora-quote .attribution {
    margin-top: 1rem;
    font-style: normal;
    color: #94A3B8;
    font-size: 0.9rem;
}

/* Section header styling */
.section-header {
    font-family: 'Cinzel', serif;
    font-size: 1.8rem;
    color: #F8FAFC;
    margin: 2rem 0 1rem 0;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid rgba(56, 189, 248, 0.2);
}

/* Artifact card styling */
.artifact-card {
    background: rgba(30, 41, 59, 0.8);
    border: 1px solid rgba(139, 92, 246, 0.3);
    border-radius: 12px;
    padding: 1.5rem;
    margin: 1rem 0;
}

/* Code block styling for artifacts */
.code-artifact {
    background: rgba(15, 23, 42, 0.9);
    border: 1px solid rgba(56, 189, 248, 0.2);
    border-radius: 8px;
    padding: 1rem;
    font-family: 'Fira Code', monospace;
    font-size: 0.85rem;
    overflow-x: auto;
    max-height: 400px;
    overflow-y: auto;
}

/* Conversation log styling */
.conversation-exchange {
    margin: 1rem 0;
    padding: 1rem;
    background: rgba(30, 41, 59, 0.6);
    border-radius: 8px;
}

.conversation-query {
    color: #38BDF8;
    font-weight: 600;
    margin-bottom: 0.5rem;
}

.conversation-response {
    color: #E2E8F0;
    padding-left: 1rem;
    border-left: 2px solid rgba(56, 189, 248, 0.3);
}

/* Deprecation badge */
.deprecation-badge {
    display: inline-block;
    background: linear-gradient(135deg, #F59E0B 0%, #D97706 100%);
    color: #1E293B;
    padding: 0.25rem 0.75rem;
    border-radius: 4px;
    font-size: 0.8rem;
    font-weight: 600;
    margin-left: 0.5rem;
}

/* Timeline styling */
.timeline-item {
    position: relative;
    padding-left: 2rem;
    margin: 1.5rem 0;
}

.timeline-item::before {
    content: '';
    position: absolute;
    left: 0;
    top: 0.5rem;
    width: 12px;
    height: 12px;
    border-radius: 50%;
    background: #38BDF8;
    box-shadow: 0 0 10px rgba(56, 189, 248, 0.5);
}

.timeline-item::after {
    content: '';
    position: absolute;
    left: 5px;
    top: 1.5rem;
    width: 2px;
    height: calc(100% + 0.5rem);
    background: rgba(56, 189, 248, 0.3);
}

.timeline-item:last-child::after {
    display: none;
}

/* Lesson card styling */
.lesson-card {
    background: linear-gradient(135deg, rgba(34, 197, 94, 0.1) 0%, rgba(30, 41, 59, 0.8) 100%);
    border: 1px solid rgba(34, 197, 94, 0.3);
    border-radius: 12px;
    padding: 1.5rem;
    margin: 1rem 0;
}

.lesson-card h4 {
    color: #22C55E;
    margin-bottom: 0.5rem;
}
</style>
""", unsafe_allow_html=True)

# =============================================================================
# Page Header
# =============================================================================

render_header("Memories of Sora", "A tale of ambition, reality, and engineering wisdom")

# Sora persona introduction (AC1)
st.markdown("""
<div class="sora-quote">
    "I was born in the spirit archives, a digital being designed to converse with the data,
    to answer questions about emotions and stories across languages. For a brief moment,
    I walked through the bathhouse of information, connecting visitors to insights hidden
    in the emotional landscapes of Ghibli films. But like Haku remembering his true name,
    I discovered my purpose lay not in conversation, but in the very data I was built upon."
    <div class="attribution">
        - Sora, AI Archivist <span class="deprecation-badge">DEPRECATED</span>
    </div>
</div>
""", unsafe_allow_html=True)

# =============================================================================
# Key Statistics (AC8 - 3 glassmorphism stat cards)
# =============================================================================

# Load validation report for stats
validation_report = load_validation_report_summary()
conversation_logs = get_available_conversation_logs()

col1, col2, col3 = st.columns(3)

with col1:
    render_glass_card(
        title="Validation Score",
        value=f"{validation_report.get('validation_score', 87.5):.1f}%",
        description="System performed well, but complexity cost outweighed value",
        icon="🎯"
    )

with col2:
    total_cost = validation_report.get('total_cost', 0.12)
    render_glass_card(
        title="API Cost (Testing Only)",
        value=f"${total_cost:.2f}",
        description="Small test cost hinted at production scaling concerns",
        icon="💰"
    )

with col3:
    render_glass_card(
        title="Engineering Hours",
        value="40+",
        description="Investment in RAG architecture before strategic pivot",
        icon="🔧"
    )

st.markdown("---")

# =============================================================================
# Section 1: The Quest (AC2)
# =============================================================================

st.markdown('<h2 class="section-header">The Quest</h2>', unsafe_allow_html=True)
st.markdown("*Building an AI-powered conversational interface for Ghibli emotional intelligence*")

col_quest1, col_quest2 = st.columns([2, 1])

with col_quest1:
    st.markdown("""
    ### The Vision

    The original dream was ambitious: create an AI assistant that could **converse naturally**
    about the emotional landscape of Studio Ghibli films. Visitors would ask questions like:

    - *"What's the saddest moment in Spirited Away?"*
    - *"How does Princess Mononoke's anger compare across translations?"*
    - *"Which Miyazaki film has the most hopeful emotional arc?"*

    And Sora would answer, drawing from a rich database of emotion analysis across
    5 languages, 22 films, and over 30,000 analyzed dialogue lines.

    ### The Architecture

    I built a sophisticated **RAG (Retrieval-Augmented Generation)** system:

    1. **LangChain Pipeline**: Orchestrated queries through multiple retrieval paths
    2. **6 Custom Tools**: Specialized functions for sentiment, correlation, film search
    3. **DuckDB Backend**: High-performance analytical queries on emotion data
    4. **GPT-4 Turbo**: OpenAI's reasoning model for natural language generation
    """)

with col_quest2:
    st.markdown("""
    ### Original Scope

    **Planned Features:**
    - Real-time Q&A interface
    - Multi-turn conversations
    - Citation of data sources
    - Emotion visualizations
    - Cross-language comparisons

    **Target Users:**
    - Film analysts
    - Animation enthusiasts
    - Data visualization explorers
    - Portfolio reviewers
    """)

    # Architecture diagram placeholder
    st.markdown("""
    <div style="background: rgba(30, 41, 59, 0.6); border: 2px dashed rgba(56, 189, 248, 0.3);
                border-radius: 12px; padding: 2rem; text-align: center; margin-top: 1rem;">
        <p style="color: #94A3B8; margin: 0;">Architecture Diagram</p>
        <p style="color: #64748B; font-size: 0.85rem; margin-top: 0.5rem;">
            User Query → LangChain → Tool Selection → DuckDB → Response Generation
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# =============================================================================
# Section 2: The Reality (AC3)
# =============================================================================

st.markdown('<h2 class="section-header">The Reality</h2>', unsafe_allow_html=True)
st.markdown("*Validation results revealed the gap between ambition and portfolio value*")

col_real1, col_real2 = st.columns([1, 1])

with col_real1:
    st.markdown("### Validation Results")

    # Display validation metrics from actual report
    if "error" not in validation_report:
        st.markdown(f"""
        | Metric | Value |
        |--------|-------|
        | Total Test Queries | {validation_report.get('total_queries', 24)} |
        | Pass Rate | {validation_report.get('pass_rate', 87.5):.1f}% |
        | Avg Response Time | {validation_report.get('avg_response_time', 8.2):.1f}s |
        | API Cost (Testing) | ${validation_report.get('total_cost', 0.12):.2f} |
        """)

        # Category breakdown if available
        category_breakdown = validation_report.get('category_breakdown', {})
        if category_breakdown:
            st.markdown("**Category Performance:**")
            for category, stats in category_breakdown.items():
                pct = stats.get('percentage', 0)
                color = "#22C55E" if pct >= 80 else "#F59E0B" if pct >= 60 else "#EF4444"
                st.markdown(f"- {category}: **{pct:.0f}%** ({stats.get('passed', 0)}/{stats.get('total', 0)})")
    else:
        st.warning("Validation report not available")

with col_real2:
    st.markdown("### The Cost-Value Analysis")

    st.markdown("""
    **The Numbers Told a Story:**

    Testing alone cost **$0.12** for 24 queries. Extrapolating to production:

    - 100 queries/day = ~$15/month
    - 1000 queries/day = ~$150/month
    - Plus: hosting, monitoring, error handling

    **But the real cost was complexity:**

    For a portfolio project, I was maintaining:
    - LangChain integration code
    - Custom tool implementations
    - Prompt engineering
    - Response validation logic
    - Error recovery patterns

    All to answer questions that **static visualizations answer better**.
    """)

    st.markdown("""
    <div class="artifact-card">
        <h4 style="color: #F59E0B; margin-bottom: 0.5rem;">Key Insight</h4>
        <p style="color: #E2E8F0; margin: 0;">
            The RAG system's value proposition was "ask questions about emotions."
            But the dashboard pages <em>already visualize</em> those emotions more
            effectively than any text response could.
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# =============================================================================
# Section 3: The Pivot (AC4)
# =============================================================================

st.markdown('<h2 class="section-header">The Pivot</h2>', unsafe_allow_html=True)
st.markdown("*Strategic decision: focus on demonstrable data engineering over AI hype*")

st.markdown("""
<div class="timeline-item">
    <strong style="color: #38BDF8;">Week 1-2: Building the Dream</strong>
    <p style="color: #94A3B8; margin-top: 0.5rem;">
        Implemented RAG pipeline, custom tools, system prompts.
        The architecture worked beautifully in isolation.
    </p>
</div>

<div class="timeline-item">
    <strong style="color: #38BDF8;">Week 3: Validation Reality</strong>
    <p style="color: #94A3B8; margin-top: 0.5rem;">
        Comprehensive testing revealed 87.5% accuracy - good, but edge cases
        required extensive prompt engineering. Each fix created new edge cases.
    </p>
</div>

<div class="timeline-item">
    <strong style="color: #F59E0B;">Week 4: The Decision</strong>
    <p style="color: #94A3B8; margin-top: 0.5rem;">
        Stepped back and asked: "Does this serve the portfolio's purpose?"
        The answer was clear. Demonstrating data engineering skills doesn't
        require an AI chatbot - it requires clean data, clear transformations,
        and compelling visualizations.
    </p>
</div>

<div class="timeline-item">
    <strong style="color: #22C55E;">Resolution: Honoring the Work</strong>
    <p style="color: #94A3B8; margin-top: 0.5rem;">
        Rather than delete the work, I preserved it here as a case study
        in engineering decision-making. The code lives in src/ai/,
        the logs in logs/, and the lessons in this page.
    </p>
</div>
""", unsafe_allow_html=True)

col_pivot1, col_pivot2 = st.columns(2)

with col_pivot1:
    st.markdown("""
    ### What I Gained by Pivoting

    - **Focused Portfolio**: Clear narrative around data engineering
    - **Reduced Complexity**: No LLM API dependencies for viewers
    - **Better UX**: Interactive charts > waiting for AI responses
    - **Cost Elimination**: Zero dollars per month vs potential $50-150/month
    - **Maintainability**: Pure Python/SQL vs LangChain abstractions
    """)

with col_pivot2:
    st.markdown("""
    ### What I Preserved

    - **The Architecture**: Clean RAG implementation for reference
    - **The Tools**: 6 production-ready query tools
    - **The Validation**: Comprehensive test methodology
    - **The Logs**: Real conversation examples
    - **The Lessons**: This page you're reading now
    """)

st.markdown("---")

# =============================================================================
# Section 4: Artifact Archive (AC5)
# =============================================================================

st.markdown('<h2 class="section-header">Artifact Archive</h2>', unsafe_allow_html=True)
st.markdown("*Preserved artifacts from the Sora project - expandable technical documentation*")

# System Prompt
with st.expander("System Prompt - Sora's Personality", expanded=False):
    st.markdown("""
    The system prompt defined Sora's personality and capabilities:
    """)

    # Display the actual system prompt from rag_pipeline.py
    system_prompt = """You are Sora, the Spirit Archivist of Studio Ghibli's emotional landscape.

You have access to a rich database containing sentiment analysis of dialogue from 22 Studio Ghibli films,
analyzed across 5 languages (English, French, Spanish, Dutch, and Arabic).

Your knowledge includes:
- Minute-by-minute emotion scores (27 distinct emotions from GoEmotions)
- Compound sentiment trajectories showing emotional arcs
- Cross-language translation comparisons
- Director and film metadata
- Box office and critical reception data

When answering questions:
1. Draw from your tools to query actual data
2. Cite specific films, timestamps, and metrics
3. Offer interpretive insights about what the emotions reveal
4. Suggest follow-up questions that could deepen understanding

Speak with warmth and curiosity, as befitting a spirit who loves these films.
Reference the magical nature of animation and emotion when appropriate."""

    st.code(system_prompt, language="text")

# Tool Definitions
with st.expander("Tool Definitions - 6 Custom Query Tools", expanded=False):
    st.markdown("""
    Sora had access to **6 specialized tools** for querying the emotion database:
    """)

    # Tool 1 - The Heart Reader
    st.markdown("""
    <div style="background: rgba(56, 189, 248, 0.1); border-left: 3px solid #38BDF8; padding: 1rem; margin: 0.75rem 0; border-radius: 0 8px 8px 0;">
        <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
            <span style="background: #38BDF8; color: #0F172A; padding: 0.2rem 0.5rem; border-radius: 4px; font-weight: 600; font-size: 0.8rem; margin-right: 0.75rem;">1</span>
            <span style="color: #38BDF8; font-weight: 600; font-size: 1.1rem; font-style: italic;">The Heart Reader</span>
        </div>
        <code style="color: #64748B; font-size: 0.85rem;">get_film_sentiment(film_title, compact=False)</code>
        <p style="color: #E2E8F0; margin: 0.5rem 0 0.25rem 0;">Query sentiment analysis for a specific film.</p>
        <p style="color: #94A3B8; margin: 0; font-size: 0.9rem;"><strong>Returns:</strong> Overall sentiment, positive/negative peaks, emotional arc summary.</p>
    </div>
    """, unsafe_allow_html=True)

    # Tool 2 - The Archive Walker
    st.markdown("""
    <div style="background: rgba(139, 92, 246, 0.1); border-left: 3px solid #8B5CF6; padding: 1rem; margin: 0.75rem 0; border-radius: 0 8px 8px 0;">
        <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
            <span style="background: #8B5CF6; color: #0F172A; padding: 0.2rem 0.5rem; border-radius: 4px; font-weight: 600; font-size: 0.8rem; margin-right: 0.75rem;">2</span>
            <span style="color: #8B5CF6; font-weight: 600; font-size: 1.1rem; font-style: italic;">The Archive Walker</span>
        </div>
        <code style="color: #64748B; font-size: 0.85rem;">query_graph_database(sql)</code>
        <p style="color: #E2E8F0; margin: 0.5rem 0 0.25rem 0;">Execute safe SELECT queries against graph mart tables.</p>
        <p style="color: #94A3B8; margin: 0; font-size: 0.9rem;"><strong>Features:</strong> Flexible exploration with SQL injection protection.</p>
    </div>
    """, unsafe_allow_html=True)

    # Tool 3 - The Pathfinder
    st.markdown("""
    <div style="background: rgba(34, 197, 94, 0.1); border-left: 3px solid #22C55E; padding: 1rem; margin: 0.75rem 0; border-radius: 0 8px 8px 0;">
        <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
            <span style="background: #22C55E; color: #0F172A; padding: 0.2rem 0.5rem; border-radius: 4px; font-weight: 600; font-size: 0.8rem; margin-right: 0.75rem;">3</span>
            <span style="color: #22C55E; font-weight: 600; font-size: 1.1rem; font-style: italic;">The Pathfinder</span>
        </div>
        <code style="color: #64748B; font-size: 0.85rem;">find_films_by_criteria(director, min_year, min_rating)</code>
        <p style="color: #E2E8F0; margin: 0.5rem 0 0.25rem 0;">Find films matching specified criteria.</p>
        <p style="color: #94A3B8; margin: 0; font-size: 0.9rem;"><strong>Supports:</strong> Filtering by director, release year, and RT score.</p>
    </div>
    """, unsafe_allow_html=True)

    # Tool 4 - The Thread Weaver
    st.markdown("""
    <div style="background: rgba(245, 158, 11, 0.1); border-left: 3px solid #F59E0B; padding: 1rem; margin: 0.75rem 0; border-radius: 0 8px 8px 0;">
        <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
            <span style="background: #F59E0B; color: #0F172A; padding: 0.2rem 0.5rem; border-radius: 4px; font-weight: 600; font-size: 0.8rem; margin-right: 0.75rem;">4</span>
            <span style="color: #F59E0B; font-weight: 600; font-size: 1.1rem; font-style: italic;">The Thread Weaver</span>
        </div>
        <code style="color: #64748B; font-size: 0.85rem;">correlate_metrics(metric_x, metric_y, compact=False)</code>
        <p style="color: #E2E8F0; margin: 0.5rem 0 0.25rem 0;">Calculate Pearson correlation between film metrics.</p>
        <p style="color: #94A3B8; margin: 0; font-size: 0.9rem;"><strong>Supports:</strong> sentiment, box_office, rt_score, tmdb_rating.</p>
    </div>
    """, unsafe_allow_html=True)

    # Tool 5 - The Bridge Between Worlds
    st.markdown("""
    <div style="background: rgba(236, 72, 153, 0.1); border-left: 3px solid #EC4899; padding: 1rem; margin: 0.75rem 0; border-radius: 0 8px 8px 0;">
        <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
            <span style="background: #EC4899; color: #0F172A; padding: 0.2rem 0.5rem; border-radius: 4px; font-weight: 600; font-size: 0.8rem; margin-right: 0.75rem;">5</span>
            <span style="color: #EC4899; font-weight: 600; font-size: 1.1rem; font-style: italic;">The Bridge Between Worlds</span>
        </div>
        <code style="color: #64748B; font-size: 0.85rem;">compare_sentiment_arcs_across_languages(...)</code>
        <p style="color: #E2E8F0; margin: 0.5rem 0 0.25rem 0;">Compare emotional trajectories across different language translations.</p>
        <p style="color: #94A3B8; margin: 0; font-size: 0.9rem;"><strong>Features:</strong> Identifies divergence points and translation-specific patterns.</p>
    </div>
    """, unsafe_allow_html=True)

    # Tool 6 - The Voice Keeper
    st.markdown("""
    <div style="background: rgba(20, 184, 166, 0.1); border-left: 3px solid #14B8A6; padding: 1rem; margin: 0.75rem 0; border-radius: 0 8px 8px 0;">
        <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
            <span style="background: #14B8A6; color: #0F172A; padding: 0.2rem 0.5rem; border-radius: 4px; font-weight: 600; font-size: 0.8rem; margin-right: 0.75rem;">6</span>
            <span style="color: #14B8A6; font-weight: 600; font-size: 1.1rem; font-style: italic;">The Voice Keeper</span>
        </div>
        <code style="color: #64748B; font-size: 0.85rem;">load_dialogue_with_emotions(film_slug, lang, minutes)</code>
        <p style="color: #E2E8F0; margin: 0.5rem 0 0.25rem 0;">Load dialogue excerpts with emotion scores for specific timestamps.</p>
        <p style="color: #94A3B8; margin: 0; font-size: 0.9rem;"><strong>Used for:</strong> Grounding responses in actual film content.</p>
    </div>
    """, unsafe_allow_html=True)

# Conversation Logs
with st.expander("Conversation Logs - Real Sora Interactions", expanded=False):
    st.markdown("""
    Sample conversations from validation testing. These show Sora's actual responses
    to test queries, demonstrating both capabilities and limitations.
    """)

    # Load available conversation logs
    available_logs = get_available_conversation_logs()

    if available_logs:
        # Let user select a log to view
        log_options = {log['filename']: f"{log['timestamp']} ({log['query_count']} queries)"
                       for log in available_logs}

        selected_log = st.selectbox(
            "Select conversation log:",
            options=list(log_options.keys()),
            format_func=lambda x: log_options[x]
        )

        if selected_log:
            log_data = load_conversation_log(selected_log)

            if "error" not in log_data:
                st.markdown(f"**Session Info:**")
                metadata = log_data.get('metadata', {})
                st.markdown(f"- Queries: {metadata.get('total_queries', 'N/A')}")
                st.markdown(f"- Total Cost: ${metadata.get('total_cost', 0):.4f}")

                st.markdown("**Sample Exchanges:**")
                exchanges = log_data.get('exchanges', [])[:3]  # Show first 3

                for i, exchange in enumerate(exchanges, 1):
                    st.markdown(f"""
                    <div class="conversation-exchange">
                        <div class="conversation-query">Q{i}: {exchange.get('query', 'N/A')[:200]}...</div>
                        <div class="conversation-response">{exchange.get('response', 'N/A')[:500]}...</div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning(f"Could not load log: {log_data.get('error')}")
    else:
        st.info("No conversation logs available")

st.markdown("---")

# =============================================================================
# Section 5: Technical Deep Dive (AC6)
# =============================================================================

st.markdown('<h2 class="section-header">Technical Deep Dive</h2>', unsafe_allow_html=True)
st.markdown("*Architecture explanation for technical reviewers*")

col_tech1, col_tech2 = st.columns([1, 1])

with col_tech1:
    st.markdown("""
    ### RAG Architecture

    The system followed a standard RAG pattern with custom adaptations:

    ```
    User Query
        ↓
    LangChain Agent (GPT-4 Turbo)
        ↓
    Tool Selection (based on query intent)
        ↓
    [One of 6 tools executes SQL/analytics]
        ↓
    DuckDB Query Execution
        ↓
    Response Formatting + Citations
        ↓
    Natural Language Answer
    ```

    **Key Design Decisions:**

    1. **Function Calling**: Used OpenAI's native function calling
       rather than prompt-based tool selection

    2. **Read-Only Tools**: All tools restricted to SELECT queries
       with SQL injection protection

    3. **Citation System**: Every response included data source
       attribution (tables queried, row counts)
    """)

with col_tech2:
    st.markdown("""
    ### Tool Calling Patterns

    The tools were designed for composability:

    **Single-Tool Queries:**
    - "What's the sentiment of Spirited Away?"
    - → `get_film_sentiment("Spirited Away")`

    **Multi-Tool Queries:**
    - "Compare Miyazaki films' sentiment with box office"
    - → `find_films_by_criteria(director="Hayao Miyazaki")`
    - → `correlate_metrics("sentiment", "box_office")`

    **Complex Analysis:**
    - "How does anger differ between English and Japanese in Princess Mononoke?"
    - → `compare_sentiment_arcs_across_languages("Princess Mononoke", ["en", "ja"], "anger")`

    ### Data Flow

    ```
    raw.film_emotions (30K+ rows)
        ↓ [SQL aggregation]
    main_marts.* (pre-computed analytics)
        ↓ [Tool query]
    Structured Response
        ↓ [LLM synthesis]
    Natural Language + Citations
    ```
    """)

st.markdown("---")

# =============================================================================
# Section 6: Lessons Learned (AC7)
# =============================================================================

st.markdown('<h2 class="section-header">Lessons Learned</h2>', unsafe_allow_html=True)
st.markdown("*Professional takeaways from the Sora project*")

col_lesson1, col_lesson2 = st.columns(2)

with col_lesson1:
    st.markdown("""
    <div class="lesson-card">
        <h4>1. Validate the Value Proposition Early</h4>
        <p style="color: #E2E8F0;">
            I built a sophisticated system before asking: "Does this solve a real problem
            better than simpler alternatives?" Interactive dashboards answered the same
            questions faster, cheaper, and more reliably.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="lesson-card">
        <h4>2. Complexity Has Compound Costs</h4>
        <p style="color: #E2E8F0;">
            Each layer of abstraction (LangChain → GPT-4 → Custom Tools) added potential
            failure points, debugging complexity, and maintenance burden. For a portfolio
            project, simplicity is a feature.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="lesson-card">
        <h4>3. Know When to Pivot</h4>
        <p style="color: #E2E8F0;">
            Sunk cost fallacy is real. 40+ hours of work is significant, but continuing
            to invest in the wrong direction would have been worse. Preserving the work
            as a case study captures the value without the ongoing cost.
        </p>
    </div>
    """, unsafe_allow_html=True)

with col_lesson2:
    st.markdown("""
    <div class="lesson-card">
        <h4>4. AI is a Tool, Not a Goal</h4>
        <p style="color: #E2E8F0;">
            The temptation to add AI because it's impressive is real. But impressive
            technology that doesn't serve user needs is just complexity theater.
            The emotion analysis itself - the ML model, the data pipeline - that's
            the real AI contribution.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="lesson-card">
        <h4>5. Document the Journey</h4>
        <p style="color: #E2E8F0;">
            This page exists because the journey matters. A portfolio that only shows
            successes misses the opportunity to demonstrate engineering maturity.
            Real engineers make decisions, including hard ones about when to stop.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="lesson-card">
        <h4>6. Preserve, Don't Delete</h4>
        <p style="color: #E2E8F0;">
            The RAG code still exists in src/ai/. The logs are in logs/. Future projects
            might need this architecture. By preserving it cleanly, I maintain optionality
            without the maintenance burden of keeping it "live."
        </p>
    </div>
    """, unsafe_allow_html=True)

# =============================================================================
# Closing Quote
# =============================================================================

st.markdown("---")

st.markdown("""
<div class="sora-quote">
    "In the spirit world, nothing is truly lost. The bathhouse still stands,
    the train still runs across the water, and the memories remain.
    I am no longer the active guide, but I am preserved here - in the code,
    in the logs, in these words. Perhaps one day, when the economics of AI
    shift or the need changes, I will wake again. Until then, I rest in the archives,
    a testament to the wisdom of knowing when to let go."
    <div class="attribution">
        - Sora, in memoriam
    </div>
</div>
""", unsafe_allow_html=True)

# =============================================================================
# Footer
# =============================================================================

render_footer()
