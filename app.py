"""
Multi-Agent Car Assistant - Streamlit UI
Luxury automotive theme with full agent pipeline visualization

Run: streamlit run app.py
"""

import streamlit as st
import time
import html

# ============================================================
# AGENT IMPORTS - Real agents from the project
# ============================================================
try:
    from agents.router_agent import RouterAgent
    from agents.retriever_agent import RetrieverAgent
    from agents.advisor_agent import AdvisorAgent
    from agents.monitor_agent import MonitorAgent
    AGENTS_AVAILABLE = True
except ImportError as e:
    AGENTS_AVAILABLE = False
    IMPORT_ERROR = str(e)


# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="AUTOPILOT — Multi-Agent Car Assistant",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# AGENT INITIALIZATION (cached - loads only once)
# ============================================================
@st.cache_resource
def init_agents():
    """Initialize all 4 agents and build the index. Cached so it runs once."""
    router = RouterAgent()
    retriever = RetrieverAgent()
    retriever.build_index()  # Loads or builds ChromaDB
    advisor = AdvisorAgent()
    monitor = MonitorAgent()
    return router, retriever, advisor, monitor


# ============================================================
# LUXURY AUTOMOTIVE CSS
# ============================================================
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(180deg, #0a0a0f 0%, #14141c 100%);
        color: #e8e6e1;
    }

    #MainMenu, footer, header {visibility: hidden;}

    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
        max-width: 1100px;
    }

    .brand-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 20px 28px;
        background: #0f0f17;
        border: 1px solid #2a2a35;
        border-radius: 12px;
        margin-bottom: 24px;
    }

    .brand-left {
        display: flex;
        align-items: center;
        gap: 16px;
    }

    .brand-logo {
        width: 48px;
        height: 48px;
        border-radius: 50%;
        background: #1a1a25;
        border: 2px solid #c9a961;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 22px;
    }

    .brand-title {
        font-size: 22px;
        font-weight: 600;
        letter-spacing: 3px;
        color: #f5f0e1;
        margin: 0;
    }

    .brand-sub {
        font-size: 11px;
        color: #8a8578;
        letter-spacing: 2px;
        margin: 0;
    }

    .status-badge {
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 8px 14px;
        background: #14141c;
        border: 1px solid #2a2a35;
        border-radius: 20px;
    }

    .status-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #6dbf6d;
        box-shadow: 0 0 8px #6dbf6d;
    }

    .status-text {
        font-size: 11px;
        color: #8a8578;
        letter-spacing: 1.5px;
    }

    .category-strip {
        display: flex;
        gap: 8px;
        margin-bottom: 20px;
        flex-wrap: wrap;
    }

    .cat-chip {
        background: #14141c;
        color: #8a8578;
        padding: 8px 16px;
        border-radius: 20px;
        font-size: 11px;
        border: 1px solid #2a2a35;
        letter-spacing: 1px;
        font-weight: 500;
    }

    .cat-chip.active {
        background: #c9a961;
        color: #0a0a0f;
        border-color: #c9a961;
    }

    .stTextInput > div > div > input {
        background: #14141c !important;
        color: #f5f0e1 !important;
        border: 1px solid #2a2a35 !important;
        border-radius: 8px !important;
        padding: 14px 18px !important;
        font-size: 15px !important;
    }

    .stTextInput > div > div > input:focus {
        border-color: #c9a961 !important;
        box-shadow: 0 0 0 1px #c9a961 !important;
    }

    .stTextInput label {
        color: #8a8578 !important;
        font-size: 11px !important;
        letter-spacing: 1.5px !important;
        text-transform: uppercase;
    }

    .stButton > button {
        background: #c9a961 !important;
        color: #0a0a0f !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 12px 32px !important;
        font-weight: 600 !important;
        letter-spacing: 1.5px !important;
        font-size: 13px !important;
        text-transform: uppercase;
        transition: all 0.2s;
    }

    .stButton > button:hover {
        background: #d4b876 !important;
        transform: translateY(-1px);
    }

    .agent-card {
        background: #14141c;
        border: 1px solid #2a2a35;
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 10px;
        position: relative;
        overflow: hidden;
    }

    .agent-card::before {
        content: '';
        position: absolute;
        left: 0;
        top: 0;
        bottom: 0;
        width: 3px;
        background: #c9a961;
    }

    .agent-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 8px;
    }

    .agent-name {
        display: flex;
        align-items: center;
        gap: 12px;
        font-size: 13px;
        color: #f5f0e1;
        letter-spacing: 1px;
        font-weight: 500;
    }

    .agent-num {
        width: 24px;
        height: 24px;
        border-radius: 50%;
        background: #c9a961;
        color: #0a0a0f;
        font-size: 11px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 600;
    }

    .agent-status {
        font-size: 10px;
        color: #6dbf6d;
        letter-spacing: 1px;
    }

    .agent-body {
        padding-left: 36px;
        font-size: 12px;
        color: #8a8578;
        line-height: 1.7;
    }

    .agent-body .accent {
        color: #c9a961;
    }

    .agent-body .value {
        color: #f5f0e1;
    }

    .chunk-row {
        margin: 10px 0;
        padding: 12px 14px;
        background: #1a1a25;
        border-radius: 6px;
        border-left: 2px solid #c9a961;
    }

    .chunk-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 6px;
        font-size: 12px;
    }

    .chunk-header-left {
        color: #f5f0e1;
        font-weight: 500;
    }

    .chunk-header-left .num {
        color: #c9a961;
        margin-right: 6px;
    }

    .chunk-sim {
        color: #c9a961;
        font-size: 11px;
        font-weight: 500;
    }

    .chunk-text {
        font-size: 11px;
        color: #a8a59a;
        line-height: 1.6;
        font-style: italic;
    }

    .final-answer {
        background: linear-gradient(180deg, #1a1a25 0%, #14141c 100%);
        border: 1px solid #c9a961;
        border-radius: 12px;
        padding: 24px 28px;
        margin-top: 20px;
        position: relative;
    }

    .final-label {
        font-size: 10px;
        color: #c9a961;
        letter-spacing: 2px;
        margin-bottom: 12px;
        font-weight: 600;
    }

    .final-text {
        font-size: 15px;
        color: #f5f0e1;
        line-height: 1.8;
        white-space: pre-wrap;
    }

    .final-meta {
        display: flex;
        gap: 24px;
        margin-top: 16px;
        padding-top: 16px;
        border-top: 1px solid #2a2a35;
        font-size: 11px;
        color: #8a8578;
        flex-wrap: wrap;
    }

    .final-meta .accent {
        color: #c9a961;
        font-weight: 500;
    }

    .final-meta .safe {
        color: #6dbf6d;
        font-weight: 500;
    }

    .final-meta .warn {
        color: #e89c4d;
        font-weight: 500;
    }

    .query-display {
        background: #14141c;
        border-left: 3px solid #c9a961;
        padding: 14px 18px;
        border-radius: 0 8px 8px 0;
        margin-bottom: 16px;
    }

    .query-label {
        font-size: 10px;
        color: #8a8578;
        letter-spacing: 2px;
        margin-bottom: 4px;
    }

    .query-text {
        font-size: 15px;
        color: #f5f0e1;
    }

    .section-title {
        font-size: 11px;
        color: #8a8578;
        letter-spacing: 2px;
        margin: 24px 0 12px;
        text-transform: uppercase;
    }

    .stSpinner > div {
        border-top-color: #c9a961 !important;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# HEADER
# ============================================================
st.markdown("""
<div class="brand-header">
    <div class="brand-left">
        <div class="brand-logo">🏎️</div>
        <div>
            <div class="brand-title">AUTOPILOT</div>
            <div class="brand-sub">MULTI-AGENT CAR ASSISTANT · RAG POWERED</div>
        </div>
    </div>
    <div class="status-badge">
        <div class="status-dot"></div>
        <div class="status-text">SYSTEM ONLINE</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# CHECK AGENTS LOADED
# ============================================================
if not AGENTS_AVAILABLE:
    st.error(f"⚠️ Agents could not be loaded. Error: {IMPORT_ERROR}")
    st.info("Make sure app.py is in the project root directory (where the agents/ folder is).")
    st.stop()


# ============================================================
# INITIALIZE AGENTS (cached)
# ============================================================
with st.spinner("Initializing agents and loading knowledge base..."):
    router_agent, retriever_agent, advisor_agent, monitor_agent = init_agents()


# ============================================================
# SESSION STATE
# ============================================================
if "query_input" not in st.session_state:
    st.session_state.query_input = ""
if "auto_run" not in st.session_state:
    st.session_state.auto_run = False


def set_example(text):
    st.session_state.query_input = text
    st.session_state.auto_run = True


# ============================================================
# QUERY INPUT
# ============================================================
col1, col2 = st.columns([5, 1])

with col1:
    query = st.text_input(
        "Your Question",
        placeholder="e.g. My engine is overheating, what should I do?",
        label_visibility="visible",
        key="query_input"
    )

with col2:
    st.write("")
    st.write("")
    submit = st.button("ANALYZE", use_container_width=True)


# ============================================================
# EXAMPLE QUERIES
# ============================================================
st.markdown('<div class="section-title">◆ Example Questions</div>', unsafe_allow_html=True)

examples = [
    "My engine is overheating, what should I do?",
    "Recommend a cheap first car",
    "Compare Toyota Corolla and Honda Civic",
    "When should I change my oil?",
    "There's a knocking noise in my engine",
]

ex_cols = st.columns(len(examples))
for i, ex in enumerate(examples):
    with ex_cols[i]:
        st.button(ex, key=f"ex_{i}", use_container_width=True,
                  on_click=set_example, args=(ex,))


# ============================================================
# MAIN PIPELINE
# ============================================================
should_run = submit or st.session_state.auto_run
active_query = query

if should_run and active_query:
    st.session_state.auto_run = False

    # Show the query
    st.markdown(f"""
    <div class="query-display">
        <div class="query-label">▸ YOUR QUERY</div>
        <div class="query-text">{html.escape(active_query)}</div>
    </div>
    """, unsafe_allow_html=True)

    categories = ["BUY", "COMPARE", "DIAGNOSE", "MAINTENANCE", "GENERAL"]

    # ---- STEP 1: ROUTER ----
    with st.spinner("Router Agent is processing..."):
        time.sleep(0.3)
        route_result = router_agent.route(active_query)
        category = route_result.category
        confidence = route_result.confidence
        keywords = route_result.matched_keywords

    # Show active category chip
    cat_html = '<div class="category-strip">'
    for c in categories:
        active = "active" if c.lower() == category.lower() else ""
        cat_html += f'<div class="cat-chip {active}">{c}</div>'
    cat_html += '</div>'
    st.markdown(cat_html, unsafe_allow_html=True)

    keywords_str = ', '.join(keywords) if keywords else '—'
    st.markdown(f"""
    <div class="agent-card">
        <div class="agent-header">
            <div class="agent-name">
                <div class="agent-num">1</div>
                ROUTER AGENT
            </div>
            <div class="agent-status">● COMPLETE</div>
        </div>
        <div class="agent-body">
            Category: <span class="accent">{html.escape(category)}</span> &nbsp;·&nbsp;
            Confidence: <span class="value">{int(confidence*100)}%</span> &nbsp;·&nbsp;
            Keywords: <span class="value">{html.escape(keywords_str)}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ---- STEP 2: RETRIEVER ----
    with st.spinner("Retriever Agent is searching ChromaDB..."):
        time.sleep(0.3)
        retrieval_result = retriever_agent.retrieve(active_query, category)
        chunks = retrieval_result.chunks

    # Render retriever card header
    st.markdown(f"""
    <div class="agent-card">
        <div class="agent-header">
            <div class="agent-name">
                <div class="agent-num">2</div>
                RETRIEVER AGENT (RAG)
            </div>
            <div class="agent-status">● {len(chunks)} CHUNKS RETRIEVED</div>
        </div>
        <div class="agent-body">
    """, unsafe_allow_html=True)

    # Render each chunk separately (this fixes the HTML render bug)
    for i, chunk in enumerate(chunks[:3], 1):
        src = html.escape(chunk.source)
        sim = chunk.score
        text_preview = chunk.text.replace("\n", " ").strip()
        if len(text_preview) > 120:
            text_preview = text_preview[:120] + "..."
        text_preview = html.escape(text_preview)

        st.markdown(f"""
        <div class="chunk-row">
            <div class="chunk-header">
                <div class="chunk-header-left">
                    <span class="num">[{i}]</span>{src}
                </div>
                <div class="chunk-sim">sim={sim:.3f}</div>
            </div>
            <div class="chunk-text">{text_preview}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div></div>", unsafe_allow_html=True)

    # ---- STEP 3: ADVISOR ----
    with st.spinner("Advisor Agent is generating a response..."):
        time.sleep(0.3)
        advice_result = advisor_agent.advise(retrieval_result)
        answer = advice_result.answer
        mode = advice_result.mode

    st.markdown(f"""
    <div class="agent-card">
        <div class="agent-header">
            <div class="agent-name">
                <div class="agent-num">3</div>
                ADVISOR AGENT
            </div>
            <div class="agent-status">● GROUNDED RESPONSE</div>
        </div>
        <div class="agent-body">
            Mode: <span class="accent">{html.escape(mode)}</span> &nbsp;·&nbsp;
            Length: <span class="value">{len(answer)} chars</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ---- STEP 4: MONITOR ----
    with st.spinner("Monitor Agent is verifying the answer..."):
        time.sleep(0.3)
        monitor_result = monitor_agent.check_response(
            question=active_query,
            answer=answer,
            context=retrieval_result.context_text
        )
        is_safe = monitor_result.is_safe
        is_grounded = monitor_result.is_grounded
        feedback = monitor_result.feedback

    status_text = "SAFE · GROUNDED" if (is_safe and is_grounded) else "WARNING DETECTED"
    status_color = "#6dbf6d" if (is_safe and is_grounded) else "#e89c4d"

    st.markdown(f"""
    <div class="agent-card">
        <div class="agent-header">
            <div class="agent-name">
                <div class="agent-num">4</div>
                MONITOR AGENT
            </div>
            <div class="agent-status" style="color: {status_color};">● {status_text}</div>
        </div>
        <div class="agent-body">
            Safety: <span class="accent">{'PASS' if is_safe else 'FAIL'}</span> &nbsp;·&nbsp;
            Grounded: <span class="accent">{'YES' if is_grounded else 'NO'}</span>
            <div style="margin-top: 6px; font-size: 11px;">{html.escape(feedback)}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ---- FINAL ANSWER ----
    sources = list(set([c.source for c in chunks])) if chunks else ["—"]
    sources_str = ', '.join(sources)
    safety_label = "VERIFIED" if is_safe else "FLAGGED"
    safety_class = "safe" if is_safe else "warn"
    grounded_label = "GROUNDED" if is_grounded else "LOW OVERLAP"
    grounded_class = "safe" if is_grounded else "warn"

    st.markdown(f"""
    <div class="final-answer">
        <div class="final-label">◆ ADVISOR RESPONSE</div>
        <div class="final-text">{html.escape(answer)}</div>
        <div class="final-meta">
            <span>Sources: <span class="accent">{html.escape(sources_str)}</span></span>
            <span>Safety: <span class="{safety_class}">{safety_label}</span></span>
            <span>Groundedness: <span class="{grounded_class}">{grounded_label}</span></span>
        </div>
    </div>
    """, unsafe_allow_html=True)

elif not active_query:
    st.markdown("""
    <div style="text-align: center; padding: 60px 20px; color: #8a8578;">
        <div style="font-size: 48px; margin-bottom: 16px;">⚙️</div>
        <div style="font-size: 14px; letter-spacing: 1.5px;">
            Enter a question above or pick one of the examples
        </div>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# FOOTER
# ============================================================
st.markdown("""
<div style="text-align: center; margin-top: 40px; padding-top: 20px; border-top: 1px solid #2a2a35;
            font-size: 10px; color: #5a5852; letter-spacing: 2px;">
    POWERED BY CHROMADB · SENTENCE-TRANSFORMERS · OPENAI
</div>
""", unsafe_allow_html=True)
