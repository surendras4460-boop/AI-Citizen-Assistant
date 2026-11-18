"""AI Citizen Assistant - Streamlit frontend.  Run: python -m streamlit run app.py"""
import os

from dotenv import load_dotenv
load_dotenv(override=True)

import pandas as pd
import streamlit as st

import importlib
import core
importlib.reload(core)

st.set_page_config(page_title="AI Citizen Assistant", page_icon="🏛️", layout="wide")

# ------------------------------------------------------------------ styling
st.markdown("""
<style>
.block-container {padding-top: 1.2rem; max-width: 1100px;}
.hero {padding: 1.4rem 1.6rem; border-radius: 16px; margin-bottom: 1rem; color: #fff;
       background: linear-gradient(120deg, #ff9933 0%, #e8590c 35%, #1b5e20 100%);}
.hero h1 {margin: 0; font-size: 1.9rem; color: #fff;}
.hero p {margin: .3rem 0 0; opacity: .95;}
.pill {display:inline-block; padding: 2px 10px; border-radius: 999px; font-size: .75rem;
       background: rgba(128,128,128,.18); margin-right: 6px;}
.src {border: 1px solid rgba(128,128,128,.3); border-radius: 10px; padding: .6rem .8rem; margin-bottom: .5rem;}
.src small {opacity: .75;}
.stat {text-align:center; padding:.5rem; border-radius:10px; background: rgba(128,128,128,.12);}
.stat b {font-size:1.3rem; display:block;}
</style>
<div class="hero">
  <h1>🏛️ AI Citizen Assistant</h1>
  <p>Ask about government schemes and your own documents · English &amp; हिंदी · answers with sources</p>
</div>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------ cached resources
@st.cache_resource(show_spinner="Loading embedding model...")
def get_embedder():
    return core.SentenceEmbedder()


@st.cache_data(show_spinner="Loading schemes dataset...")
def get_schemes_df():
    return core.load_schemes_df("schemes.csv").reset_index(drop=True)


@st.cache_resource(show_spinner="Indexing schemes dataset (first run only)...")
def get_scheme_index():
    chunks = core.schemes_to_chunks(get_schemes_df())
    idx = core.VectorIndex("schemes", get_embedder(), persist_path="chroma_schemes")
    idx.add(chunks)
    return idx


# ------------------------------------------------------------------ session state
ss = st.session_state
for k, v in {"messages": [], "user_chunks": [], "file_key": None, "user_index": None,
             "skipped": {}, "summary": None, "last_sources": []}.items():
    ss.setdefault(k, v)


def answer(question):
    indexes = [ss.user_index] + ([get_scheme_index()] if use_schemes else [])
    hits = core.retrieve(indexes, question)
    try:
        ans = core.generate(core.qa_prompt(question, hits, ss.messages))
    except Exception as e:  # API key missing, rate limit, network
        ans = f"⚠️ Could not reach the language model: {e}"
    return ans, hits


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("📂 Your documents")
    files = st.file_uploader("Upload PDFs (resume, marksheets, certificates, scheme guidelines)",
                             type="pdf", accept_multiple_files=True)
    use_schemes = st.toggle("Use government schemes dataset", value=True,
                            help="Indian schemes in English and Hindi")
    
    api_key_val = os.getenv("GEMINI_API_KEY")
    if not api_key_val:
        st.warning("⚠️ **GEMINI_API_KEY required for AI answers**")
        api_key_input = st.text_input("Enter Gemini API Key", type="password", placeholder="AIzaSy...",
                                      help="Get a free key from https://aistudio.google.com")
        if api_key_input:
            key_clean = api_key_input.strip()
            os.environ["GEMINI_API_KEY"] = key_clean
            env_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
            with open(env_file, "w", encoding="utf-8") as f:
                f.write(f"GEMINI_API_KEY={key_clean}\n")
            st.success("✅ Key saved to .env!")
            st.rerun()
        else:
            st.caption("Get your free key from [Google AI Studio](https://aistudio.google.com).")
    else:
        st.success("✅ GEMINI_API_KEY active")
    st.caption("Privacy: embedding and search run locally. Retrieved text and your question "
               "are sent to the Gemini API to generate answers.")
    if st.button("🗑️ Clear chat", use_container_width=True):
        ss.messages = []
        st.rerun()

# rebuild the per-session index only when the set of uploaded files changes
key = tuple(sorted((f.name, f.size) for f in files)) if files else None
if key != ss.file_key:
    ss.file_key, ss.messages, ss.user_chunks, ss.skipped, ss.summary = key, [], [], {}, None
    ss.user_index = core.VectorIndex("citizen_documents", get_embedder())
    for f in files or []:
        try:
            chunks, skipped = core.pdf_to_chunks(f, f.name)
        except Exception as e:
            st.sidebar.error(f"{f.name}: could not read ({e})")
            continue
        ss.user_chunks += chunks
        if skipped:
            ss.skipped[f.name] = skipped
    if ss.user_chunks:
        ss.user_index.add(ss.user_chunks)

with st.sidebar:
    c1, c2 = st.columns(2)
    c1.markdown(f"<div class='stat'><b>{len(files or [])}</b>files</div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='stat'><b>{len(ss.user_chunks)}</b>chunks</div>", unsafe_allow_html=True)
    for name, pages in ss.skipped.items():
        st.warning(f"**{name}**: pages {pages} have no text layer (scanned?) and were skipped.")

# chat input is declared at top level so Streamlit pins it to the bottom of the page
typed = st.chat_input("Ask about a scheme or your documents (English / हिंदी)...")
question = typed or ss.pop("pending_q", None)

tab_chat, tab_docs, tab_schemes = st.tabs(["💬 Ask", "📄 My documents", "🔎 Scheme explorer"])

# ------------------------------------------------------------------ tab 1: chat
with tab_chat:
    if not ss.messages and not question:
        st.markdown("**Try one of these:**")
        samples = ["What documents are required for a post matric scholarship?",
                   "Who is eligible for PM Kisan Samman Nidhi?",
                   "मुझे किसानों के लिए योजनाएँ बताइए",
                   "Which scheme documents am I missing based on my uploaded files?"]
        cols = st.columns(2)
        for i, s in enumerate(samples):
            if cols[i % 2].button(s, key=f"sample{i}", use_container_width=True):
                ss.pending_q = s
                st.rerun()

    for m in ss.messages:
        with st.chat_message(m["role"], avatar="🧑" if m["role"] == "user" else "🏛️"):
            st.markdown(m["content"])

    if question:
        with st.chat_message("user", avatar="🧑"):
            st.markdown(question)
        with st.chat_message("assistant", avatar="🏛️"):
            with st.spinner("Searching documents and generating answer..."):
                ans, hits = answer(question)
            st.markdown(ans)
        ss.messages += [{"role": "user", "content": question}, {"role": "assistant", "content": ans}]
        ss.last_sources = hits[:8]

    if ss.last_sources:
        with st.expander(f"📎 Sources used for the last answer ({len(ss.last_sources)} shown)"):
            st.caption("These are the retrieved passages, not independently verified evidence.")
            for h in ss.last_sources:
                kind = "Your document" if h.meta["kind"] == "user_doc" else "Schemes dataset"
                st.markdown(
                    f"<div class='src'><span class='pill'>{kind}</span>"
                    f"<b>{h.meta['source']}</b> · {h.meta['page']} "
                    f"<small>· similarity {h.meta['score']:.2f}</small><br>"
                    f"<small>{h.text[:220]}{'...' if len(h.text) > 220 else ''}</small></div>",
                    unsafe_allow_html=True)

# ------------------------------------------------------------------ tab 2: documents
with tab_docs:
    if not ss.user_chunks:
        st.info("Upload text-based PDFs in the sidebar, then press **Analyze**.")
    else:
        st.write(f"Loaded **{len(files)}** file(s) → **{len(ss.user_chunks)}** chunks.")
        if st.button("🔍 Analyze my documents", type="primary"):
            with st.spinner("Summarizing..."):
                try:
                    ss.summary = core.generate(core.summary_prompt(ss.user_chunks))
                except Exception as e:
                    ss.summary = f"⚠️ Could not reach the language model: {e}"
        if ss.summary:
            st.markdown(ss.summary)
            st.download_button("⬇️ Download summary", ss.summary, "document_summary.md")

# ------------------------------------------------------------------ tab 3: scheme explorer
with tab_schemes:
    try:
        df = get_schemes_df()
    except Exception as e:
        st.error(f"Could not load the schemes dataset: {e}")
        st.stop()

    f1, f2, f3, f4 = st.columns([3, 2, 2, 1.3])
    text = f1.text_input("Search scheme name", placeholder="e.g. scholarship, किसान")
    cats = sorted(df["schemeCategory"].dropna().unique()) if "schemeCategory" in df else []
    cat = f2.selectbox("Category", ["All"] + cats)
    levels = sorted(df["level"].dropna().unique()) if "level" in df else []
    level = f3.selectbox("Level", ["All"] + levels)
    lang = f4.radio("Language", ["EN", "हिंदी"], horizontal=True)

    view = df
    if text:
        mask = df["scheme_name"].fillna("").str.contains(text, case=False, regex=False)
        if "scheme_name_hindi" in df:
            mask |= df["scheme_name_hindi"].fillna("").str.contains(text, case=False, regex=False)
        view = view[mask]
    if cat != "All":
        view = view[view["schemeCategory"] == cat]
    if level != "All":
        view = view[view["level"] == level]

    st.caption(f"{len(view)} of {len(df)} schemes")
    for _, r in view.head(25).iterrows():
        title = (core._clean(r.get("scheme_name_hindi")) if lang == "हिंदी" else "") or core._clean(r.get("scheme_name"))
        with st.expander(f"{title}  ·  {core._clean(r.get('schemeCategory'))}"):
            body = core._clean(r.get("details_hi")) if lang == "हिंदी" else core._clean(r.get("details"))
            st.write(body or core._clean(r.get("details")))
            fields_config = [
                ("💰 Benefits" if lang != "हिंदी" else "💰 मुख्य लाभ", "benefits_hi" if lang == "हिंदी" else "benefits", "benefits"),
                ("✅ Eligibility" if lang != "हिंदी" else "✅ पात्रता मानदंड", "eligibility_hi" if lang == "हिंदी" else "eligibility", "eligibility"),
                ("📝 How to apply" if lang != "हिंदी" else "📝 आवेदन प्रक्रिया", "application_hi" if lang == "हिंदी" else "application", "application"),
                ("📄 Documents required" if lang != "हिंदी" else "📄 आवश्यक दस्तावेज़", "documents_hi" if lang == "हिंदी" else "documents", "documents"),
            ]
            for label, primary_field, fallback_field in fields_config:
                val = core._clean(r.get(primary_field)) or core._clean(r.get(fallback_field))
                if val:
                    st.markdown(f"**{label}**  \n{val}")
            if st.button("Ask the assistant about this scheme", key=f"ask_{r.name}"):
                ss.pending_q = f"Explain the {core._clean(r.get('scheme_name'))} scheme: benefits, eligibility and required documents."
                st.rerun()
    if len(view) > 25:
        st.caption("Showing the first 25 matches. Narrow the search to see others.")
