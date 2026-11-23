import json
from pathlib import Path

import streamlit as st

from src import config
from src.generator import RAGPipeline
from src.ingestion import sync_documents

st.set_page_config(
    page_title="Hybrid RAG Engine",
    page_icon="📚",
    layout="wide",
)


@st.cache_resource
def load_pipeline() -> RAGPipeline:
    if not config.CHUNKS_PATH.exists():
        sync_documents()
    return RAGPipeline()


def get_indexed_files() -> dict:
    if config.MANIFEST_PATH.exists():
        return json.loads(config.MANIFEST_PATH.read_text(encoding="utf-8"))
    return {}


# --- Sidebar: Controls & Document Management ---
with st.sidebar:
    st.title("⚙️ Engine Controls")

    mode = st.selectbox(
        "Retrieval Mode",
        options=["hybrid", "hybrid_rerank", "vector"],
        index=0,
        help="Benchmarked default: 'hybrid' (0.875 MRR @ 170ms).",
    )
    top_k = st.slider("Top-K Chunks", min_value=2, max_value=8, value=4)
    use_cache = st.toggle("Enable SQLite Cache", value=True)

    st.divider()
    st.subheader("📄 Knowledge Base")

    uploaded_files = st.file_uploader(
        "Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if uploaded_files and st.button("Sync Uploaded PDFs", use_container_width=True):
        with st.spinner("Parsing tables & syncing SHA-256 index..."):
            for uf in uploaded_files:
                dest = config.DATA_DIR / Path(uf.name).name
                dest.write_bytes(uf.getbuffer())
            stats = sync_documents()
            st.cache_resource.clear()
        st.success(
            f"Synced! Indexed: {stats['indexed']} | Skipped: {stats['skipped']} | "
            f"Total chunks: {stats['total_chunks']}"
        )

    manifest = get_indexed_files()
    if manifest:
        st.caption(f"Indexed Documents ({len(manifest)}):")
        for fname, info in manifest.items():
            st.markdown(f"- `{fname}` ({info['num_chunks']} chunks)")
    else:
        st.warning("No PDFs indexed yet. Upload a PDF above.")

# --- Main Chat Interface ---
st.title("📚 Hybrid RAG Document Engine")
st.caption(
    "Local ChromaDB + Regex BM25 + RRF + Optional TinyBERT Reranker | "
    "Persistent Versioned SQLite Cache | Streaming Gemini"
)

try:
    rag = load_pipeline()
except Exception as exc:
    st.error(f"Initialization Error: {exc}")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("meta"):
            m = msg["meta"]
            st.caption(
                f"Mode: `{m['mode']}` | Model: `{m['model_used']}` | "
                f"Cache Hit: `{m['cache_hit']}` | Retrieval: `{m['retrieval_ms']}ms` | "
                f"TTFT: `{m['ttft_ms']}ms` | Total: `{m['total_ms']}ms`"
            )
            if m.get("sources"):
                with st.expander(f"View {len(m['sources'])} Retrieved Sources"):
                    for i, src in enumerate(m["sources"], start=1):
                        tbl_tag = " | Table" if src.get("has_table") else ""
                        st.markdown(
                            f"**#{i} — `{src['source']}` (Page {src['page']}{tbl_tag})** "
                            f"· Score: `{src['score']}`"
                        )
                        st.code(src["text"], language="markdown")

if prompt := st.chat_input("Ask a question about your indexed documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        stream, meta = rag.stream_answer(
            prompt, mode=mode, top_k=top_k, use_cache=use_cache
        )
        full_response = st.write_stream(stream)

        st.caption(
            f"Mode: `{meta['mode']}` | Model: `{meta['model_used']}` | "
            f"Cache Hit: `{meta['cache_hit']}` | Retrieval: `{meta['retrieval_ms']}ms` | "
            f"TTFT: `{meta['ttft_ms']}ms` | Total: `{meta['total_ms']}ms`"
        )

        if meta.get("sources"):
            with st.expander(f"View {len(meta['sources'])} Retrieved Sources"):
                for i, src in enumerate(meta["sources"], start=1):
                    tbl_tag = " | Table" if src.get("has_table") else ""
                    st.markdown(
                        f"**#{i} — `{src['source']}` (Page {src['page']}{tbl_tag})** "
                        f"· Score: `{src['score']}`"
                    )
                    st.code(src["text"], language="markdown")

    st.session_state.messages.append(
        {"role": "assistant", "content": full_response, "meta": meta}
    )