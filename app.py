import os
import time
import streamlit as st
from news_agent.retrieval import VectorIndex, encoder_from_metadata
from news_agent.rag import GeminiGenerator, answer_question

st.set_page_config(page_title="News Evidence Search", layout="wide")
st.title("News Evidence Search")
st.caption("Find source passages, then optionally ask Gemini to answer with citations.")


@st.cache_resource
def load_resources(path):
    index = VectorIndex.load(path)
    return index, encoder_from_metadata(index.metadata)


index_path = os.getenv("NEWS_INDEX_PATH", "artifacts/index")
try:
    index, encoder = load_resources(index_path)
except (OSError, ValueError, ImportError) as error:
    st.info("The news collection is not ready. Build an index using the README setup steps.")
    st.stop()

if index.metadata.get("demo"):
    st.info("Demo collection: fictional articles and lexical retrieval. These are not live news or model-quality results.")
with st.form("search"):
    question = st.text_input("What would you like to find?")
    k = st.slider("Number of passages", 1, 10, 5)
    generate_answer = st.checkbox("Generate a cited answer with Gemini", value=False)
    submitted = st.form_submit_button("Search")

if submitted and question.strip():
    start = time.perf_counter()
    evidence = index.search(encoder.encode([question])[0], k=k)
    if generate_answer:
        try:
            result = answer_question(question, evidence, GeminiGenerator())
            st.subheader("Answer")
            st.write(result["answer"])
            st.caption("Citations are checked against retrieved IDs. Factual support still requires source review.")
        except Exception:
            st.error("A cited answer could not be generated. Review the passages below or try again later.")
    st.subheader("Retrieved passages")
    for passage in evidence:
        with st.expander(f"{passage['title'] or passage['chunk_id']} · similarity {passage['score']:.3f}", expanded=True):
            st.write(passage["text"])
            st.caption(f"Citation: [{passage['chunk_id']}] · Published: {passage.get('published', '')}")
            source = passage.get("source", "")
            if source.startswith(("https://", "http://")):
                st.link_button("Read source", source)
    st.caption(f"Completed in {time.perf_counter() - start:.2f} seconds")
