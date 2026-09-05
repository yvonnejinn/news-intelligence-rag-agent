import streamlit as st
import pandas as pd

st.set_page_config(page_title="News Label Review", layout="wide")
st.title("News Label Review")
st.write("Upload news, review each sentiment label, and download your annotations. "
         "Keep this reviewed set separate from training data.")
uploaded = st.file_uploader("News CSV", type=["csv"])
if uploaded is not None:
    frame = pd.read_csv(uploaded).fillna("")
    if not {"document_id", "text"}.issubset(frame):
        st.error("CSV must contain document_id and text.")
        st.stop()
    if frame["document_id"].duplicated().any() or frame["document_id"].eq("").any():
        st.error("Each row needs a unique nonempty document_id.")
        st.stop()
    for col in ("human_label", "reviewer", "notes"):
        if col not in frame:
            frame[col] = ""
    if "reviewed" not in frame:
        frame["reviewed"] = False
    else:
        frame["reviewed"] = frame["reviewed"].astype(str).str.lower().isin(["true", "1"])
    edited = st.data_editor(frame, hide_index=True, disabled=[c for c in frame if c not in
        {"human_label", "reviewer", "notes", "reviewed"}],
        column_config={"human_label": st.column_config.SelectboxColumn("Human label",
            options=["", "negative", "neutral", "positive"]),
            "reviewed": st.column_config.CheckboxColumn("Reviewed")})
    invalid = edited["reviewed"] & (~edited["human_label"].isin(["negative", "neutral", "positive"])
                                    | edited["reviewer"].str.strip().eq(""))
    if invalid.any():
        st.warning("Reviewed rows need a valid label and reviewer name before export.")
    st.download_button("Download annotations", edited.to_csv(index=False).encode("utf-8"),
                       "reviewed_news.csv", "text/csv", disabled=bool(invalid.any()))
