import pandas as pd
import streamlit as st

from utils.ui import apply_global_styles, hero

st.set_page_config(page_title="Analysis History", page_icon="🕘", layout="wide")
apply_global_styles()
hero("Session Analysis History", "Predictions made during this browser session")

history = st.session_state.get("analysis_history", [])
if not history:
    st.info("No analyses have been completed in this session yet.")
    st.stop()

df = pd.DataFrame(history)
st.dataframe(df, use_container_width=True, hide_index=True)

st.download_button(
    "Download session history as CSV",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name="xray_analysis_history.csv",
    mime="text/csv",
    use_container_width=True,
)

if st.button("Clear session history"):
    st.session_state.analysis_history = []
    st.rerun()
