import streamlit as st
import tempfile
from inference import main
import os

st.set_page_config(page_title="Cricket Commentary Generator", layout="centered")

st.title("🏏 Cricket Commentary Generator")
st.write("Upload a cricket video to generate AI-powered commentary with ambient sound.")

uploaded_video = st.file_uploader("Choose a video...", type=["mp4", "mov", "avi"])

if uploaded_video is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded_video.read())
        tmp_path = tmp.name

    st.video(tmp_path)

    if st.button("Generate Commentary"):
        with st.spinner("Processing video... Please wait."):
            output_path = main(tmp_path)

        if output_path and os.path.exists(output_path):
            st.success("✅ Commentary video generated!")
            st.video(output_path)
        else:
            st.error("⚠️ Failed to generate video.")

