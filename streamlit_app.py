"""
Streamlit demo. Prompt + style selector + motion-scale slider + mp4 preview.
"""
import os
import tempfile

import streamlit as st

st.set_page_config(page_title="video-motion-adapter", layout="centered")
st.title("text -> short video (scaffold)")
st.caption("Draft UI. The underlying pipeline is not wired to a real backbone in this repo; see the README.")

prompt = st.text_area("prompt", value="a cat surfing on a rainbow, cinematic")
style = st.selectbox("style", options=["(none)", "anime", "cinematic"], index=0)
col1, col2, col3 = st.columns(3)
with col1:
    duration = st.slider("duration (s)", 1.0, 8.0, 4.0, 0.5)
with col2:
    fps = st.slider("fps", 6, 24, 12)
with col3:
    motion_scale = st.slider("motion scale", 0.0, 2.0, 1.0, 0.1)
cfg_scale = st.slider("cfg scale", 0.0, 15.0, 7.5)
seed = st.number_input("seed (blank = random)", value=42)

if st.button("generate", type="primary"):
    from src.inference.t2v import run as run_single

    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.close()
    with st.spinner("rendering..."):
        run_single(
            prompt=prompt,
            out_path=tmp.name,
            duration=duration,
            fps=int(fps),
            cfg_scale=cfg_scale,
            motion_scale=motion_scale,
            style=None if style == "(none)" else style,
            seed=int(seed) if seed else None,
        )
    st.video(tmp.name)
    with open(tmp.name, "rb") as f:
        st.download_button("download mp4", f.read(), file_name="clip.mp4", mime="video/mp4")
