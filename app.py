import gradio as gr
from inference import main
import os

def launch():
    iface = gr.Interface(
        fn=main,
        inputs=gr.Video(label="Upload Cricket Video"),
        outputs=gr.Video(label="Generated Commentary Video"),
        title="Cricket Commentary Generator",
        description="Upload a cricket video to generate AI-powered commentary with ambient sound."
    )

demo.launch(
    server_name="0.0.0.0",
    server_port=int(os.environ.get("PORT", 7860))
)

