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

    iface.launch(
        server_name="0.0.0.0",  # Required for Lightning.ai
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", 7860)),
        debug=True
    )

if __name__ == "__main__":
    launch()
