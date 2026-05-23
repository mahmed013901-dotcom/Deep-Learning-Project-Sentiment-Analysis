import gradio as gr
import requests

API_URL="http://127.0.0.1:5000/predict"

def analyze(text):

    response=requests.post(
        API_URL,
        json={"text":text}
    )

    result=response.json()

    return (
        result["prediction"],
        str(result["confidence"])+"%",
        result["cleaned"]
    )


demo=gr.Interface(
    fn=analyze,
    inputs=gr.Textbox(
        lines=4,
        label="Enter text"
    ),

    outputs=[
        gr.Textbox(label="Sentiment"),
        gr.Textbox(label="Confidence"),
        gr.Textbox(label="Processed Text")
    ],

    title="Tweet Sentiment Analyzer"
)

demo.launch()