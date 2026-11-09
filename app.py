
import gradio as gr
from transformers import pipeline

sentiment_model = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest"
)

neutral_phrases = [
    "okay", "ok", "just ok", "just okay",
    "average", "normal", "nothing special",
    "so so", "it's fine", "it is fine"
]

def analyze_sentiment(text):
    if not text or not text.strip():
        return "Please enter some text.", 0

    text = text.strip()

    if text.lower() in neutral_phrases:
        return "NEUTRAL (Rule-based)", 0

    result = sentiment_model(text)[0]
    label = result["label"].upper()
    confidence = round(result["score"] * 100, 2)

    return label, confidence

def analyze_multiple(text):
    if not text.strip():
        return [], "Please enter reviews."

    reviews = [r.strip() for r in text.split("\n") if r.strip()]
    results = []
    counts = {"POSITIVE": 0, "NEGATIVE": 0, "NEUTRAL": 0}

    for review in reviews:
        label, confidence = analyze_sentiment(review)
        sentiment = label.split()[0]
        if sentiment in counts:
            counts[sentiment] += 1
        results.append([review, label, confidence])

    summary = (
        f"Positive: {counts['POSITIVE']} | "
        f"Negative: {counts['NEGATIVE']} | "
        f"Neutral: {counts['NEUTRAL']}"
    )

    return results, summary

with gr.Blocks(title="Sentiment Analysis AI") as app:
    gr.Markdown("# Sentiment Analysis AI")
    gr.Markdown("Analyze text and customer reviews using AI.")

    with gr.Tab("Single Text Analysis"):
        input_text = gr.Textbox(
            label="Enter your text",
            lines=4,
            placeholder="Type your review here..."
        )
        analyze_btn = gr.Button("Analyze Sentiment")
        output_label = gr.Textbox(label="Sentiment")
        output_confidence = gr.Number(label="Confidence (%)")

        analyze_btn.click(
            analyze_sentiment,
            inputs=input_text,
            outputs=[output_label, output_confidence]
        )

    with gr.Tab("Multiple Reviews Analysis"):
        reviews_input = gr.Textbox(
            label="Enter reviews (one review per line)",
            lines=8
        )
        multiple_btn = gr.Button("Analyze Reviews")
        results_table = gr.Dataframe(
            headers=["Review", "Sentiment", "Confidence (%)"],
            label="Analysis Results"
        )
        summary_output = gr.Textbox(label="Summary")

        multiple_btn.click(
            analyze_multiple,
            inputs=reviews_input,
            outputs=[results_table, summary_output]
        )

if __name__ == "__main__":
   app.launch(server_name="0.0.0.0", server_port=10000)
