
import os
import gradio as gr
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

def analyze_sentiment(text):
    if not text or not text.strip():
        return "Please enter some text.", "N/A"

    score = analyzer.polarity_scores(text)["compound"]

    if score >= 0.05:
        sentiment = "Positive 😊"
    elif score <= -0.05:
        sentiment = "Negative 😞"
    else:
        sentiment = "Neutral 😐"

    return sentiment, round(score, 3)


def analyze_multiple(text):
    if not text.strip():
        return []

    results = []
    for review in text.split("\n"):
        if review.strip():
            sentiment, score = analyze_sentiment(review)
            results.append([review, sentiment, score])

    return results


with gr.Blocks(title="Sentiment Analysis AI") as app:
    gr.Markdown("# Sentiment Analysis AI")
    gr.Markdown("Analyze text and customer reviews using AI.")

    with gr.Tab("Single Text Analysis"):
        user_text = gr.Textbox(
            label="Enter your text",
            placeholder="Type your review here..."
        )
        analyze_btn = gr.Button("Analyze Sentiment")
        result = gr.Textbox(label="Sentiment")
        score = gr.Textbox(label="Compound Score")

        analyze_btn.click(
            analyze_sentiment,
            inputs=user_text,
            outputs=[result, score]
        )

    with gr.Tab("Multiple Reviews"):
        reviews = gr.Textbox(
            label="Enter reviews (one per line)",
            lines=6
        )
        batch_btn = gr.Button("Analyze Reviews")
        output = gr.Dataframe(
            headers=["Review", "Sentiment", "Compound Score"],
            interactive=False
        )

        batch_btn.click(
            analyze_multiple,
            inputs=reviews,
            outputs=output
        )


app.launch(
    server_name="0.0.0.0",
    server_port=int(os.environ.get("PORT", 10000)),
    share=False
)
