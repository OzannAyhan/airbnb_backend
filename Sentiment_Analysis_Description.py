

import torch
from transformers import pipeline
from tqdm import tqdm

def analyze_sentiment(combined_data, batch_size=32):
    """
    Performs sentiment analysis on property descriptions.

    Args:
    - combined_data: The DataFrame containing property descriptions.
    - batch_size: The size of batches for processing descriptions.

    Returns:
    - combined_data: The updated DataFrame with added 'Positivity_Score(1to5)' and 'sentiment_score' columns.
    """
    # Use a publicly available multilingual sentiment analysis model
    model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
    device = 0 if torch.cuda.is_available() else -1

    # Initialize the sentiment analysis pipeline
    sentiment_analyzer = pipeline(
        "sentiment-analysis",
        model=model_name,
        tokenizer=model_name,
        device=device,
        framework='pt',
        truncation=True
    )

    def split_text(text, max_length=512):
        """Split text into chunks that fit within the maximum length."""
        words = text.split()
        return [' '.join(words[i:i + max_length]) for i in range(0, len(words), max_length)]

    def get_sentiment_scores_batch(texts, batch_size=batch_size):
        """Get sentiment scores for a batch of texts."""
        sentiments = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Processing Batches"):
            batch = texts[i:i+batch_size]
            batch_split = [chunk for text in batch for chunk in split_text(text)]
            batch_results = sentiment_analyzer(batch_split)
            sentiments.extend(batch_results)
        return sentiments

    def process_sentiment_results(results):
        """Process sentiment results to extract labels and scores."""
        labels = [result['label'] for result in results]
        scores = [result['score'] for result in results]
        return labels, scores

    # Apply sentiment analysis in batches
    descriptions = combined_data['description'].tolist()
    sentiment_results = get_sentiment_scores_batch(descriptions, batch_size)
    sentiment_labels, sentiment_scores = process_sentiment_results(sentiment_results)

    # Add the sentiment results to the DataFrame
    combined_data['Positivity_Score(1to5)'] = sentiment_labels
    combined_data['sentiment_score'] = sentiment_scores

    return combined_data