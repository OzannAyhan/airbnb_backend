import os
import pandas as pd
import torch
from transformers import pipeline
from tqdm import tqdm

def process_city_reviews(combined_data, city, dataset_dir='/content', start_date='2023-07-01', end_date='2024-06-30', filename='florence_final_data.csv'):
    """
    Load, preprocess, perform sentiment analysis on reviews, and merge results into combined_data DataFrame for a given city.

    Args:
    - city: Name of the city to load reviews for.
    - dataset_dir: Directory where the dataset CSV file is located.
    - start_date: Start date for filtering reviews (format 'YYYY-MM-DD').
    - end_date: End date for filtering reviews (format 'YYYY-MM-DD').
    - filename: Filename of the combined data CSV file.

    Returns:
    - combined_data: DataFrame with added 'Positivity_Scores(1to5)' column.
    """

    # Function to optimize memory usage
    def optimize_memory(df):
        for col in df.select_dtypes(include=['float']):
            df[col] = pd.to_numeric(df[col], downcast='float')
        for col in df.select_dtypes(include=['int']):
            df[col] = pd.to_numeric(df[col], downcast='integer')
        for col in df.select_dtypes(include=['object']):
            num_unique_values = len(df[col].unique())
            num_total_values = len(df[col])
            if num_unique_values / num_total_values < 0.5:
                df[col] = df[col].astype('category')
        return df

    # Load and preprocess reviews
    base_path = os.path.join(dataset_dir, f"{city}_reviews{{}}.csv")
    reviews = []
    for i in range(1, 5):
        file_path = base_path.format(i)
        df = pd.read_csv(file_path)
        df = optimize_memory(df)
        reviews.append(df)
    florence_reviews = pd.concat(reviews, ignore_index=True)

    # Drop unnecessary columns
    florence_reviews.drop(columns=['reviewer_id', 'reviewer_name', 'id'], inplace=True)

    # Ensure 'date' column is in datetime format
    florence_reviews['date'] = pd.to_datetime(florence_reviews['date'], errors='coerce')

    # Define the start and end dates for filtering
    start_date = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date)

    # Filter the DataFrame to include only rows within the date range
    florence_reviews = florence_reviews[(florence_reviews['date'] >= start_date) & (florence_reviews['date'] <= end_date)]

    # Create a set of ids from combined_data
    ids_set = set(combined_data['id'])

    # Filter florence_reviews to keep only rows where listing_id is in ids_set
    florence_reviews = florence_reviews[florence_reviews['listing_id'].isin(ids_set)]

    # Check if GPU is available and set device
    device = 0 if torch.cuda.is_available() else -1

    # Use a publicly available multilingual sentiment analysis model
    model_name = "nlptown/bert-base-multilingual-uncased-sentiment"

    # Initialize the sentiment analysis pipeline
    sentiment_analyzer = pipeline("sentiment-analysis", model=model_name, tokenizer=model_name, device=device)

    # Function to clean and prepare comments
    def prepare_comments(comments):
        return [str(comment) if pd.notna(comment) else '' for comment in comments]

    # Function to get sentiment scores for a batch
    def get_sentiment_scores_batch(texts, batch_size=256, max_length=512):
        sentiments = []
        num_batches = (len(texts) + batch_size - 1) // batch_size
        for i in tqdm(range(num_batches), desc="Processing Batches"):
            batch = texts[i * batch_size:(i + 1) * batch_size]
            # Truncate texts to max_length
            truncated_batch = [text[:max_length] for text in batch]
            # Use the pipeline directly on the truncated texts
            batch_results = sentiment_analyzer(truncated_batch)
            sentiments.extend(batch_results)
        return sentiments

    # Function to process sentiment results and extract labels and scores
    def process_sentiment_results(results):
        labels = [result['label'] for result in results]
        scores = [result['score'] for result in results]
        return labels, scores

    # Prepare comments
    comments = prepare_comments(florence_reviews['comments'].tolist())
    batch_size = 256
    sentiment_results = get_sentiment_scores_batch(comments, batch_size)
    sentiment_labels, sentiment_scores = process_sentiment_results(sentiment_results)

    # Add the sentiment results to the florence_reviews DataFrame
    florence_reviews['sentiment_label'] = sentiment_labels
    florence_reviews['sentiment_score'] = sentiment_scores

    # Group by 'listing_id' and calculate the average sentiment score
    average_sentiment = florence_reviews.groupby('listing_id')['sentiment_score'].mean().reset_index()

    # Rename the column to 'average_sentiment_score'
    average_sentiment.rename(columns={'sentiment_score': 'average_sentiment_score'}, inplace=True)

    # Merge the average sentiment scores back into the original DataFrame
    florence_reviews = pd.merge(florence_reviews, average_sentiment, on='listing_id', how='left')

    # Define a function to assign star labels based on the average sentiment score
    def assign_star_label(avg_score):
        if 0.00 <= avg_score <= 0.20:
            return "1 star"
        elif 0.21 <= avg_score <= 0.40:
            return "2 stars"
        elif 0.41 <= avg_score <= 0.60:
            return "3 stars"
        elif 0.61 <= avg_score <= 0.80:
            return "4 stars"
        elif 0.81 <= avg_score <= 1.00:
            return "5 stars"
        else:
            return "Unknown"

    # Apply the function to the 'average_sentiment_score' column to create a new 'Customer_Positivity_Ranking(1to5)' column
    florence_reviews['Customer_Positivity_Ranking(1to5)'] = florence_reviews['average_sentiment_score'].apply(assign_star_label)

    # Merge the 'Customer_Positivity_Ranking(1to5)' column from florence_reviews into combined_data based on the listing_id and id
    updated_combined_data = pd.merge(combined_data,
                                     florence_reviews[['listing_id', 'Customer_Positivity_Ranking(1to5)']],
                                     left_on='id',
                                     right_on='listing_id',
                                     how='left')

    # Drop the 'listing_id' column if it's not needed in combined_data
    updated_combined_data.drop(columns=['listing_id'], inplace=True)

    # Fill missing values in 'Customer_Positivity_Ranking(1to5)' with "No reviews found"
    updated_combined_data['Customer_Positivity_Ranking(1to5)'].fillna("No reviews found", inplace=True)

    return updated_combined_data