def classify_property_descriptions(combined_data, batch_size=32):
    """
    Classifies property descriptions into categories (Luxury, Standard, Economy) using a zero-shot classification model.

    Args:
    - combined_data: The DataFrame containing property descriptions.
    - batch_size: The size of batches for processing descriptions.

    Returns:
    - combined_data: The updated DataFrame with an added 'category' column.
    """
    # Create a new DataFrame for processing
    processed_data = combined_data.copy()

    # Remove duplicate descriptions in the processed DataFrame
    processed_data = processed_data.drop_duplicates(subset=['id', 'description'])

    # Initialize the zero-shot classification pipeline with GPU if available
    device = 0 if torch.cuda.is_available() else -1
    classifier = pipeline("zero-shot-classification", model="joeddav/xlm-roberta-large-xnli", device=device)

    # Define the candidate labels
    candidate_labels = ["Luxury", "Standard", "Economy"]

    # Function to classify descriptions in batches
    def classify_descriptions_batch(descriptions, batch_size=batch_size):
        results = []
        for i in tqdm(range(0, len(descriptions), batch_size), desc="Processing Batches"):
            batch = descriptions[i:i+batch_size]
            batch_results = classifier(batch, candidate_labels=candidate_labels)
            results.extend([result['labels'][0] for result in batch_results])
        return results

    # Apply the classification to the processed dataset
    processed_data['category'] = classify_descriptions_batch(processed_data['description'].tolist())

    # Keep only 'id', 'description', and 'category' columns in processed_data
    processed_data = processed_data[['id', 'description', 'category']]

    # Merge the 'category' column from processed_data into combined_data based on both 'id' and 'description'
    combined_data = pd.merge(combined_data, processed_data[['id', 'description', 'category']], on=['id', 'description'], how='left')

    return combined_data
