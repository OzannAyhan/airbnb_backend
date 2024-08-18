def prepare_combined_data(combined_calender, combined_listings_extended):
    # Rename columns
    combined_calender = combined_calender.rename(columns={'listing_id': 'id'})
    combined_listings_extended = combined_listings_extended.rename(columns={'price': 'prices', 'ratings': 'review_scores_rating'})

    # Ensure 'id' is an integer
    combined_calender['id'] = combined_calender['id'].astype(int)
    combined_listings_extended['id'] = combined_listings_extended['id'].astype(int)

    # Perform the merge
    combined_data = pd.merge(combined_calender, combined_listings_extended, on='id', how='left', suffixes=('', '_listings'))

    # Drop rows where 'prices' column is empty (NaN or empty string)
    combined_data = combined_data.dropna(subset=['prices'])

    # Drop unnecessary columns
    combined_data = combined_data.drop(columns=['prices'])

    # Ensure 'price' column is treated as string
    combined_data['price'] = combined_data['price'].astype(str)

    # Remove dollar signs and commas using str.replace
    combined_data['price'] = combined_data['price'].str.replace('$', '', regex=False)  # First remove dollar signs
    combined_data['price'] = combined_data['price'].str.replace(',', '', regex=False)  # Then remove commas

    # Convert to float
    combined_data['price'] = combined_data['price'].astype(float)

    # Convert 'reviews_per_month' to numeric, coercing errors to NaN
    combined_data['reviews_per_month'] = pd.to_numeric(combined_data['reviews_per_month'], errors='coerce')

    # Fill NaN values in 'reviews_per_month' with the mean
    combined_data['reviews_per_month'] = combined_data['reviews_per_month'].fillna(combined_data['reviews_per_month'].mean())

    # Convert 'date' column to datetime
    combined_data['date'] = pd.to_datetime(combined_data['date'], errors='coerce')

    # Drop rows where 'date' could not be parsed
    combined_data = combined_data.dropna(subset=['date'])

    # Filter based on the date condition
    combined_data = combined_data[(combined_data['date'] >= '2023-09-01') & (combined_data['date'] < '2024-06-30')]

    return combined_data