def process_city_listings(city_name):
    def load_listings(city):
        base_path = "/content/{}_listings{}.csv"
        listings = []
        for i in range(1, 5):
            file_path = base_path.format(city, i)
            listings.append(pd.read_csv(file_path, dtype=str, low_memory=False))
        combined_listings = pd.concat(listings, ignore_index=True)
        return combined_listings

    def load_listings_long(city):
        base_path = "/content/{}_listings{}_long.csv"
        listings_long = []
        for i in range(1, 5):
            file_path = base_path.format(city, i)
            listings_long.append(pd.read_csv(file_path, dtype=str, low_memory=False))
        combined_listings_long = pd.concat(listings_long, ignore_index=True)
        return combined_listings_long

    # Load the data
    combined_listings = load_listings(city_name)
    combined_listings_long = load_listings_long(city_name)

    # Drop duplicates based on 'id' and 'host_id' columns
    combined_listings = combined_listings.drop_duplicates(subset=['id', 'host_id'])
    combined_listings_long = combined_listings_long.drop_duplicates(subset=['id', 'host_id'])

    # List of columns to merge from combined_listings_long
    columns_to_merge = ['description', 'host_total_listings_count', 'neighbourhood_cleansed', 'amenities', 'review_scores_rating']

    # Merge data frames based on 'id' column
    combined_listings_extended = combined_listings.merge(combined_listings_long[['id'] + columns_to_merge], on='id', how='left')

    # Drop unnecessary columns
    combined_listings_extended = combined_listings_extended.drop(columns=['license', 'last_review', 'latitude', 'longitude', 'room_type', 'neighbourhood_group', 'neighbourhood', 'minimum_nights'], errors='ignore')

    # Compute the mean of the 'review_scores_rating' column
    mean_review_score = combined_listings_extended['review_scores_rating'].astype(float).mean()

    # Impute the NaN values with the mean
    combined_listings_extended['review_scores_rating'] = combined_listings_extended['review_scores_rating'].astype(float).fillna(mean_review_score)

    # Fill NaN values in 'description' with values from 'name'
    combined_listings_extended['description'].fillna(combined_listings_extended['name'], inplace=True)

    # Ensure the data type of id
    combined_listings_extended['id'] = combined_listings_extended['id'].astype(int)

    return combined_listings_extended