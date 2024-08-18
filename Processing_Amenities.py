import pandas as pd
from collections import Counter

def process_combined_data(combined_data):
    # Function to handle amenities column
    def handle_amenities(amenities):
        if isinstance(amenities, str):
            # Remove brackets, quotes, and split by commas
            return amenities.strip("[]").replace('"', '').split(", ")
        elif isinstance(amenities, list):
            # Directly convert the list to a space-separated string
            return amenities
        else:
            # Handle other unexpected types
            return []

    # Apply the function to the 'amenities' column
    combined_data['amenities_list'] = combined_data['amenities'].apply(handle_amenities)

    # Function to get top amenities with frequencies as percentages for each neighborhood
    def get_top_amenities_with_percentages(group, top_n=10):
        # Flatten the list of amenities for the current neighborhood
        all_amenities = [amenity for sublist in group for amenity in sublist]
        # Count the frequency of each amenity
        amenity_counts = Counter(all_amenities)
        # Calculate the total number of listings in the neighborhood
        total_listings = len(group)
        # Get the top_n amenities with their counts and calculate percentages
        top_amenities_with_percentages = [(amenity, count, (count / total_listings) * 100) for amenity, count in amenity_counts.most_common(top_n)]
        # Format the result as "Amenity (Count, Percentage%)"
        return ', '.join([f"{amenity} ({count}, {percentage:.2f}%)" for amenity, count, percentage in top_amenities_with_percentages])

    # Create a new DataFrame to store the top amenities with percentages for each neighborhood
    neighborhood_top_amenities_with_percentages = combined_data.groupby('neighbourhood_cleansed')['amenities_list'].apply(get_top_amenities_with_percentages).reset_index()

    # Rename the columns for clarity
    neighborhood_top_amenities_with_percentages.columns = ['neighbourhood_cleansed', 'top_amenities_with_percentages']

    # Merge the top amenities with percentages back into the original DataFrame
    combined_data = pd.merge(combined_data, neighborhood_top_amenities_with_percentages, on='neighbourhood_cleansed', how='left')

    return combined_data
