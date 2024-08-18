import pandas as pd

def process_city_calender(city_name):
    base_path = "/content/calendar_{}{}.csv"
    listings = []

    # Read and log the number of rows for each file
    for i in range(1, 5):
        file_path = base_path.format(city_name, i)
        df = pd.read_csv(file_path, dtype=str, low_memory=False)
        print(f"Rows in calendar_{city_name}{i}.csv: {df.shape[0]}")
        listings.append(df)

    # Concatenate all DataFrames into a single DataFrame
    combined_calender = pd.concat(listings, ignore_index=True)
    print(f"Initial combined rows: {combined_calender.shape[0]}")

    # Filter the combined DataFrame
    combined_calender = combined_calender[combined_calender['available'] != 'f']
    print(f"Rows after filtering 'available' column: {combined_calender.shape[0]}")

    # Drop the specified columns
    columns_to_drop = ['adjusted_price', 'minimum_nights', 'maximum_nights']
    combined_calender = combined_calender.drop(columns=columns_to_drop, errors='ignore')

    # Convert 'date' column to datetime, handling errors by coercing invalid dates to NaT
    combined_calender['date'] = pd.to_datetime(combined_calender['date'], errors='coerce')

    # Log the number of invalid dates
    invalid_dates_count = combined_calender['date'].isna().sum()
    print(f"Number of rows with invalid dates: {invalid_dates_count}")

    # Drop rows with NaT in 'date' column
    combined_calender = combined_calender.dropna(subset=['date'])
    print(f"Rows after dropping invalid dates: {combined_calender.shape[0]}")

    # Update 'date' column to have the first day of the month
    combined_calender['date'] = combined_calender['date'].apply(lambda x: x.replace(day=1))

    # Sort the DataFrame by 'listing_id' and 'date'
    combined_calender = combined_calender.sort_values(by=['listing_id', 'date'])

    # Drop duplicates based on 'listing_id' and 'date' columns
    combined_calender = combined_calender.drop_duplicates(subset=['listing_id', 'date'])
    print(f"Rows after dropping duplicates: {combined_calender.shape[0]}")

    # Ensure 'listing_id' is read as an integer
    combined_calender['listing_id'] = combined_calender['listing_id'].apply(lambda x: int(float(x)))


    return combined_calender