import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import warnings


def arima_forecast_and_save(city_name, combined_data, output_dir='/content'):

    def grid_search_arima(train_data, p_values, d_values, q_values):
        best_aic = float("inf")
        best_order = None
        best_model = None

        for p in p_values:
            for d in d_values:
                for q in q_values:
                    try:
                        model = ARIMA(train_data, order=(p, d, q))
                        model_fit = model.fit()
                        aic = model_fit.aic
                        if aic < best_aic:
                            best_aic = aic
                            best_order = (p, d, q)
                            best_model = model_fit
                    except Exception as e:
                        continue

        return best_model, best_order

    # Prepare city data
    city_data = combined_data.copy()
    city_data["date"] = pd.to_datetime(city_data["date"])
    city_data['price'] = pd.to_numeric(city_data['price'], downcast='float')

    unique_neighbourhoods = city_data['neighbourhood_cleansed'].dropna().unique()

    results = []

    for neighbourhood in unique_neighbourhoods:
        neighbourhood_data = city_data[city_data['neighbourhood_cleansed'] == neighbourhood]
        avg_prices = neighbourhood_data.groupby('date')['price'].mean()
        avg_prices.index = pd.to_datetime(avg_prices.index)
        avg_prices = avg_prices.asfreq('MS').dropna()

        if len(avg_prices) < 2:
            print(f"Skipping {neighbourhood} in {city_name} due to insufficient data after removing NaN values.")
            continue

        train_part = avg_prices['2023-09-01':'2024-06-30']
        if len(train_part) < 6:
            print(f"Skipping {neighbourhood} in {city_name} due to insufficient train data.")
            continue

        p_values = range(0, 4)
        d_values = range(0, 2)
        q_values = range(0, 4)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            best_model, best_order = grid_search_arima(train_part, p_values, d_values, q_values)

        if best_model is None:
            fallback_model = ARIMA(train_part, order=(0, 1, 0))
            fallback_model_fit = fallback_model.fit()
            forecasted_values = fallback_model_fit.forecast(steps=2)
        else:
            forecasted_values = best_model.forecast(steps=2)

        forecasted_values_df = pd.DataFrame({
            'neighbourhood_cleansed': [neighbourhood] * 2,
            'date': pd.date_range(start='2024-07-01', periods=2, freq='MS'),
            'price': forecasted_values
        })

        results.append(forecasted_values_df)

    combined_results_df = pd.concat(results)

    # Ensure 'date' columns are in the same format
    combined_results_df['date'] = pd.to_datetime(combined_results_df['date'])
    city_data['date'] = pd.to_datetime(city_data['date'])

    # Concatenate the forecasted data with the original city data
    final_combined_df = pd.concat([city_data, combined_results_df], ignore_index=True)

    # Sorting the final combined dataframe by neighbourhood and date
    final_combined_df = final_combined_df.sort_values(by=['neighbourhood_cleansed', 'date'])

    # Save the updated combined_data back to the CSV file
    output_file_path = f'{output_dir}/{city_name}_final_data.csv'
    final_combined_df.to_csv(output_file_path, index=False)
    print(f"Data saved to {output_file_path}")

    # Display DataFrame information
    final_combined_df.info()

    return final_combined_df