import gdown
import zipfile
from io import BytesIO


# Define a dictionary with city names and their corresponding Google Drive file IDs
city_file_ids = {
    'barcelona': '1UjnQFcFt-E9TGJqga1ZWAUm5b_Te1F0D',
    'florence': '1IZcnHW8SITPXHQ9pbZQ2In0zBlLaoaWN',
    'lisbon': '1fbBLxVfv3CJjpHWAI0p80XLxpMqVPAMQ',
    'madrid': '1GKx_WKYYBRi-i5CaozXyzg1gx59wI52y',
    'mallorca': '1wcaxi6wWu2G7MQLf0u0WRFggVNgnAQ0y',
    'milan': '185PYRDsnd2W2Y2o3G-LiTsdFv90lr6C6',
    'rome': '1FYLxvyuQGvVyf2VijEfeRxFw3G16RAXT',
}

# Function to download and extract data for the specified city
def download_and_extract_city_data(city_name):
    # Get the file ID for the selected city
    file_id = city_file_ids.get(city_name)

    # If file_id is None, the city is not available
    if not file_id:
        raise ValueError(f"No file ID found for city: {city_name}")

    # Destination path for extraction
    destination_path = '/content'  # Extract directly to /content

    # Direct download link using gdown
    download_url = f'https://drive.google.com/uc?export=download&id={file_id}'

    # Download the zip file content directly into memory
    zip_file_content = BytesIO()
    gdown.download(download_url, zip_file_content, quiet=False)

    # Move the cursor to the beginning of the BytesIO object
    zip_file_content.seek(0)

    # Check the contents of the zip file
    with zipfile.ZipFile(zip_file_content, 'r') as zip_ref:
        zip_file_list = zip_ref.namelist()
        print(f"Files in the zip for {city_name}: {zip_file_list}")

        # Extract all files to the specified directory
        zip_ref.extractall(destination_path)

    print(f"Files for {city_name} have been extracted to: {destination_path}")