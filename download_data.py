import urllib.request
import zipfile
import os

# URL for SMS Spam Collection Dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
zip_path = "smsspamcollection.zip"
extract_path = "data"

def download_data():
    if not os.path.exists(extract_path):
        os.makedirs(extract_path)
    
    print(f"Downloading dataset from {url}...")
    try:
        urllib.request.urlretrieve(url, zip_path)
        print("Download complete.")
        
        print("Extracting files...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        print(f"Extracted to {extract_path}")
        
        # Cleanup zip file
        os.remove(zip_path)
        print("Cleanup complete.")
        
    except Exception as e:
        print(f"Error downloading data: {e}")

if __name__ == "__main__":
    download_data()
