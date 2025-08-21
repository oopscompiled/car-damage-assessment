import os
import zipfile
import gdown

def download():
    FILE_ID = "1y4ONLZLbC8musrt5gxSzbGx22vkeEUgz"
    DATA_URL = f"https://drive.google.com/uc?id={FILE_ID}"
    DATA_DIR = 'data'

    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    else:
        print('Data folder already exists')

    zip_path = os.path.join(DATA_DIR, 'dataset.zip')

    if not os.path.exists(zip_path):
        print("Downloading data...")
        gdown.download(DATA_URL, zip_path, quiet=False)

    print("Unzipping the data...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(DATA_DIR)

    os.remove(zip_path)
    print("Data has been downloaded.")

if __name__ == "__main__":
    print("Running data_loader")
    download()
