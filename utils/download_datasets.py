import os
import requests
import zipfile
import gzip
import pickle
from tqdm import tqdm

# Datasets to download
DATASETS = [
    {
        "name": "Structured.zip",
        "url": "http://pages.cs.wisc.edu/~anhai/data1/deepmatcher_data/Structured.zip",
    },
    {
        "name": "Textual.zip",
        "url": "http://pages.cs.wisc.edu/~anhai/data1/deepmatcher_data/Textual.zip",
    },
    {
        "name": "Dirty.zip",
        "url": "http://pages.cs.wisc.edu/~anhai/data1/deepmatcher_data/Dirty.zip",
    },
]


def download_data():
    # Download each dataset
    for dataset in DATASETS:
        dataset_name = dataset["name"]
        dataset_url = dataset["url"]
        save_path = os.path.join(repo_dir, dataset_name)

        print(f"\nDownloading {dataset_name}...")

        # Show progress bar
        response = requests.get(dataset_url, stream=True)
        total_size_in_bytes = int(response.headers.get("content-length", 0))
        block_size = 1024  # 1KB
        progress_bar = tqdm(
            total=total_size_in_bytes, unit="B", unit_scale=True, desc=dataset_name
        )

        # Download the dataset file with progress bar
        with open(save_path, "wb") as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()

        # Extract the contents
        print(f"Unzipping {dataset_name}...")
        with zipfile.ZipFile(save_path, "r") as zip_ref:
            zip_ref.extractall(repo_dir)
        os.remove(save_path)

        print(f"Download complete: {dataset_name}")


#


def download_from_github(file_name, url, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    file_path = os.path.join(save_path, file_name)
    if not os.path.exists(file_path):
        print(f"Downloading {file_name} from {url}")
        response = requests.get(url)
        if response.status_code == 200:
            with open(file_path, "wb") as f:
                f.write(response.content)
            print(f"File {file_name} downloaded successfully.")
        else:
            print(
                f"Failed to download {file_name}. Status code: {response.status_code}"
            )
    else:
        print(f"File {file_name} already exists")
    return file_path


if __name__ == "__main__":
    # Directory for storing the datasets
    repo_dir = "input"

    # Create the repository directory if it doesn't exist
    os.makedirs(repo_dir, exist_ok=True)

    # Download data
    download_data()

    print("\nAll Magellan datasets have been downloaded and extracted.")

    print("Downloading the Mannheim datasets...\n")

    # List of dataset folders to iterate through
    dataset_folders = [
        "abt-buy",
        "amazon-google",
        "dblp-acm",
        "dblp-scholar",
        "walmart-amazon",
        "wdc",
    ]

    # Base GitHub URL for the datasets
    base_url = "https://github.com/wbsg-uni-mannheim/MatchGPT/raw/main/LLMForEM/data"

    # Folder to save the downloaded data
    save_path = "input/manheim_data"

    # Iterate over each dataset folder and download the corresponding sampled dataset
    for folder in dataset_folders:
        file_name = f"{folder}-sampled_gs.pkl.gz"
        url = f"{base_url}/{folder}/{file_name}"

        # Download the file
        file_path = download_from_github(file_name, url, save_path)

        # Load the data
        try:
            with gzip.open(file_path, "rb") as f:
                data = pickle.load(f)

            # Display the data (replace this with any other processing you'd like)
            print(f"Data from {file_name}:")
            print(data[["id_left", "id_right"]])
        except Exception as e:
            print(f"Error loading the file {file_name}: {e}")

    print("\nAll Manheim datasets have been downloaded and extracted.")
