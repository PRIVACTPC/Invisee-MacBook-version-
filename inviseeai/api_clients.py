import json
import os
import requests
import time
import netCDF4
import pandas as pd
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# === 1. Define base client ===
class BaseAPIClient:
    def authenticate(self, parent_tk_window=None):
        import tkinter.simpledialog as sd
        self.api_key = sd.askstring("API Key", "Enter your API key:", show='*')
        if not self.api_key:
            raise Exception("API key is required.")

    def list_datasets(self):
        return list_datasets()

    def list_versions(self, dataset_name):
        return list_versions(dataset_name)

    def list_files(self, dataset_name, version_id):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def get_download_url(self, dataset_name, version_id, filename):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def download_file(self, dataset_name, filename, version_id):
        temp_dir = os.path.join(SCRIPT_DIR, "temp_downloads")
        os.makedirs(temp_dir, exist_ok=True)

        url = self.get_download_url(dataset_name, version_id, filename)
        local_path = os.path.join(temp_dir, filename)
        download_file(url, local_path)

        if filename.endswith(".nc"):
            csv_path = local_path.replace(".nc", ".csv")
            nc_to_csv(local_path, csv_path)
            os.remove(local_path)
            return Path(csv_path)

        return Path(local_path)


# === 2. KNMI API Client ===
class KNMIClient(BaseAPIClient):
    def list_files(self, dataset_name, version_id):
        base_url = f"https://api.dataplatform.knmi.nl/open-data/v1/datasets/{dataset_name}/versions/{version_id}/files"
        headers = {"Authorization": self.api_key}
        params = {"maxKeys": 1000}
        all_files = []
        next_page_token = None

        while True:
            if next_page_token:
                params['nextPageToken'] = next_page_token
            else:
                params.pop('nextPageToken', None)

            while True:
                resp = requests.get(base_url, headers=headers, params=params)
                if resp.status_code == 429:
                    print("Rate limited (429). Retrying in 5 seconds...")
                    time.sleep(5)
                else:
                    break

            resp.raise_for_status()
            data = resp.json()
            all_files.extend(file_obj['filename'] for file_obj in data.get('files', []))

            if data.get('isTruncated'):
                next_page_token = data['nextPageToken']
            else:
                break
        return all_files

    def get_download_url(self, dataset_name, version_id, filename):
        url = f"https://api.dataplatform.knmi.nl/open-data/v1/datasets/{dataset_name}/versions/{version_id}/files/{filename}/url"
        headers = {"Authorization": self.api_key}
        while True:
            resp = requests.get(url, headers=headers)
            if resp.status_code == 429:
                print("Rate limited (429). Retrying in 5 seconds...")
                time.sleep(5)
            else:
                break
        resp.raise_for_status()
        return resp.json()["temporaryDownloadUrl"]


# === 3. Dummy Client for testing ===
class DummyAPIClient(BaseAPIClient):
    def list_files(self, dataset_name, version_id):
        return ["example1.csv", "example2.csv"]

    def get_download_url(self, dataset_name, version_id, filename):
        return f"https://dummy.api/{dataset_name}/{version_id}/{filename}"


# === 4. Helper Functions ===
def load_dataset_catalog(catalog_path="datasets.json"):
    full_path = os.path.join(SCRIPT_DIR, catalog_path)
    with open(full_path, "r", encoding="utf-8") as f:
        return json.load(f)

def list_datasets():
    return [d["name"] for d in load_dataset_catalog()]

def list_versions(dataset_name):
    catalog = load_dataset_catalog()
    for d in catalog:
        if d["name"] == dataset_name:
            return d["versions"]
    return []

def download_file(download_url, output_path):
    with requests.get(download_url, stream=True) as r:
        r.raise_for_status()
        with open(output_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

def nc_to_csv(nc_path, csv_path):
    ds = netCDF4.Dataset(nc_path)
    data = {}
    for var in ds.variables:
        try:
            data[var] = ds.variables[var][:].flatten()
        except Exception:
            continue
    if not data:
        print(f"No suitable variables found in {nc_path} to convert.")
        ds.close()
        return
    min_len = min(len(v) for v in data.values())
    for k in data:
        data[k] = data[k][:min_len]
    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False)
    ds.close()


# === 5. Optional test entry point ===
if __name__ == "__main__":
    print("This module is meant to be imported, not run directly.")