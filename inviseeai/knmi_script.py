import os
import sys
import requests
from dotenv import load_dotenv

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from api_client_logic import get_client

load_dotenv()

def prompt_user_selection(options, label):
    print(f"\n{label}:")
    for i, item in enumerate(options, 1):
        print(f"{i}. {item}")
    while True:
        try:
            choice = int(input("Select number: "))
            if 1 <= choice <= len(options):
                return options[choice - 1]
            else:
                print("Invalid number.")
        except ValueError:
            print("Please enter a valid number.")

def main():
    api_key = os.getenv("KNMI_API_KEY")
    if not api_key:
        print("Missing API key. Please set KNMI_API_KEY in your .env file.")
        return

    client = get_client("KNMI")
    client.api_key = api_key

    try:
        # Step 1: Use hardcoded list of datasets
        datasets = [
            "Actuele10mindataKNMIstations",
            "radar_rainfall",
            "synopdata",
            "daggegevens"
        ]
        dataset = prompt_user_selection(datasets, "Available Datasets")

        # Step 2: Use hardcoded versions for selected dataset
        dataset_versions = {
            "Actuele10mindataKNMIstations": ["2"],
            "radar_rainfall": ["1"],
            "synopdata": ["1"],
            "daggegevens": ["1"]
        }
        versions = dataset_versions.get(dataset, [])
        version = prompt_user_selection(versions, f"Versions for '{dataset}'")

        # Step 3: List files
        files = client.list_files(dataset, version)
        if not files:
            print("No files found.")
            return
        filename = prompt_user_selection(files, f"Files in '{dataset}' version '{version}'")

        # Step 4: Confirm and download
        confirm = input(f"\nDownload '{filename}'? (y/n): ").lower()
        if confirm == 'y':
            path = client.download_file(dataset, filename, version)
            print("File saved to:")
            print(path)
        else:
            print("Download cancelled.")

    except Exception as e:
        print("Error:", str(e))

if __name__ == "__main__":
    main()
