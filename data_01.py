import requests
import zipfile
from pathlib import Path

data_path = Path("data/")
image_path = data_path / "pizza_steak_sushi"
if image_path.is_dir():
    print(f"{image_path} directory exists.")
else:
    print(f"Did not find {image_path} directory, creating one...")
    image_path.mkdir(parents=True, exist_ok=True)
        with open(data_path / "pidata.zip", "wb") as f:
        request = requests.get("https//mydata.zip")
        print("Downloading data...")
        f.write(request.content)

    with zipfile.ZipFile(data_path / "pidata.zip", "r") as zip_ref:
        print("Unzipping data...") 
        zip_ref.extractall(image_path)
