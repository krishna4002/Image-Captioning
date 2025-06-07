"""
import os
import zipfile
import requests
import subprocess

def download_file(url, save_path):
    print(f"Downloading: {url}")
    r = requests.get(url, stream=True)
    r.raise_for_status()
    with open(save_path, 'wb') as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)

def unzip_file(zip_path, extract_dir):
    print(f"Extracting: {zip_path} to {extract_dir}")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)


def download_coco():
    data_dir = "data/coco"
    os.makedirs(data_dir, exist_ok=True)

    img_url = 'http://images.cocodataset.org/zips/val2017.zip'
    ann_url = 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'

    img_zip = os.path.join(data_dir, "val2017.zip")
    ann_zip = os.path.join(data_dir, "annotations.zip")

    img_folder = os.path.join(data_dir, "val2017")
    ann_folder = os.path.join(data_dir, "annotations")

    if not os.path.exists(img_folder):
        download_file(img_url, img_zip)
        unzip_file(img_zip, data_dir)
    else:
        print("COCO val2017 images already downloaded.")

    if not os.path.exists(ann_folder):
        download_file(ann_url, ann_zip)
        unzip_file(ann_zip, data_dir)
    else:
        print("COCO annotations already downloaded.")

if __name__ == "__main__":
    download_coco()
    print("All datasets downloaded and extracted!")
"""
import os
import requests

def download_file(url, destination):
    response = requests.get(url)
    with open(destination, 'wb') as f:
        f.write(response.content)

def download_iu_xray():
    data_dir = "data/iu_xray"
    os.makedirs(data_dir, exist_ok=True)
    json_path = os.path.join(data_dir, "iu_xray.json")
    if not os.path.exists(json_path):
        print("⬇ Downloading missing IU X-Ray JSON captions...")
        json_url = "https://huggingface.co/datasets/Jyothirmai/iu-xray-dataset/resolve/main/iu_xray.json"
        download_file(json_url, json_path)
        print("✅ Downloaded iu_xray.json.")

if __name__ == "__main__":
    download_iu_xray()
    print("All datasets downloaded and extracted!")