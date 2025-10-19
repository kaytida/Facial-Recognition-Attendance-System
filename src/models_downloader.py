import os
import requests
from tqdm import tqdm
import bz2

def download_file(url, output_path):
    """Download file from a URL with a progress bar."""
    if os.path.exists(output_path):
        print(f"[INFO] {output_path} already exists. Skipping download.")
        return

    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 KB
    t = tqdm(total=total_size, unit='iB', unit_scale=True, desc=os.path.basename(output_path))

    with open(output_path, 'wb') as f:
        for data in response.iter_content(block_size):
            t.update(len(data))
            f.write(data)
    t.close()

    if total_size != 0 and t.n != total_size:
        print("[ERROR] Something went wrong during the download.")

def extract_bz2(bz2_path, output_path):
    """Extract a .bz2 compressed file."""
    if os.path.exists(output_path):
        print(f"[INFO] {output_path} already exists. Skipping extraction.")
        return
    print(f"[INFO] Extracting {bz2_path} ...")
    with bz2.BZ2File(bz2_path) as fr, open(output_path, 'wb') as fw:
        fw.write(fr.read())
    print(f"[INFO] Extracted to {output_path}")

def download_dlib_models(models_dir="models"):
    os.makedirs(models_dir, exist_ok=True)

    # URLs for dlib models
    files = {
        "shape_predictor_68_face_landmarks.dat.bz2":
            "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2",
        "dlib_face_recognition_resnet_model_v1.dat.bz2":
            "http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2"
    }

    for bz2_name, url in files.items():
        bz2_path = os.path.join(models_dir, bz2_name)
        dat_path = bz2_path[:-4]  # remove .bz2 extension

        download_file(url, bz2_path)
        extract_bz2(bz2_path, dat_path)

if __name__ == "__main__":
    download_dlib_models()
