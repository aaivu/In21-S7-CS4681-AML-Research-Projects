import requests

url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
output_file = "sam_vit_h_4b8939.pth"

with requests.get(url, stream=True) as r:
    r.raise_for_status()
    with open(output_file, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)

print("Download completed!")
