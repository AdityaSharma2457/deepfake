
import os
import base64
from pathlib import Path

import requests

ACCOUNT_ID = os.environ.get("CLOUDFLARE_ACCOUNT_ID")
API_TOKEN = os.environ.get("CLOUDFLARE_API_TOKEN")

MODEL = "@cf/stabilityai/stable-diffusion-xl-base-1.0"

if not ACCOUNT_ID or ACCOUNT_ID == "YOUR_CLOUDFLARE_ACCOUNT_ID":
    raise SystemExit("Set CLOUDFLARE_ACCOUNT_ID to your real Cloudflare account ID.")
if not API_TOKEN or API_TOKEN == "YOUR_CLOUDFLARE_API_TOKEN":
    raise SystemExit("Set CLOUDFLARE_API_TOKEN to a Workers AI API token.")

url = (
    
    f"https://api.cloudflare.com/client/v4/accounts/"
    f"{ACCOUNT_ID}/ai/run/{MODEL}"
)

headers = {
    "Authorization": f"Bearer {API_TOKEN}",
    "Content-Type": "application/json"
}

payload = {
    "prompt": "A photorealistic orange cat sitting on a wooden table, "
              "cinematic lighting, highly detailed, 4k",
    "negative_prompt": "blurry, low quality, distorted",
    "width": 1024,
    "height": 1024,
    "num_steps": 20,
    "guidance": 7.5,
    "seed": 42
}

try:
    response = requests.post(url, headers=headers, json=payload, timeout=180)
except requests.RequestException as exc:
    raise SystemExit(f"Cloudflare request failed: {exc}") from exc

if not response.ok:
    raise SystemExit(f"Cloudflare returned HTTP {response.status_code}: {response.text}")

content_type = response.headers.get("Content-Type", "")
if "image" in content_type:
    image_bytes = response.content
else:
    result = response.json()
    if not result.get("success"):
        raise SystemExit(f"Cloudflare model error: {result}")
    image_data = result.get("result", {}).get("image")
    if not image_data:
        raise SystemExit(f"Cloudflare returned no image: {result}")
    image_bytes = base64.b64decode(image_data)

output_path = Path("generated.png")
output_path.write_bytes(image_bytes)
print(f"Generation complete: {output_path.resolve()}")
