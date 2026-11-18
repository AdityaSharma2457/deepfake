import os
from pathlib import Path

from huggingface_hub import InferenceClient
from huggingface_hub.errors import HfHubHTTPError


token = os.environ.get("HF_TOKEN")
if not token:
    raise SystemExit(
        "HF_TOKEN is not set. Set it in your environment before running this script."
    )

client = InferenceClient(provider="fal-ai", token=token)

try:
    image = client.text_to_image(
        "Astronaut riding a horse",
        model="stabilityai/stable-diffusion-xl-base-1.0",
    )
except HfHubHTTPError as exc:
    if getattr(exc.response, "status_code", None) == 402:
        raise SystemExit(
            "Hugging Face returned 402 Payment Required: this routed fal.ai request "
            "has no available inference credits. Add credits / billing in Hugging Face, "
            "use your own fal.ai provider key, or run a diffusion model locally."
        ) from None
    raise

output_path = Path("generated_image.png")
image.save(output_path)
print(f"Saved image to {output_path.resolve()}")
