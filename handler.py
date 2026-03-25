import os
import uuid
import base64
import boto3
import runpod
from io import BytesIO
from PIL import Image

# --- R2 Configuration ---
R2_ACCOUNT_ID = os.environ["R2_ACCOUNT_ID"]
R2_ACCESS_KEY_ID = os.environ["R2_ACCESS_KEY_ID"]
R2_SECRET_ACCESS_KEY = os.environ["R2_SECRET_ACCESS_KEY"]
R2_BUCKET = os.environ.get("R2_BUCKET", "ai-assets")
R2_PUBLIC_URL = os.environ["R2_PUBLIC_URL"]

s3 = boto3.client(
    "s3",
    endpoint_url=f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com",
    aws_access_key_id=R2_ACCESS_KEY_ID,
    aws_secret_access_key=R2_SECRET_ACCESS_KEY,
    region_name="auto",
)

TMP = "/tmp/ltx"
os.makedirs(TMP, exist_ok=True)

# --- Model (loaded once at worker startup) ---
ltx = None


def load_model():
    global ltx
    if ltx is not None:
        return

    import torch
    from ltx_pipelines import DistilledPipeline
    from ltx_pipelines.config import QuantizationPolicy

    print("Loading LTX-2.3 Distilled (FP8)...")
    ltx = DistilledPipeline.from_pretrained(
        "Lightricks/LTX-2.3-fp8",
        quantization=QuantizationPolicy.fp8_scaled_mm(),
    )
    ltx.skip_memory_cleanup = True
    print("LTX-2.3 Distilled ready.")


def upload_to_r2(local_path: str, key: str) -> str:
    s3.upload_file(
        local_path,
        R2_BUCKET,
        key,
        ExtraArgs={"ContentType": "video/mp4"},
    )
    return f"{R2_PUBLIC_URL.rstrip('/')}/{key}"


def handler(job):
    job_input = job["input"]
    job_id = job.get("id", str(uuid.uuid4()))

    # Required
    prompt = job_input.get("prompt")
    if not prompt:
        return {"error": "prompt is required"}

    # Optional params
    mode = job_input.get("mode", "text-to-video")
    height = job_input.get("height", 720)
    width = job_input.get("width", 1280)
    num_frames = job_input.get("num_frames", 121)

    out_path = f"{TMP}/{job_id}.mp4"

    try:
        if mode == "image-to-video":
            # Decode base64 image
            image_b64 = job_input.get("image")
            if not image_b64:
                return {"error": "image (base64) is required for image-to-video"}

            img_path = f"{TMP}/{job_id}_input.jpg"
            img = Image.open(BytesIO(base64.b64decode(image_b64))).convert("RGB")
            img.save(img_path, "JPEG")

            ltx.generate(
                prompt=prompt,
                image_path=img_path,
                height=height,
                width=width,
                num_frames=num_frames,
                output_path=out_path,
            )

            os.remove(img_path)

        else:
            # text-to-video (default)
            ltx.generate(
                prompt=prompt,
                height=height,
                width=width,
                num_frames=num_frames,
                output_path=out_path,
            )

        # Upload to R2
        r2_key = f"videos/{job_id}.mp4"
        public_url = upload_to_r2(out_path, r2_key)
        os.remove(out_path)

        return {"url": public_url, "job_id": job_id}

    except Exception as e:
        # Cleanup on failure
        for f in [out_path, f"{TMP}/{job_id}_input.jpg"]:
            if os.path.exists(f):
                os.remove(f)
        return {"error": str(e)}


# Load model at worker startup (before accepting jobs)
load_model()

runpod.serverless.start({"handler": handler})
