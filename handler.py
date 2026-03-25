"""
RunPod Serverless handler for LTX-2.3 video generation via Wan2GP.

Supports:
  - text-to-video:  { "prompt": "...", ... }
  - image-to-video: { "prompt": "...", "image": "<base64>", ... }

Returns: { "url": "https://pub-xxx.r2.dev/videos/<id>.mp4" }
"""

import os
import sys
import uuid
import base64
import time
from pathlib import Path

import boto3
import runpod

# ---------------------------------------------------------------------------
# R2 configuration (set via RunPod env vars)
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Wan2GP paths (match the base Docker image layout)
# ---------------------------------------------------------------------------
WANGP_ROOT = Path("/workspace/Wan2GP")
OUTPUT_DIR = Path(os.environ.get("W2GP_OUTPUTS", "/workspace/wan2gp/outputs"))
TMP_DIR = Path("/tmp/ltx-inputs")
TMP_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Wan2GP session — loaded once, kept warm across requests
# ---------------------------------------------------------------------------
session = None


def load_model():
    """Initialize the Wan2GP session with LTX-2.3 Distilled + flash attention."""
    global session

    if session is not None:
        return

    # Wan2GP root must be on sys.path for shared.api import
    if str(WANGP_ROOT) not in sys.path:
        sys.path.insert(0, str(WANGP_ROOT))

    from shared.api import init

    profile = os.environ.get("W2GP_PROFILE", "4")

    print(f"[handler] Initializing Wan2GP session (profile={profile}, attention=flash)...")
    t0 = time.time()

    session = init(
        root=WANGP_ROOT,
        output_dir=OUTPUT_DIR,
        cli_args=["--attention", "flash", "--profile", profile],
        console_output=True,
    )

    print(f"[handler] Wan2GP session ready in {time.time() - t0:.1f}s")


def upload_to_r2(local_path: str, key: str) -> str:
    """Upload a file to Cloudflare R2 and return the public URL."""
    s3.upload_file(
        local_path,
        R2_BUCKET,
        key,
        ExtraArgs={"ContentType": "video/mp4"},
    )
    return f"{R2_PUBLIC_URL.rstrip('/')}/{key}"


def handler(job):
    """
    RunPod serverless handler.

    Input schema:
        prompt (str, required):       Text prompt for generation.
        image (str, optional):        Base64-encoded image for image-to-video.
        width (int, default 1280):    Output width.
        height (int, default 832):    Output height (832x1280 = 9:16 portrait).
        num_frames (int, default 121): Number of frames (121 = ~5s at 24fps).
        steps (int, default 8):       Inference steps (distilled model uses 8).
        seed (int, optional):         Random seed for reproducibility.

    Returns:
        { "url": "<public R2 URL>", "job_id": "<uuid>" }
    """
    job_input = job["input"]
    job_id = job.get("id", str(uuid.uuid4()))

    # --- Validate required fields ---
    prompt = job_input.get("prompt")
    if not prompt:
        return {"error": "prompt is required"}

    # --- Optional parameters ---
    width = job_input.get("width", 1280)
    height = job_input.get("height", 832)
    num_frames = job_input.get("num_frames", 121)
    steps = job_input.get("steps", 8)
    seed = job_input.get("seed")
    image_b64 = job_input.get("image")

    # --- Build Wan2GP settings dict ---
    settings = {
        "model_type": "ltx2_22B_distilled",
        "prompt": prompt,
        "resolution": f"{width}x{height}",
        "num_inference_steps": steps,
        "video_length": num_frames,
        "force_fps": "24",
    }

    if seed is not None:
        settings["seed"] = int(seed)

    # --- Image-to-video: save base64 image to temp file ---
    input_image_path = None
    if image_b64:
        try:
            input_image_path = str(TMP_DIR / f"{job_id}_input.png")
            image_data = base64.b64decode(image_b64)
            with open(input_image_path, "wb") as f:
                f.write(image_data)
            settings["start_image"] = input_image_path
        except Exception as e:
            return {"error": f"Failed to decode base64 image: {e}"}

    # --- Run generation ---
    try:
        print(f"[handler] Job {job_id}: submitting {'image' if image_b64 else 'text'}-to-video")
        t0 = time.time()

        gen_job = session.submit_task(settings)

        # Stream progress events (logged to RunPod)
        for event in gen_job.events.iter(timeout=1.0):
            if event.kind == "progress":
                p = event.data
                print(f"[handler] {p.phase} {p.progress}% step={p.current_step}/{p.total_steps}")
            elif event.kind == "error":
                print(f"[handler] ERROR: {event.data}")

        result = gen_job.result()
        elapsed = time.time() - t0

        if not result.success:
            error_msgs = [str(e) for e in result.errors]
            print(f"[handler] Job {job_id} FAILED: {error_msgs}")
            return {"error": "; ".join(error_msgs), "job_id": job_id}

        if not result.generated_files:
            return {"error": "Generation succeeded but no output files found", "job_id": job_id}

        # --- Upload first output to R2 ---
        output_path = result.generated_files[0]
        r2_key = f"videos/{job_id}.mp4"
        public_url = upload_to_r2(output_path, r2_key)

        print(f"[handler] Job {job_id} done in {elapsed:.1f}s → {public_url}")

        # Cleanup local output
        try:
            os.remove(output_path)
        except OSError:
            pass

        return {"url": public_url, "job_id": job_id}

    except Exception as e:
        print(f"[handler] Job {job_id} exception: {e}")
        return {"error": str(e), "job_id": job_id}

    finally:
        # Cleanup temp input image
        if input_image_path and os.path.exists(input_image_path):
            try:
                os.remove(input_image_path)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Startup: load model once, then accept RunPod jobs
# ---------------------------------------------------------------------------
print("[handler] Loading model at worker startup...")
load_model()
print("[handler] Starting RunPod serverless handler...")
runpod.serverless.start({"handler": handler})
