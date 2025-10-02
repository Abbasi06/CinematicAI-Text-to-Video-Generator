import torch
from diffusers import AutoPipelineForText2Image
from PIL import Image
import os
from dotenv import load_dotenv
load_dotenv()

HUGGING_FACE_TOKEN = os.getenv("HUGGING_FACE_TOKEN")


# ----- SDXL Turbo Quantized Model -----
# Model ID for SDXL-Turbo
model_id = "stabilityai/sdxl-turbo"

# Load the pipeline with 4-bit quantization enabled
pipeline_turbo_quantized = AutoPipelineForText2Image.from_pretrained(
    model_id,
    
    torch_dtype=torch.float16,
    variant="fp16",
    load_in_4bit=True  # This is the key parameter for quantization
).to("cuda")

# prompt = "A majestic robotic eagle soaring through a cyberpunk city at night, rain-slicked streets below, neon signs reflecting on its metallic feathers, cinematic lighting."

prompt = "A majestic dragon soars over a medieval stone castle at dusk. The dragon’s scales glint in the fading light as it circles above turrets and battlements. Mist drifts around towers. The camera slowly pans and pulls back to show the full castle under dramatic skies, with torches flickering on the walls. Fantasy, cinematic style, high detail."

# Generate the image - it only needs one step!
image = pipeline_turbo_quantized(
    prompt=prompt,
    num_inference_steps=1,
    guidance_scale=0.0
).images[0]

image.save("quantized_sdxl_turbo_output.png")
print("Image from quantized SDXL-Turbo saved!")
del pipeline_turbo_quantized
torch.cuda.empty_cache()


## Image to Vide Generation
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_video

print("\n--- Part 2: Generating Videos with Stable Video Diffusion ---")

# --- Step 2A: Load the SVD Pipeline ---
# This model is specifically for image-to-video generation
svd_pipeline = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid-xt",
    torch_dtype=torch.float16,
    variant="fp16"
).to("cuda")

# --- Step 2C: Process the SDXL-TURBO Image into a Video ---
print("\nProcessing the quantized SDXL-Turbo seed image into video...")
seed_image_turbo = load_image("quantized_sdxl_turbo_output.png")

# Resize the Turbo image as well
seed_image_turbo = seed_image_turbo.resize((1024, 576))

# Generate video frames from the second image
video_frames_turbo = svd_pipeline(
    seed_image_turbo,
    num_frames=25,
    decode_chunk_size=8,
    motion_bucket_id=127,
    fps=7,
    noise_aug_strength=0.02
).frames[0]

# Export the frames to a different video file
export_to_video(video_frames_turbo, "sdxl_turbo_output_video.mp4")
print("Video generated from SDXL-Turbo image saved as sdxl_turbo_output_video.mp4")

print("\nAll tasks complete!")