
# # --- Flux.1 Model ---
import torch
from diffusers import FluxPipeline
from PIL import Image
import os
from dotenv import load_dotenv
load_dotenv()

HUGGING_FACE_TOKEN = os.getenv("HUGGING_FACE_TOKEN")


# # --- Step 1B: Generate Image with FLUX.1 ---
# print("\nLoading FLUX.1 Pipeline...")
# # FLUX has a specific pipeline class
# flux_pipeline = FluxPipeline.from_pretrained(
#     "black-forest-labs/FLUX.1-dev",
#     torch_dtype=torch.bfloat16, token = HUGGING_FACE_TOKEN
# ).to("cuda")

# print("Generating image with FLUX.1...")
# # Generate the image using the same prompt for a fair comparison
# flux_image = flux_pipeline(prompt=prompt).images[0]

# # Save the image
# flux_image.save("flux_seed_image.png")
# print("FLUX.1 seed image saved as flux_seed_image.png")

# # del flux_pipeline
# torch.cuda.empty_cache()


# --- Flux.1 Quantized Model ---
model_id_flux_s = "black-forest-labs/FLUX.1-schnell"
pipeline_flux_s_quantized = FluxPipeline.from_pretrained(
    model_id_flux_s,
    torch_dtype=torch.float16,
    load_in_4bit=True, token = HUGGING_FACE_TOKEN  # Same quantization parameter
).to("cuda")

# Generate the image (likely needs more than 1 step, but still very few)
image = pipeline_flux_s_quantized(
    prompt=prompt,
    num_inference_steps=8 # A guess for a fast, small model
).images[0]

image.save("quantized_flux_s_output.png")
print("Image from quantized FLUX.1-S saved!")