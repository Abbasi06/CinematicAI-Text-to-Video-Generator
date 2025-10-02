# Backend.py
import torch
from diffusers import AutoPipelineForText2Image, StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_video
from PIL import Image
import os
from dotenv import load_dotenv

load_dotenv()
HUGGING_FACE_TOKEN = os.getenv("HUGGING_FACE_TOKEN")

class TextToVideoGenerator:
    def __init__(self, model_choice: str = "SDXL Turbo"):
        """
        Initializes the generator by loading the selected text-to-image model and the SVD model.
        """
        self.model_choice = model_choice
        print(f"--- Initializing Generator with {model_choice} ---")
        torch.cuda.empty_cache()
        # --- Load the chosen Text-to-Image Model ---
        if self.model_choice == "SDXL Turbo":
            print("Loading Quantized SDXL-Turbo...")
            self.t2i_pipeline = AutoPipelineForText2Image.from_pretrained(
                "stabilityai/sdxl-turbo",
                dtype=torch.float16,
                variant="fp16",
                load_in_4bit=True
            ).to("cuda")
        elif self.model_choice == "SDXL Base":
            print("Loading Quantized SDXL-Base...")
            self.t2i_pipeline = AutoPipelineForText2Image.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0",
                dtype=torch.float16,
                variant="fp16",
                load_in_4bit=True
            ).to("cuda")
        else:
            raise ValueError("Invalid model choice provided.")

        # --- Load the Stable Video Diffusion Model ---
        print("Loading Stable Video Diffusion Pipeline...")
        self.svd_pipeline = StableVideoDiffusionPipeline.from_pretrained(
            "stabilityai/stable-video-diffusion-img2vid-xt",
            torch_dtype=torch.float16,
            variant="fp16"
        ).to("cuda")
        print("--- Models loaded successfully. Generator is ready. ---")

    def generate(self, prompt: str, image_output_path: str = "seed_image.png", video_output_path: str = "output_video.mp4"):
        """
        Generates a video from a text prompt using the pre-loaded models.
        """
        print(f"\n--- Generating with {self.model_choice} for prompt: '{prompt}' ---")

        # --- Set inference parameters based on the chosen model ---
        if self.model_choice == "SDXL Turbo":
            t2i_params = {"num_inference_steps": 1, "guidance_scale": 0.0}
        else: # SDXL Base
            t2i_params = {"num_inference_steps": 25, "guidance_scale": 7.5}

        # Part 1: Generate Seed Image
        print("Generating seed image...")
        seed_image = self.t2i_pipeline(prompt=prompt, **t2i_params).images[0]
        seed_image.save(image_output_path)
        print(f"Seed image saved to {image_output_path}")

        # Part 2: Generate Video
        print("Processing seed image into video...")
        loaded_seed_image = load_image(image_output_path)
        loaded_seed_image = loaded_seed_image.resize((1024, 576))

        video_frames = self.svd_pipeline(
            loaded_seed_image,
            num_frames=25,
            decode_chunk_size=8,
            motion_bucket_id=127,
            fps=7,
            noise_aug_strength=0.02
        ).frames[0]

        export_to_video(video_frames, video_output_path)
        print(f"Video saved to {video_output_path}")
        
        return image_output_path, video_output_path