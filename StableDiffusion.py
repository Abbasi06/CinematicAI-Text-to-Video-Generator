# Backend.py
import torch
from diffusers import AutoPipelineForText2Image, StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_video
import gc # Garbage Collector interface
import os
from dotenv import load_dotenv
import time

load_dotenv()
HUGGING_FACE_TOKEN = os.getenv("HUGGING_FACE_TOKEN")

# Class TextToVideoGenerator:
class TextToVideoGenerator:
    def __init__(self, model_choice: str = "SDXL Turbo"):
        """
        Initializer is now very lightweight. It only stores the model choice.
        Models will be loaded and unloaded on demand.
        """
        self.model_choice = model_choice
    
    def generate(self, prompt: str, image_output_path: str = "seed_image.png", video_output_path: str = "output_video.mp4"):
        """
        Generates a video by loading and unloading models sequentially to save VRAM.
        
        Returns: 
            (str, str, float, float): image_path, video_path, t2i_time, svd_time
        """
        # Generate Seed Image
        t2i_pipeline = None 
        start_t2i_time = time.time()
        try:
            print("Loading Text-to-Image model...")
            if self.model_choice == "SDXL Turbo":
                t2i_pipeline = AutoPipelineForText2Image.from_pretrained(
                    "stabilityai/sdxl-turbo", dtype=torch.float16, variant="fp16", load_in_4bit=True
                )
                t2i_params = {"num_inference_steps": 1, "guidance_scale": 0.0}
            else: # SDXL Base
                t2i_pipeline = AutoPipelineForText2Image.from_pretrained(
                    "stabilityai/stable-diffusion-xl-base-1.0", dtype=torch.float16, variant="fp16", load_in_4bit=True
                )
                t2i_params = {"num_inference_steps": 25, "guidance_scale": 7.5}
        
            print("Generating seed image...")
            seed_image = t2i_pipeline(prompt=prompt, **t2i_params).images[0]
            seed_image.save(image_output_path)
            print(f"Seed image saved to {image_output_path}")
            
            end_t2i_time = time.time()
            t2i_duration = end_t2i_time - start_t2i_time

        finally:
            # Unload the T2I model from memory
            if t2i_pipeline:
                del t2i_pipeline
            gc.collect()
            torch.cuda.empty_cache()
            print("Text-to-Image model unloaded from VRAM.")


        # Generate Video
        svd_pipeline = None
        start_svd_time = time.time()
        try:
            print("\nLoading Stable Video Diffusion model...")
            svd_pipeline = StableVideoDiffusionPipeline.from_pretrained(
                "stabilityai/stable-video-diffusion-img2vid", torch_dtype=torch.float16, variant="fp16", load_in_4bit= True
            )
            # svd_pipeline.to("cuda")
            svd_pipeline.enable_model_cpu_offload()
            
            print("Processing seed image into video...")
            loaded_seed_image = load_image(image_output_path).resize((1024, 576))
            
            video_frames = svd_pipeline(
                loaded_seed_image, num_frames=14, decode_chunk_size=1, num_inference_steps= 10, motion_bucket_id=127, fps=7, noise_aug_strength=0.02
            ).frames[0]
            
            export_to_video(video_frames, video_output_path)
            print(f"Video saved to {video_output_path}")

            end_svd_time = time.time()
            svd_duration = end_svd_time - start_svd_time
            
        finally:
            # Unload the SVD model from memory
            if svd_pipeline:
                del svd_pipeline
            gc.collect()
            torch.cuda.empty_cache()
            print("Stable Video Diffusion model unloaded from VRAM.")
        
        return image_output_path, video_output_path, t2i_duration, svd_duration


