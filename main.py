# main.py

import streamlit as st
from StableDiffusion import TextToVideoGenerator
import os
import time

# --- Page Configuration ---
st.set_page_config(
    page_title="Cinematic AI",
    page_icon="🎬",
    layout="wide"
)

# --- Caching the Generator ---
# This is the most important step for performance.
# It ensures the heavy models are loaded only ONCE.
@st.cache_resource
def load_generator(model_choice):
    """Loads and caches the generator based on the selected model."""
    return TextToVideoGenerator(model_choice=model_choice)

# --- UI Layout ---
st.title("🎬 Cinematic AI: Your Ideas in Motion")
st.markdown("Describe a scene, and watch as AI crafts a unique video clip for you.")

# Create a directory for outputs if it doesn't exist
if not os.path.exists("output"):
    os.makedirs("output")

# --- User Inputs ---
with st.container():
    prompt = st.text_area("Enter your prompt:", value="", placeholder="e.g., An astronaut riding a horse on the moon, photorealistic", height=100)
    
    col1, col2 = st.columns([1, 1])
    with col1:
        model_choice = st.radio(
            "Choose a Model:",
            ("SDXL Turbo", "SDXL Base"),
            captions=("Fastest generation, good for rapid ideas.", "Higher quality, but much slower.")
        )
    with col2:
        st.write("") # Spacer
        st.write("") # Spacer
        generate_button = st.button("Generate Video", type="primary", use_container_width=True)

# --- Generation Logic ---
if generate_button:
    start_time = time.time()
    if not prompt:
        st.warning("Please enter a prompt to generate a video.")
    else:
        # Define unique output paths for each generation
        timestamp = int(time.time())
        image_path = os.path.join("output", f"seed_image_{timestamp}.png")
        video_path = os.path.join("output", f"generated_video_{timestamp}.mp4")
        
        # Display a spinner while the models are working
        with st.spinner(f"Generating with {model_choice}, this may take a few minutes..."):
            try:
                # Load the appropriate generator (will be cached)
                generator = load_generator(model_choice)
                
                # Run the generation process
                final_image_path, final_video_path = generator.generate(
                    prompt=prompt,
                    image_output_path=image_path,
                    video_output_path=video_path
                )
                elapsed_time = int(time.time())- timestamp


                # --- Display the results ---
                st.subheader("Results")
                
                res_col1, res_col2 = st.columns(2)
                with res_col1:
                    st.image(final_image_path, caption="Generated Seed Image")
                with res_col2:
                    # To display the video, we need to read it as bytes
                    with open(final_video_path, "rb") as video_file:
                        video_bytes = video_file.read()
                    st.video(video_bytes)
                print(f"Took {elapsed_time}s to generate the video.")

            except Exception as e:
                st.error(f"An error occurred: {e}")