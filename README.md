# 🎬 Cinematic AI: Text-to-Video Generator

**Cinematic AI** is a generative AI application that transforms text prompts into short, high-quality video clips. Built with **Streamlit** and the **Diffusers** library, this project leverages state-of-the-art latent diffusion models to synthesize videos in a resource-efficient manner.

This project demonstrates the implementation of a multi-stage generation pipeline (Text-to-Image $\to$ Image-to-Video) optimized for consumer-grade hardware through aggressive memory management techniques.

## 🚀 Features

* **Multi-Model Architecture:**
    * **SDXL Turbo:** For rapid prototyping and fast inference.
    * **SDXL Base 1.0:** For high-fidelity, photorealistic image generation.
    * **Stable Video Diffusion (SVD):** Converts the generated seed images into 14-frame video sequences.
* **Memory Optimization:** Implements **Sequential Model Offloading**, loading and unloading models from VRAM on-demand to prevent Out-Of-Memory (OOM) errors on limited GPU resources.
* **4-Bit Quantization:** Utilizes 4-bit loading strategies to drastically reduce model footprint while maintaining generation quality.
* **Interactive UI:** A clean Streamlit interface allowing users to toggle models, visualize the intermediate seed image, and watch the final generated video side-by-side.

## 🛠️ Tech Stack

* **Python 3.10+**
* **Deep Learning:** PyTorch, Diffusers, Transformers
* **Optimization:** Accelerate, BitsAndBytes (for quantization)
* **Frontend:** Streamlit
* **Utilities:** Python-dotenv, OpenCV (via SVD export)

## ⚙️ Installation

1.  **Clone the repository**
    ```bash
    git clone [https://github.com/Abbasi06/CinematicAI-Text-to-Video-Generator.git](https://github.com/Abbasi06/CinematicAI-Text-to-Video-Generator.git)
    cd CinematicAI-Text-to-Video-Generator
    ```

2.  **Create a virtual environment**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: Ensure you have `torch`, `diffusers`, `streamlit`, `transformers`, `accelerate`, `bitsandbytes`, and `python-dotenv` installed)*

4.  **Set up Environment Variables**
    Create a `.env` file in the root directory and add your Hugging Face token:
    ```bash
    HUGGING_FACE_TOKEN=your_hf_token_here
    ```
    *This is required to download the SDXL and SVD models.*

## 🖥️ Usage

1.  Run the Streamlit application:
    ```bash
    streamlit run main.py
    ```

2.  Open your browser to the local URL provided (usually `http://localhost:8501`).

3.  **Workflow:**
    * Enter a text prompt (e.g., *"An astronaut riding a horse on the moon, photorealistic"*).
    * Select your model: **SDXL Turbo** (Fast) or **SDXL Base** (High Quality).
    * Click **Generate Video**.

4.  **Outputs:**
    * The application will display the generated seed image and the resulting video.
    * Files are automatically saved to the local `output/` directory with unique timestamps.

## 🧩 Architecture & Optimization

The core logic resides in `StableDiffusion.py`. To ensure the application runs on standard GPUs (e.g., NVIDIA T4 or RTX series) without crashing, the pipeline uses a **Garbage Collection Strategy**:

1.  **Stage 1 (T2I):** The Text-to-Image pipeline loads (4-bit), generates the seed image, and is immediately deleted from memory.
2.  **Cleanup:** `gc.collect()` and `torch.cuda.empty_cache()` are invoked to clear VRAM.
3.  **Stage 2 (I2V):** Only after cleanup does the Stable Video Diffusion pipeline load to process the video.

## 📂 Project Structure

```text
├── main.py                # Frontend UI logic (Streamlit)
├── StableDiffusion.py     # Backend generation logic and memory management
├── .env                   # Environment variables (API Keys)
├── output/                # Directory for generated images and videos
└── requirements.txt       # Project dependencies
