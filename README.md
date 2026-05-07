# 🛰️ SARNet: SAR to Optical Image Translation & Terrain Classification

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](#) *(**Demo Link:** Insert your deployed Streamlit link here)*

SARNet is an end-to-end Deep Learning pipeline designed to bridge the gap between radar and optical satellite imagery. It utilizes a **CycleGAN** architecture to translate grayscale Synthetic Aperture Radar (SAR - Sentinel-1) imagery into colorized optical imagery (Sentinel-2). Furthermore, it passes the generated optical image into a downstream **ResNet-18 Classifier** to automatically categorize the terrain.

---

## 🏗️ System Architecture

The pipeline consists of three primary Deep Learning models running seamlessly in a Streamlit web application:

### 1. CycleGAN Generator (SAR $\rightarrow$ Optical)
* **Architecture:** ResNet-based encoder-decoder (6 Residual Blocks).
* **Function:** Takes a single-channel (grayscale) SAR image patch as input and synthesizes a 3-channel (RGB) optical image.
* **Why it matters:** SAR imagery can penetrate clouds and be captured at night, but it is notoriously difficult for humans to interpret. This generator provides a human-readable optical estimate of the ground truth.

### 2. PatchGAN Discriminator
* **Architecture:** 70x70 PatchGAN.
* **Function:** Evaluates overlapping patches of the generated optical image to determine if the local textures look "Real" or "Fake". 
* **Role in App:** In the inference pipeline, the discriminator provides a **Realism Score** (0-100%), allowing the user to gauge the structural confidence of the generated image.

### 3. Terrain Classifier
* **Architecture:** Fine-tuned ResNet-18.
* **Function:** Takes the *generated* optical image, resizes it to 224x224, applies ImageNet normalization, and classifies the terrain into one of four classes:
  * 🌾 `Agriculture`
  * 🏜️ `Barrenland`
  * 🏞️ `Grassland`
  * 🏙️ `Urban`

---

## 🚀 Application Workflow

1. **Input:** The user uploads a Sentinel-1 SAR image (or selects a sample asset).
2. **Preprocessing:** The image is converted to grayscale, resized to 256x256, and normalized to `[-1, 1]`.
3. **Translation:** The image is passed through the Generator to synthesize the RGB Optical equivalent.
4. **Analysis:** 
   * The generated image is scored by the Discriminator.
   * The generated image is classified by the ResNet-18 Terrain Classifier.
5. **Output:** The application displays the synthesized optical image (with a download option), the predicted terrain class, and the discriminator's realism score.

---

## 🛠️ Local Installation & Usage

To run this project locally, clone the repository and install the dependencies. *Note: The `requirements.txt` is specifically configured with CPU-only PyTorch wheels for lightweight deployment on Streamlit Community Cloud.*

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/SARNet.git
cd SARNet

# 2. Create a virtual environment
python -m venv sarnet_env
sarnet_env\Scripts\activate  # Windows
# source sarnet_env/bin/activate  # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the Streamlit application
streamlit run app.py
```

## 📁 Repository Structure

* `app.py`: The main Streamlit web application script.
* `model.py`: PyTorch class definitions for the Generator, PatchGAN Discriminator, and ResNet Classifier.
* `utils.py`: Helper functions for image tensor preprocessing and postprocessing.
* `requirements.txt`: Environment dependencies.
* `Results/Epochs/`: Directory containing the compiled, lightweight model weights:
  * `inference_model.pth`: The stripped CycleGAN weights.
  * `subset_optical_classifier.pth`: The fine-tuned ResNet-18 weights.

---
*Developed using PyTorch & Streamlit.*
