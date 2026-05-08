import streamlit as st
import torch
from PIL import Image
import os
import io
import torchvision.transforms as transforms
from model import Generator, PatchGANDiscriminator, get_optical_classifier
from utils import preprocess_image, postprocess_tensor

# Streamlit App Configuration
st.set_page_config(
    page_title="SARNet",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
CHECKPOINT_PATH = "Results/Epochs/inference_model.pth"
CLASSIFIER_PATH = "Results/Epochs/subset_optical_classifier.pth"
SAMPLE_IMAGE_PATH = "assets/sample_sar.png"
CLASS_NAMES = ['agri', 'barrenland', 'grassland', 'urban']

@st.cache_resource
def load_models():
    generator, discriminator, classifier = None, None, None
    
    # Load CycleGAN models
    if os.path.exists(CHECKPOINT_PATH):
        try:
            generator = Generator(in_channels=1, out_channels=3, ngf=64, n_res_blocks=6)
            discriminator = PatchGANDiscriminator(in_channels=3, ndf=64)
            checkpoint = torch.load(CHECKPOINT_PATH, map_location=torch.device('cpu'))
            generator.load_state_dict(checkpoint['G_AB'])
            discriminator.load_state_dict(checkpoint['D_B'])
            generator.eval()
            discriminator.eval()
        except Exception as e:
            st.error(f"Error loading CycleGAN models: {e}")
            
    # Load Optical Classifier
    if os.path.exists(CLASSIFIER_PATH):
        try:
            classifier = get_optical_classifier(num_classes=4)
            classifier.load_state_dict(torch.load(CLASSIFIER_PATH, map_location=torch.device('cpu')))
            classifier.eval()
        except Exception as e:
            st.error(f"Error loading Classifier: {e}")

    return generator, discriminator, classifier

def main():
    st.title("🛰️ SARNet: SAR to Optical Image Translation")
    st.markdown("Convert grayscale Synthetic Aperture Radar (SAR) imagery into colorized optical imagery using CycleGAN and classify the terrain type.")

    # Sidebar
    with st.sidebar:
        st.header("SARNet Settings")
        st.info("**Model Architecture:** CycleGAN (ResNet Generator + PatchGAN Discriminator)")
        st.info("**Classifier:** ResNet-18 (Fine-Tuned)")
        st.info("**Device:** CPU Inference")
        
        st.markdown("### How to use:")
        st.markdown("1. Upload a SAR image (or use the sample).")
        st.markdown("2. Click 'Translate and Classify'.")
        st.markdown("3. View the generated optical image, Discriminator score, and Terrain classification.")
        
        use_sample = st.button("Try Sample Image", use_container_width=True)

    # Main area
    col1, col2 = st.columns(2)
    
    uploaded_file = None
    if use_sample:
        if os.path.exists(SAMPLE_IMAGE_PATH):
            uploaded_file = SAMPLE_IMAGE_PATH
        else:
            st.warning("Sample image not found in assets.")
    else:
        uploaded_file = st.file_uploader("Upload SAR Image (PNG, JPG, TIF)", type=["png", "jpg", "jpeg", "tif", "tiff"])

    if uploaded_file:
        try:
            # Display Input
            input_image = Image.open(uploaded_file).convert("RGB")
            with col1:
                st.subheader("Input SAR Image")
                st.image(input_image, use_column_width=True)
            
            # Action button
            if st.button("Translate and Classify", type="primary"):
                generator, discriminator, classifier = load_models()
                
                if generator is None or discriminator is None:
                    st.error(f"🚀 **CycleGAN Model not found or failed to load.** Please ensure {CHECKPOINT_PATH} exists.")
                    return
                
                with st.spinner("Translating and Classifying..."):
                    # Preprocess for CycleGAN
                    input_tensor = preprocess_image(input_image)
                    
                    # Inference
                    with torch.no_grad():
                        # 1. Generate Optical Image
                        output_tensor = generator(input_tensor)
                        
                        # 2. Discriminator Realism Score
                        d_out = discriminator(output_tensor)
                        realism_score = torch.sigmoid(d_out).mean().item()
                        
                        # Postprocess generator output to PIL Image
                        output_image = postprocess_tensor(output_tensor)
                        
                        # 3. Classify the generated optical image
                        pred_class_name = "N/A"
                        confidence = 0.0
                        if classifier is not None:
                            # Preprocess for Classifier
                            classifier_transform = transforms.Compose([
                                transforms.Resize((224, 224)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                            ])
                            class_input = classifier_transform(output_image).unsqueeze(0)
                            
                            # Classification Prediction
                            class_logits = classifier(class_input)
                            probs = torch.softmax(class_logits, dim=1)
                            confidence, pred_idx = torch.max(probs, dim=1)
                            pred_class_name = CLASS_NAMES[pred_idx.item()].upper()
                            confidence = confidence.item()
                        else:
                            st.warning(f"Optical Classifier not found at {CLASSIFIER_PATH}. Skipping classification.")

                    with col2:
                        st.subheader("Output Optical Image")
                        st.image(output_image, use_column_width=True)
                        
                        # Download button
                        buf = io.BytesIO()
                        output_image.save(buf, format="PNG")
                        byte_im = buf.getvalue()
                        st.download_button(
                            label="Download Result",
                            data=byte_im,
                            file_name="translated_optical.png",
                            mime="image/png",
                            use_container_width=True
                        )
                        
                    # Metrics & Results
                    st.divider()
                    st.subheader("Analysis Results")
                    
                    # Classification Results
                    if classifier is not None:
                        st.info(f"📍 **Predicted Terrain Class:** {pred_class_name} (Confidence: {confidence*100:.1f}%)")
                    
                    # Realism Score Display
                    score_pct = realism_score * 100
                    if score_pct >= 50:
                        st.success(f"🤖 **Discriminator Realism Score:** {score_pct:.2f}% (Classified as REAL)")
                    else:
                        st.warning(f"🤖 **Discriminator Realism Score:** {score_pct:.2f}% (Classified as FAKE)")
                        
                    st.progress(realism_score)
                    
        except Exception as e:
            st.error(f"Error processing image: {e}")

if __name__ == "__main__":
    main()
