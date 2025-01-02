import gradio as gr
import torch
import torch.nn.functional as F
import numpy as np
import xgboost as xgb
from tensorflow.keras.models import load_model
from torchvision import transforms
from PIL import Image
import tensorflow as tf
import pickle
import cv2

# Load all models
models = {
    "ResNet": load_model("models/UltimateOne_ResNet.h5"),
    # "ResNet-CUTMIX": load_model("/content/drive/MyDrive/ModelCV/Model Sigma Balls/Cutmix/ResNet50 - Cutmix Sepcialized.h5"),
    "Xception": load_model("models/UltimateOne_Xception.h5"),
    # "Xception + Additional Dataset": load_model("models/Xception_with_dataold.h5"),
    # "Xception-CUTMIX": load_model("/content/drive/MyDrive/ModelCV/Model Sigma Balls/Cutmix/Xception - Cutmix Specialized.h5"),
    # "ViT": torch.load("models/vit_entire_model (1).pth", map_location=torch.device('cpu')),
    "XGBoost": pickle.load(open("models/xgb_model_hsv+dataold.pkl", "rb")),
    "XGBoost-FMPEG": pickle.load(open("models/xgb_model_fmpeg+dataold.pkl", "rb"))
}

# Class labels
class_names = ["Not AI-generated", "AI-generated"]

# Define preprocessing for PyTorch model
torch_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

import cv2
import numpy as np
from skimage import feature
from scipy.fftpack import dct

# Helper function for DCT-based Color Layout
def compute_dct(block):
    return dct(dct(block.T, norm='ortho').T, norm='ortho')

# Color Layout Descriptor (CLD) using DCT
def extract_color_layout(image):
    # Resize to 8x8 for compact representation
    image_resized = cv2.resize(image, (8, 8))
    # Convert to YCrCb color space
    image_ycc = cv2.cvtColor(image_resized, cv2.COLOR_BGR2YCrCb)

    # Split the Y, Cr, and Cb channels
    y, cr, cb = cv2.split(image_ycc)

    # Compute DCT coefficients on each channel
    y_dct = compute_dct(y)
    cr_dct = compute_dct(cr)
    cb_dct = compute_dct(cb)

    # Use only a subset of the DCT coefficients (e.g., top-left 6x6 block)
    features = np.hstack([y_dct[:6, :6].flatten(), cr_dct[:6, :6].flatten(), cb_dct[:6, :6].flatten()])
    return features

# Color Structure Descriptor (CSD) using structured histogram
def extract_color_structure(image):
    # Convert to HSV color space for color information
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Use a structuring element (e.g., 3x3) to capture spatial layout of colors
    struct_elem = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    image_morphed = cv2.morphologyEx(image_hsv, cv2.MORPH_CLOSE, struct_elem)

    # Calculate histogram with structure in mind
    hist = cv2.calcHist([image_morphed], [0, 1], None, [8, 8], [0, 180, 0, 256])
    cv2.normalize(hist, hist)

    return hist.flatten()

# Edge Histogram Descriptor (EHD) using directional edges
def extract_edge_histogram(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect edges using Canny
    edges = cv2.Canny(gray, 100, 200)

    # Compute directionality of edges
    angles = feature.hog(edges, orientations=5, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=False)

    return angles

# Homogeneous Texture Descriptor (HTD) using LBP and GLCM
def extract_homogeneous_texture(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Compute Local Binary Patterns (LBP) as texture measure
    lbp = feature.local_binary_pattern(gray, P=8, R=1, method='uniform')

    # Histogram of LBP values (normalized)
    (hist_lbp, _) = np.histogram(lbp.ravel(), bins=np.arange(0, 11), range=(0, 10))
    hist_lbp = hist_lbp.astype("float")
    hist_lbp /= hist_lbp.sum()

    # Use GLCM (Gray-Level Co-occurrence Matrix) for second-order texture
    glcm = feature.graycomatrix(gray, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], symmetric=True, normed=True)
    contrast = feature.graycoprops(glcm, 'contrast').flatten()
    homogeneity = feature.graycoprops(glcm, 'homogeneity').flatten()

    # Combine LBP and GLCM features
    texture_features = np.hstack([hist_lbp, contrast, homogeneity])

    return texture_features

# Main function to extract MPEG-7-like features
def extract_mpeg7_features(image):
    # image = cv2.imread(image_path)

    color_layout = extract_color_layout(image)
    color_structure = extract_color_structure(image)
    edge_histogram = extract_edge_histogram(image)
    homogeneous_texture = extract_homogeneous_texture(image)

    # Combine all features into a single feature vector
    features = np.hstack([color_layout, color_structure, edge_histogram, homogeneous_texture])

    return features

def preprocess_image(image, size=(224, 224)):
  # Load the image from the file path
  image = np.array(image)

  # Check if the image was loaded successfully
  if image is None:
      raise ValueError(f"Image not found or unable to load: {image}")

  # Resize the image to the given size
  image_resized = cv2.resize(image, size)
  return image_resized

def hsv_histogram_features(image, bins=(8, 8, 8)):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv_image], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()

    # Ensure the histogram is 512-dimensional by padding or truncating
    if len(hist) < 512:
        hist = np.pad(hist, (0, 512 - len(hist)), 'constant')
    elif len(hist) > 512:
        hist = hist[:512]

    return hist

def extract_features(image):
    # Preprocess image
    image = preprocess_image(image)

    # Extract individual features
    hsv_feat = hsv_histogram_features(image)

    # Combine all features into a single feature vector (if more features are needed, they can be added here)
    return np.array(hsv_feat).reshape(1, -1)

def extract_features_ffmpeg(image):
  image = preprocess_image(image)

  # Extract individual features
  ffmpeg_feat = extract_mpeg7_features(image)

  # Combine all features into a single feature vector (if more features are needed, they can be added here)
  return np.array(ffmpeg_feat).reshape(1, -1)

# Define the prediction function
def classify_image(image, model_name):
    if model_name in ["ResNet", "Xception", "Xception + Additional Dataset"]:
        # Use TensorFlow model
        model = models[model_name]
        image = np.array(image)
        image = tf.image.resize(image, (224, 224))
        image = image / 255.0
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        prediction = model.predict(image)
        confidence = prediction[0][0]
        predicted_class = "AI-generated" if confidence > 0.5 else "Not AI-generated"
        confidence = confidence if confidence > 0.5 else 1 - confidence
    elif model_name == "ViT":
        # Use PyTorch model
        model = models[model_name]
        image = torch_transform(image).unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            outputs = model(image)
            raw_output = outputs.logits if hasattr(outputs, "logits") else outputs
            probabilities = F.softmax(raw_output, dim=1).numpy()
        confidence = probabilities[0][0]  # Assuming index 0 is "Human-generated"
        predicted_class = "AI-generated" if confidence > 0.5 else "Not AI-generated"
        confidence = confidence if confidence > 0.5 else 1 - confidence
    elif model_name in ["XGBoost", "XGBoost-FMPEG"]:
        features = extract_features(image)
        prediction = models["XGBoost"].predict(features)
        confidence = prediction[0]
        predicted_class = "AI-generated" if confidence > 0.5 else "Not AI-generated"
        confidence = confidence if confidence > 0.5 else 1 - confidence

    return f"{predicted_class}"

# Define the Gradio function to handle inputs
def predict_with_example(uploaded_image, selected_example, model_name):
    # If uploaded image is provided, use it
    if uploaded_image is not None:
        image = uploaded_image
    # Otherwise, use the selected example image
    elif selected_example is not None:
        image = Image.open(selected_example)  # Load the selected image
    else:
        return "Please upload an image or select an example."

    # Classify the image using the selected model
    return classify_image(image, model_name)

# Path to example images
example_image_paths = [
    "comvis_testing_data/018_generated.png",
    "comvis_testing_data/127_generated.png",
    "comvis_testing_data/038_generated.png",
    "comvis_testing_data/130_generated.png",
    "comvis_testing_data/147_generated.png",
    "comvis_testing_data/Salinan inpainting.png",
    "comvis_testing_data/inpainting (3).png",
    "comvis_testing_data/inpainting (2).png",
    "comvis_testing_data/inpainting (1).png",
    "comvis_testing_data/inpainting.png",
    "comvis_testing_data/018_original.png",
    "comvis_testing_data/038_original.png",
    "comvis_testing_data/127_original.png",
    "comvis_testing_data/130_original.png",
    "comvis_testing_data/147_original.png",
    "comvis_testing_data/Salinan original.png",
    "comvis_testing_data/original (4).png",
    "comvis_testing_data/original (3).png",
    "comvis_testing_data/original (2).png",
    "comvis_testing_data/original (1).png",
]

interface = gr.Interface(
    fn=predict_with_example,
    inputs=[
        gr.Image(type="pil", label="Upload Image (Optional)"),
        gr.Gallery(value=example_image_paths, label="Select Example Image (Optional)", interactive=True),
        gr.Dropdown(list(models.keys()), label="Select Model")
    ],
    outputs=gr.Textbox(label="Prediction"),
    title="AI vs Human Image Classifier",
    description="Upload an image or select an example image, and select a model to classify whether it was generated by AI or created by a human. The dataset used for training the model is from https://www.kaggle.com/datasets/danielmao2019/deepfakeart?select=similar which dataset source is from wikiart for the non ai generated image and for the AI generated image is using the StabilityAI Stable Difussion 2 Inpainting https://huggingface.co/stabilityai/stable-diffusion-2-inpainting"
)

interface.launch(debug=True)