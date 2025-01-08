# 🔧 Deep Learning Project

## Class: LA01
**Course:** 🌐 Deep Learning  
**University:** 🎓 Bina Nusantara University  
**Major:** 💻 Computer Science 🎓    

## 🔰 Group Members
- **2602058932** - Jovan Amarta Liem  
- **2602070351** - Jonathan Surya Sanjaya  
- **2602178911** - Cecillia Tjung  
- **2602160510** - Tasya Aulianissa

## 🕹️ Overview
We created this project to solve the problem of using artificial intelligence (AI) in art, especially art that produced by AI. AI provides many benefits, but if its uncontrolled, it can cause problems such as plagiarism, pirating, and economic loss for the original artist who created the art. 

We create a solution to the problem by building AI models that can detect whether the art or the image is an AI generated image. To create the models, we took the datasets from:
- https://www.kaggle.com/datasets/ravidussilva/real-ai-art
- https://www.kaggle.com/datasets/danielmao2019/deepfakeart

The models that we created consists of 5 models, 3 deep learning models, and 2 machine learning models with different feature extraction method. Here are the models that we created: 
- ResNet50V2
- Xception
- ViT (Google / ViT - base - patch16 - 224)
- XGBoost with HSV
- Random Forest with HSV
- XGBoost with HSV + Edge Detection
- Random Forest with HSV + Edge Detection
- XGBoost with MPEG7
- Random Forest with MPEG7

## 🖼️ Dataset Samples

### AI-Generated Art
![AI Art Sample](Images/inpainting.png)

### Human-Created Art
![Human Art Sample](Images/original.png)

# Experiment Results

## Previous Experiments
| **Model**               | **Total Images** | **Accuracy** |
|-------------------------|------------------|--------------|
| **XGB + HSV**           | 18000 images     | 62,25%       |
| **XGB + HSV + Edge**    | 18000 images     | 63,89%       |
| **XGB + MPEG7**         | 3600 images      | 64,17%       |
| **RF + HSV**            | 18000 images     | 60,75%       |
| **RF + HSV + Edge**     | 18000 images     | 62,36%       |
| **RF + MPEG7**          | 3600 images      | 58,47%       |
| **Xception**            | 18000 images     | 72,16%       |
| **ResNet**              | 18000 images     | 72,80%       |
| **ViT (Vision Transformer)** | 18000 images | 80,80%       |

## Latest
| **Model**               | **Total Images** | **Accuracy** |
|--------------------------|------------------|-------------|
| **XGB + HSV**           | 940 images       | 89,36%       |
| **XGB + HSV + Edge**    | 940 images       | 88,30%       |
| **XGB + MPEG7**         | 940 images       | 87,23%       |
| **XGB + EDGE**          | 940 images       | 49,47%       |
| **RF + HSV**            | 940 images       | 89,36%       |
| **RF + HSV + Edge**     | 940 images       | 88,30%       |
| **RF + MPEG7**          | 940 images       | 74,47%       |
| **RF + EDGE**           | 940 images       | 56,91%       |
| **Xception**            | 3660 images      | 81%          |
| **ResNet**              | 3660 images      | 75%          |
| **ViT (Vision Transformer)** | 3660 images      | 81,14%       |
---

# 💻 Installation Guide 

## 🏠 Local Setup
1. 🐍 Ensure you have **Python 3.10** installed on your system.  
   👉 [Download Python here](https://www.python.org/downloads/) if needed.  

2. 📥 Clone this repository to your local machine:  
   ```bash
   git clone https://github.com/JovanLiem/AI-Art-Detection.git
3. 📦 Install the required packages using the following commands:
    ```bash
    pip install -r requirements.txt

4. 👉 Download the model from the given link here:  
[🔗 Download Model](https://github.com/JovanLiem/AI-Art-Detection/tree/main/Model)  
Or visit this tutorial to directly implement the model in your Jupyter Notebook:  
[📖 Model Tutorial](https://github.com/JovanLiem/AI-Art-Detection/blob/main/Model/DownloadModelTutorial.ipynb)

5. 🛠️ Ensure the path in `app.py` is fixed according to your system, including the models and testing dataset.

6. 🚀 Run the Application with the following command in your terminal or command prompt:

   - For Windows or Linux:
     ```bash
     python app.py
     ```
   - For macOS (if needed):
     ```bash
     python3 app.py
     ```

## Gradio
We've made it super easy for you to try out our project! Simply click on the links below to test the models instantly via Gradio on Hugging Face. 🖱️✨:
1. **XGBoost (HSV & MPEG-7)**, **Xception**, and **ResNet50V2**  
   👉 [Try it here!](https://huggingface.co/spaces/jovanliem/ai_generated_art_detector)  

2. **ViT (Google / ViT-base-patch16-224)**  
   👉 [Check it out here!](https://huggingface.co/spaces/jovanliem/ai_generated_art_detector_ViT)

<br>
<br>
<br>

🚀 Enjoy the application! 🎉