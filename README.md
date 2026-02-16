**🔥 Image Detection System**


Python · PyTorch · Computer Vision · AICTE


**📋 Project Description**

This project implements an AI-generated image detection system using a hybrid deep learning and frequency analysis approach.
The system analyzes uploaded images and determines whether they are real photographs or AI-generated content.

The model combines pixel-level texture analysis and semantic understanding to improve reliability.
Built using PyTorch and pretrained transformer models, it performs real-time inference and provides confidence scores along with visual evaluation graphs.



**🎯 Objectives**

Build an intelligent system to identify synthetic images

Combine signal processing and deep learning detection methods

Provide real-time classification with confidence

Generate evaluation graphs for analysis

Demonstrate practical computer vision application for AICTE project



**✨ Features**

✅ Hybrid Detection: Combines FFT analysis and CLIP model

✅ Face-Aware Analysis: Uses MTCNN for region-focused inspection

✅ Real-Time Prediction: < 1 second processing

✅ Confidence Score Output

✅ Dataset Evaluation Mode

✅ Visualization Graphs & Charts

✅ No Model Training Required (Pretrained AI)

✅ Ready for Demonstration



**🛠️ Technologies Used**

    | Technology          | Purpose                      |
    | ------------------- | ---------------------------- |
    | Python 3.10         | Programming Language         |
    | PyTorch             | Deep Learning Framework      |
    | Transformers (CLIP) | Image semantic understanding |
    | Facenet-Pytorch     | Face detection               |
    | NumPy               | Image frequency analysis     |
    | Matplotlib          | Visualization                |
    | Pillow              | Image processing             |

  


**📊 Dataset Information**

Source: Mixed real photographs and AI-generated images

Classes: Real, AI-Generated

Format: RGB images

Evaluation: Manual labeled test set




**🚀 Installation & Setup**

Step 1: Install Dependencies

    pip install torch torchvision transformers facenet-pytorch pillow matplotlib numpy pandas

Step 2: Open Notebook

    Upload notebook to Google Colab and run all cells.

    

**📖 Usage Guide**

Single Image Prediction:

  1.Upload image

  2.System analyzes image

  3.Displays classification and confidence

Batch Evaluation

Upload test_images folder:

    test_images/
     real1.jpg
     real2.jpg
     ai1.png
     ai2.png


System generates:

  Prediction table
  
  Accuracy
  
  Graph analysis

  
  

**📈 Model Performance**

Real-time prediction: < 1 second

Hybrid decision improves reliability

Works across faces and general scenes




**🎨 Visualization Output**

The project produces multiple evaluation charts:

  Prediction distribution bar chart
  
  Confidence histogram
  
  Real vs AI pie chart
  
  Detector agreement scatter plot
  
  Result prediction image display

  
  

**💡 Real-World Applications**

📰 Fake news detection

🔐 Digital forensics

📱 Social media verification

🧾 Evidence validation

🧑‍💻 Deepfake identification




**🔧 Detection Method Details**

Frequency Analysis (FFT):

  Detects unnatural pixel texture patterns common in AI images

CLIP Transformer:

  Understands image realism using vision-language embeddings

Decision Fusion:

    Combined Score =
    0.65 × FFT Score + 0.35 × CLIP Score

Final Classes:

  Real

  AI-Generated

  Possibly AI-Generated




**🧪 Testing the Model**

Example Usage

    fft_score = face_only_fft_score(image)

    clip_score = clip_ai_score(image)

    label, confidence = image_decision(fft_score, clip_score)

    print(label, confidence)



**📚 Learning Outcomes**

This project demonstrates:

✅ Computer Vision fundamentals

✅ Transformer model inference

✅ Frequency domain image analysis

✅ Ensemble decision systems

✅ Visualization & evaluation

✅ Practical AI deployment



**📝 Requirements**

    torch

    torchvision

    transformers

    facenet-pytorch

    pillow

    matplotlib

    numpy

    pandas



**🎓 Academic Information**

Project Type: Machine Learning - Classification

Course: MS ELEVATE AICTE Program

Domain: Computer Vision & Pattern Recognition

Difficulty Level: Intermediate

Estimated Time: 4-6 hours



**📊 Project Status**

    ✅ Complete – Ready for submission

**Made with ❤️ for AICTE MS ELEVATE Program**
