

---

# 🌱 Plant Disease Detection for Tomato Plants using Deep Learning 🍅

Welcome to the **Plant Disease Detection for Tomato Plants** project! This repository contains the implementation of a deep learning model designed to detect diseases in tomato plants using convolutional neural networks (CNNs). The project leverages state-of-the-art techniques to enhance early disease detection, improve crop yield, and promote sustainable agricultural practices. 🚜🌾

## 📋 Project Overview

### Problem Statement
Tomato plant diseases can significantly impact crop yields and farmer income. Traditional methods for disease detection are often inaccurate and time-consuming, leading to delayed treatments and increased crop losses. 

### Solution
This project addresses the challenge by developing a CNN model that can accurately identify various tomato plant diseases. The model is integrated with a user-friendly interface for both real-time detection using a webcam and manual image upload analysis.

### Key Metrics
- **Accuracy:** 94.77%
- **Dataset Size:** 18,000+ images
- **Classes:** Multiple tomato plant diseases

## 🛠️ Tech Stack

- **Programming Languages:** Python 🐍
- **Deep Learning Framework:** TensorFlow, Keras 🤖
- **Image Processing:** OpenCV 📷
- **Web Framework:** Streamlit 🌐
- **Data Handling:** Pandas, NumPy 📊
- **Visualization:** Matplotlib 📈
- **GPU Acceleration:** NVIDIA RTX 2050 with CUDA 11.2, cuDNN 8.1 🚀

## 🚀 Features

- **Real-Time Detection:** Uses webcam to analyze tomato leaves in real-time.
- **Manual Image Upload:** Allows users to upload images for disease detection.
- **Detailed Disease Information:** Provides information on detected diseases and recommended treatments.
- **Data Augmentation:** Enhances the dataset variability with techniques like rotation, flipping, and zooming.



## 💻 Installation

### Prerequisites
- Python 3.7 or higher
- NVIDIA GPU with CUDA 11.2 and cuDNN 8.1

### Steps

1. **Clone the repository:**
    ```bash
    git clone https://github.com/sriramkrish68/tomato-disease-detector.git
    cd tomato-disease-detector
    ```

2. **Create a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4. **Download and preprocess the dataset:**
    ```bash
    # Ensure the dataset is placed in the `data` folder
    python src/data_preprocessing.py
    ```

5. **Train the model:**
    ```bash
    python src/train.py
    ```

6. **Evaluate the model:**
    ```bash
    python src/evaluation.py
    ```

7. **Run the Streamlit app:**
    ```bash
    streamlit run src/app.py
    ```

## 📊 Evaluation Metrics

- **Accuracy:** Proportion of correctly classified samples.
- **Precision:** Proportion of true positive samples among the predicted positives.
- **Recall:** Proportion of true positive samples among the actual positives.
- **F1 Score:** Harmonic mean of precision and recall.

## 🖼️ Sample Outputs

##Architecture
![fig 1](https://github.com/user-attachments/assets/ce304929-70aa-4be5-aaa8-e40adcbf3bff)

##User Interface
![fig 7 b](https://github.com/user-attachments/assets/3123611f-df85-4cdd-b87a-97df1ff9d53c)


### Confusion Matrix
![Confusion Matrix](images/confusio![Uploading fig 4.png…]()
n_matrix.png)

### Accuracy and Loss Graphs
![Accuracy and Loss](images/accuracy_loss.png)
<img width="227" alt="fig 6 b" src="https://github.com/user-attachments/assets/924d8617-b51c-4852-a261-6f17dafcb634">

## 🤝 Contributing

We welcome contributions to enhance the project! Please follow these steps:

1. Fork the repository
2. Create a new branch (`git checkout -b feature-branch`)
3. Commit your changes (`git commit -m 'Add new feature'`)
4. Push to the branch (`git push origin feature-branch`)
5. Open a pull request


## 🙏 Acknowledgements

- **Kaggle:** For providing the dataset
- **TensorFlow and Keras:** For the deep learning frameworks
- **Streamlit:** For the web framework
- **OpenCV:** For image processing

---

Happy Coding! 🌟
