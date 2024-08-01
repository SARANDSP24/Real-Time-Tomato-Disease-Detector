# # #old code
# import streamlit as st
# import cv2
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.image import img_to_array

# # Load model
# model = load_model('best_model.keras')
# class_names = ['Tomato___Bacterial_spot','Tomato___Early_blight','Tomato___healthy','Tomato___Late_blight','Tomato___Leaf_Mold','Tomato___Septoria_leaf_spot','Tomato___Spider_mites Two-spotted_spider_mite','Tomato___Target_Spot','Tomato___Tomato_mosaic_virus','Tomato___Tomato_Yellow_Leaf_Curl_Virus']

# # Streamlit App
# st.title('Tomato Disease Detection Tool')

# st.sidebar.title('Settings')
# confidence_threshold = st.sidebar.slider('Confidence Threshold', 0.0, 1.0, 0.5)

# def predict(image):
#     image = cv2.resize(image, (150, 150))
#     image = img_to_array(image)
#     image = np.expand_dims(image, axis=0)
#     image = image / 255.0
#     predictions = model.predict(image)
#     score = np.max(predictions)
#     label = class_names[np.argmax(predictions)]
#     return label, score

# run = st.checkbox('Run')
# FRAME_WINDOW = st.image([])

# cap = cv2.VideoCapture(0)

# while run:
#     ret, frame = cap.read()
#     if not ret:
#         break
#     label, score = predict(frame)
#     if score > confidence_threshold:
#         cv2.putText(frame, f'{label}: {score:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
#     FRAME_WINDOW.image(frame)

# cap.release()
# cv2.destroyAllWindows()

#####
#####
#####
#####
#disease detection 2 modes:

import cv2
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load the trained model
model = load_model('best_model.keras')

# Define class labels and solutions for each class
class_labels = [
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___healthy',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus'
]

solutions = {
    'Tomato___Bacterial_spot': 'Use copper-based bactericides. Ensure proper plant spacing and avoid overhead watering.',
    'Tomato___Early_blight': 'Apply fungicides like chlorothalonil or mancozeb. Rotate crops and remove infected plants.',
    'Tomato___healthy': 'No action needed. Your plant is healthy.',
    'Tomato___Late_blight': 'Apply fungicides like metalaxyl. Remove and destroy infected plants.',
    'Tomato___Leaf_Mold': 'Ensure good air circulation. Apply fungicides like chlorothalonil.',
    'Tomato___Septoria_leaf_spot': 'Use fungicides like chlorothalonil. Remove and destroy infected leaves.',
    'Tomato___Spider_mites Two-spotted_spider_mite': 'Use miticides or insecticidal soap. Ensure proper watering and humidity.',
    'Tomato___Target_Spot': 'Apply fungicides like mancozeb or copper-based products. Remove infected leaves.',
    'Tomato___Tomato_mosaic_virus': 'Remove and destroy infected plants. Disinfect tools and avoid smoking around plants.',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': 'Use insecticides to control whiteflies. Remove infected plants.'
}

def predict_and_display(image):
    # Preprocess the image
    img = cv2.resize(image, (150, 150))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0

    # Predict the class
    predictions = model.predict(img)
    predicted_class = np.argmax(predictions, axis=1)[0]
    predicted_label = class_labels[predicted_class]
    solution = solutions[predicted_label]

    return predicted_label, solution

st.title('Plant Disease Detection Tool')

# Add a sidebar for image upload
st.sidebar.title('Upload an Image')
uploaded_file = st.sidebar.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Load and display the uploaded image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    st.image(image, channels="BGR", caption='Uploaded Image.', use_column_width=True)

    # Predict and display the results
    label, solution = predict_and_display(image)
    st.success(f'Predicted Class: {label}')
    st.info(f'Solution: {solution}')

# OpenCV feed
option = st.selectbox(
    'Choose an option',
    ('Realtime Detection', 'Upload Image')
)

if option == 'Realtime Detection':
    st.title("Real-Time Detection")
    st.write("Click 'Start Detection' to begin the camera feed.")

    start_detection = st.button('Start Detection', key='start_button')

    if start_detection:
        stframe = st.empty()
        cap = cv2.VideoCapture(0)

        stop_detection = st.button('Stop Detection', key='stop_button')

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.write("Failed to capture image")
                break
            # Predict and display the class on the frame
            label, solution = predict_and_display(frame)
            cv2.putText(frame, f'Class: {label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            stframe.image(frame, channels="BGR", use_column_width=True)
            
            if stop_detection:
                cap.release()
                break

        cap.release()
        cv2.destroyAllWindows()




