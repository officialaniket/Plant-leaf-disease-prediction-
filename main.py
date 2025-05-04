import streamlit as st
import tensorflow as tf
import numpy as np

# Tensorflow Model Prediction
def model_prediction(test_image):
    cnn = tf.keras.models.load_model(r"C:\\Users\\anike\\Downloads\\Plant_Disease_Prediction\\Plant_Disease_Prediction\\trained_plant_disease_model.keras")

    # Load image directly from the uploaded file
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to a batch
    predictions = cnn.predict(input_arr)

    return np.argmax(predictions)  # Return index of highest probability

# Function to get solution based on disease name
def get_disease_solution(disease_name):
    solutions = {
        "Tomato___healthy": "No treatment required. Maintain proper watering, fertilization, and monitor regularly for any new symptoms.",
        "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Remove and destroy infected plants immediately. Control whiteflies using insecticidal soap or neem oil. Use virus-resistant tomato varieties. Install yellow sticky traps.",
        "Potato___Early_blight": "Apply fungicides containing chlorothalonil or mancozeb. Avoid overhead watering. Rotate crops annually. Remove and destroy infected leaves.",
        "Tomato___Early_blight": "Apply fungicides containing chlorothalonil or mancozeb. Water at the soil level. Practice crop rotation and proper sanitation.",
        "Apple___Cedar_apple_rust": "Apply fungicides like myclobutanil or propiconazole early in the season. Remove nearby juniper trees. Prune infected branches. Use resistant apple varieties.",
        "Apple___Apple_scab": "Spray fungicides (captan, myclobutanil) at bud break. Rake and destroy fallen leaves. Ensure good air circulation. Choose scab-resistant varieties.",
        "Corn_(maize)___Common_rust_": "Use resistant corn varieties. Apply fungicides if infection is severe. Rotate crops. Destroy infected debris after harvest."
    }
    return solutions.get(disease_name, "Solution not available for this disease.")

# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Solution", "Disease Recognition"])

# Main Page
if app_mode == "Home":
    st.header("PLANT LEAF DISEASE PREDICTION SYSTEM")
    image_path = r"C:\Users\anike\Downloads\Plant_Disease_Prediction\Plant_Disease_Prediction\Plant Intro.jpeg"
    st.image(image_path, use_container_width=True)
    st.markdown("""
    Welcome to the Plant Leaf Disease Prediction System! 🌿🔍
    
    Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

    ### How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
    3. **Results:** View the results and recommendations for further action.

    ### Why Choose Us?
    - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

    ### Get Started
    Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!

    ### About Us
    Learn more about the project, our team, and our goals on the **About** page.
    """)

# About Project
elif app_mode == "About":
    st.header("About")
    st.markdown("""
    #### About Dataset
    This dataset is recreated using offline augmentation from the original dataset. The original dataset can be found on this GitHub repo.
    This dataset consists of about 87K RGB images of healthy and diseased crop leaves categorized into 38 different classes. The total dataset is divided into 80/20 ratio of training and validation set preserving the directory structure.
    A new directory containing 33 test images is created later for prediction purpose.

    #### Content
    1. train (70295 images)
    2. test (33 images)
    3. validation (17572 images)
    """)

# Prediction Page
elif app_mode == "Disease Recognition":
    st.header("Disease Recognition")
    test_image = st.file_uploader("Choose an Image:")
    if st.button("Show Image") and test_image is not None:
        st.image(test_image, use_container_width=True)


    # Predict button
    if st.button("Predict") and test_image is not None:
        st.snow()
        st.write("Our Prediction")
        result_index = model_prediction(test_image)
        
        # Reading Labels
        class_name = [
            'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
            'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
            'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
            'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
            'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
            'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
            'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
            'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
            'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
            'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
            'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
            'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
            'Tomato___healthy'
        ]

        predicted_disease = class_name[result_index]
        st.success(f"Model is Predicting it's a {predicted_disease}")

        # Show solution for predicted disease
        solution = get_disease_solution(predicted_disease)
        st.info(f"Solution: {solution}")

# Solution Page
elif app_mode == "Solution":
    st.header("Solution")
    st.markdown("""
    After detecting the disease, appropriate solutions and treatment methods are provided automatically
    based on the prediction results. You can check the 'Disease Recognition' page to see prediction and solutions together.
    """)
