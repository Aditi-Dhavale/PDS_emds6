from keras.models import load_model
import cv2
from PIL import Image
import numpy as np
import streamlit as st
import time  # Import for simulating the loading bar

# Setting page config without the theme argument
st.set_page_config(
    page_title="EMDS-6 Classifier",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🔬"  # A microscope emoji as the favicon
)

# Custom CSS and HTML
st.markdown(f"""
<style>
    /* Style for the main button */
    .stButton>button {{
        background-color: #E694FF;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 10px 15px;
        font-size: 16px;
    }}
    .stButton>button:hover {{
        background-color: #C464FF;
        color: white;
    }}

    /* Background image for the entire page */
    body, .main {{
        background-image: url('https://img.freepik.com/free-photo/3d-medical-background-with-virus-cells-dna-strand_1048-11512.jpg?w=1060&t=st=1698162991~exp=1698163591~hmac=1fc015d1beec5853e480ce885726115dd592d91ff49acf1ef82b8fd89b6c8c22');
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
        color: #E2E2E2;
    }}

    /* Making the main content and sidebar background transparent */
    div.main {{
        background: none;
    }}
    .block-container {{
        background: none;
        box-shadow: none;
    }}
    .sidebar .sidebar-content {{
        background-color: rgba(47, 44, 63, 0.6);  // Semi-transparent background
    }}

    #custom-title {{
        color: white;  
        font-size: 44px;  // Adjusted to 44px 
        font-weight: bold;
    }}

    #custom-upload-text {{
        color: white;
        font-size: 24px;  // Ensured it's 24px
    }}

</style>
""", unsafe_allow_html=True)

# Using HTML to customize specific text
st.markdown(f'<div id="custom-title">Deep Learning based EMDS-6 microorganism data classifier developed by PixelProphets</div>', unsafe_allow_html=True)
st.markdown('<div id="custom-upload-text">Upload the Image from any of 21 classes of microorganism.</div>', unsafe_allow_html=True)

# Loading the model and other operations
# Note: The path to the model is specific to your system. Update it accordingly.
saved_model1 = load_model(r'C:\Users\Abhishek Sinha\College\SEM 3\PDS\fip_mobilenet.hdf5')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "tiff", "bmp"])

if uploaded_file is not None:
    imge = Image.open(uploaded_file)
    st.image(imge, caption='Uploaded Image')

    if st.button('PREDICT'):
        # Simulate loading bar
        latest_iteration = st.empty()
        bar = st.progress(0)
        for i in range(100):
            # Update the progress bar with each iteration.
            latest_iteration.text(f'Predicting... {i+1}%')
            bar.progress(i + 1)
            time.sleep(0.01)

        Categories1 = ['Type 1', 'Type 2', 'Type 3', 'Type 4', 'Type 5', 'Type 6', 'Type 7', 'Type 8', 'Type 9', 'Type 10', 'Type 11', 'Type 12', 'Type 13',
                       'Type 14', 'Type 15', 'Type 16', 'Type 17', 'Type 18', 'Type 19', 'Type 20', 'Type 21']

        st.write('Result.....')
        flat_data = []
        imge = np.array(imge)
        img1 = cv2.resize(imge, (224, 224), interpolation=cv2.INTER_NEAREST)
        norm_image = cv2.normalize(img1, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        if norm_image.shape[2] == 4:
            norm_image = norm_image[:, :, :3]

        y = np.expand_dims(norm_image, axis=0)

        y_out1 = saved_model1.predict(y)
        y_out1 = np.round(y_out1)
        y_out1 = Categories1[y_out1.argmax()]

        st.title(f' PREDICTED OUTPUT: {y_out1}')

st.text("")
st.text('Made by Aditi Dhavale, Abhishek Sinha, Anshul Shinde, Amrut Ghadge')