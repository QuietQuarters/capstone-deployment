
#Import of necessary libraries
import streamlit as st
import tensorflow as tf
import urllib.request
import numpy as np
import base64
from tensorflow.keras.models import load_model 
from tensorflow.keras.preprocessing import image 
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import cv2
############################################################################

#Functions use for preprocessing of images

def hist_equalization_all(image):
    #Converting image from BGR to RGB color space
    RGB_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    #Split the RGB image into individual channels
    r_channel, g_channel, b_channel = cv2.split(RGB_image)

    #Apply histogram equalization to each channel
    equalized_r_channel = cv2.equalizeHist(r_channel)
    equalized_g_channel = cv2.equalizeHist(g_channel)
    equalized_b_channel = cv2.equalizeHist(b_channel)

    #Merge the equalized channels back into an RGB image
    equalized_image = cv2.merge((equalized_r_channel, equalized_g_channel, equalized_b_channel))
    
    #Converting the RGB image back to BGR for the function, contrast_show()
    equalized_image = cv2.cvtColor(equalized_image, cv2.COLOR_RGB2BGR)
    
    return equalized_image

def increase_yellow_saturation_all(image):
    #Converting the image from BGR to HSV color space
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    #Define the lower and upper thresholds for yellow color in HSV
    lower_yellow = np.array([20, 100, 100], dtype=np.uint8)
    upper_yellow = np.array([40, 255, 255], dtype=np.uint8)

    #Creating a mask for yellow pixels
    yellow_mask = cv2.inRange(image_hsv, lower_yellow, upper_yellow)

    #Split the HSV image into individual channels
    h, s, v = cv2.split(image_hsv)

    #Increase the saturation of yellow pixels
    s_yellow = s[yellow_mask > 0]
    s_yellow = np.clip(s_yellow + 255, 0, 255)
    s[yellow_mask > 0] = s_yellow

    #Merging the modified channels back into the HSV image
    image_hsv = cv2.merge((h, s, v))

    #Convert the modified HSV image to BGR
    modified_image = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2BGR)
   
    return modified_image
############################################################################

#Add background image to streamlit app
@st.experimental_memo
def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()


img = get_img_as_base64("image.jpg")

page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url("https://github.com/QuietQuarters/capstone-deployment/blob/main/theming-showcase-main/streamlit_backgrd.jpg");
background-size: 180%;
background-position: top left;
background-repeat: no-repeat;
background-attachment: local;
}}

[data-testid="stHeader"] {{
background: rgba(0,0,0,0);
}}

[data-testid="stToolbar"] {{
right: 2rem;
}}
</style>
"""

#Setting title of the streamlit app
st.title('Diseased Vegetable Image Classifier')

#Loading of optimal model trained
@st.cache_resource #to store the loaded model in cache for faster app run time 
def model_load():
    model_url = 'https://github.com/QuietQuarters/capstone-deployment/raw/main/theming-showcase-main/MobileNetV2_model/model.19.h5'
    model_path = 'model.19.h5'
    urllib.request.urlretrieve(model_url, model_path)
    model = tf.keras.models.load_model(model_path)
    return model

model=model_load()

#Extracting uploaded file
file = st.file_uploader("For single images only. Note: Refrain from loading images with presence of water droplets for better results.", type=["jpg", "png", "jpeg"])
    
if file is None:
    st.write(f'<p style="font-size:26px;color:black;">Upload an image above to start!</p>', unsafe_allow_html=True)
else:
    # Convert uploaded file to bytes in desired data type format
    file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)

    # Decode image using OpenCV
    test_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    image_view = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)

    #Display of picture
    st.image(image_view)

    #Preprocessing methods
    test_image = hist_equalization_all(test_image)
    test_image = increase_yellow_saturation_all(test_image)

    #Formatting images to fitting arrays and mobilenetv2 weights
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    test_image = preprocess_input(test_image)

    #Gathering predictions
    result = model.predict(test_image)
    
    for pred in result:
        if pred[0] > 0.5:
            text = 'Vegetable is classified as'
            class01 = 'healthy'
            st.write(f'<span style="font-size:26px;">{text}</span> <span style="font-size:26px;color:lightgreen;">{class01}</span>', 
                     unsafe_allow_html=True)
        else:
            text02 = 'Vegetable might be'
            class02 = 'diseased'
            text03 = 'Follow up with personal check.'
            st.write(f'<span style="font-size:26px;">{text02}</span> <span style="font-size:26px;color:red;">{class02}</span>', unsafe_allow_html=True)
            st.write(f'<span style="font-size:26px;">{text03}</span>', unsafe_allow_html=True)
