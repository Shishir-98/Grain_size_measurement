import os
import cv2
import numpy as np
from skimage import measure
import matplotlib.pyplot as plt
import urllib.request
import streamlit as st
import warnings
import GrainCalculations as gc
warnings.filterwarnings('ignore')



os.makedirs("uploaded_images", exist_ok=True)
st.title('Grain Size Measurement of Micro Structure')
upload_file = st.file_uploader("Upload an image", type=['png', 'jpg', 'jpeg'])
col1, col2 = st.columns(2)
with col1:
    user_input = st.text_input('Magnification')
with col2:
    drop_down = st.selectbox("Select the Model", ['Intercept','Plainmetric'])

if upload_file is not None:
    image_path = os.path.join("uploaded_images", upload_file.name)

    with open(image_path, "wb") as f:
        f.write(upload_file.getbuffer())
    imagepass = cv2.imread(image_path)
    
    if drop_down == 'Plainmetric':
        originalImage, ImageWithCircle, marked_image, num_grains = gc.callPlanimetric(imagepass)
        plt.imshow(cv2.cvtColor(originalImage, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.title("Input Micrograph")
        st.pyplot(plt)
        plt.imshow(cv2.cvtColor(ImageWithCircle, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.title("Test Pattren")
        st.pyplot(plt)
        plt.imshow(cv2.cvtColor(marked_image, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.title("Analysed Output")
        st.pyplot(plt)
        st.write(f"Number of grains: {num_grains}")
        
    elif drop_down == "Intercept":
        originalImage, intersections, image_withCircles, output_image = gc.callIntercept(imagepass)
        plt.imshow(cv2.cvtColor(originalImage, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.title("Input Micrograph")
        st.pyplot(plt)
        plt.imshow(cv2.cvtColor(image_withCircles, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.title("Test Pattren")
        st.pyplot(plt)
        plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
        plt.title("Analysed Output")
        st.pyplot(plt)
        st.write(f"Intersections for 1st circle(Inner most) : {intersections[0]}")
        st.write(f"Intersections for 2nd circle(Middle) : {intersections[1]}")
        st.write(f"Intersections for 3rd circle(Outer most) : {intersections[2]}")
        st.write(f"Total intersections: {intersections[0]} + {intersections[1]} + {intersections[2]} = {sum(intersections)}")
         
else:
    st.write(" ")
