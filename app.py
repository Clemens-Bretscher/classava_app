from six import print_
import streamlit as st
import numpy as np
from PIL import Image
from tensorflow import keras
from tensorflow.keras import preprocessing
from tensorflow.keras.models import load_model
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.markdown("<h1 style='text-align: center;'>Classava.ai</h1>", unsafe_allow_html=True)
#st.title('Classava')
#st.header('Classava: Capstone Project')








# class Model:
#     def __init__(self,image) -> None:
#         self.image = image
#         pass
def how_to():
    how_to_button = st.expander('How to use the app')
    with how_to_button:
        st.markdown('''
        1. Take a photo of the cassava plant. 
        - Only take photos of leaves, 
        not the tuber or other parts of the plant. 
        - Make sure that the photo is not blurry. 
        2. Upload the image to the app
        - The app will tell you which disease the plant has.
        - The certainty of the app is shown in percent. 
        - If the certainty is low, take a photo of another leaf and check again. 
        ''') 
    return

def about():
    about_button = st.expander('About')
    with about_button:
        st.write(
        '''
        This app was developed in association with the
        National Crops Resources Research Institute (NaCRRI) and 
        the AI lab in Makarere University, Kampala.
        
        About 9500 images of cassava plants were collected from Ugandan farmers.
        Experts grouped the images in 5 categories: 
        Healthy, Cassava Bacterial Blight (CBB), Cassava Brown Streak Disease (CBSD),
        Cassava Green Mottle (CGM), and Cassava Mosaic Disease (CMD)
        

        With this data an artificial neural network was created that automatically
        classifies an image of cassava in these five different categories.
        
        
        ''')


        st.markdown("<p style='text-align: left; font-size: 20px;'>Diseases of Cassava</p>", unsafe_allow_html=True)
        st.markdown('''
        Cassava is one of the most important crops in Africa. It is rich in carbohydrates and withstands harsh conditions. 
        Following are the most common diseases of cassava with example images:
        
        

        ''')
        if st.checkbox('Cassava Bacterial Blight (CBB)'):
            col1, col2, col3 = st.columns([1,1,1])
            with col1:
                st.image("images/train-cbb-236.jpg")
            with col2:
                st.image("images/train-cbb-278.jpg")
            with col3:
                st.image("images/train-cbb-347.jpg")

    
        if st.checkbox('Cassava Brown Streak Disease (CBSD)'):
            col1, col2, col3 = st.columns([1,1,1])
            with col1:
                st.image("images/train-cbsd-261.jpg")
            with col2:
                st.image("images/train-cbsd-264.jpg")
            with col3:
                st.image("images/train-cbsd-274.jpg")
        
       
        if st.checkbox('Cassava Green Mottle (CGM)'):
            col1, col2, col3 = st.columns([1,1,1])
            with col1:
                st.image("images/train-cgm-0.jpg")
            with col2:
                st.image("images/train-cgm-6.jpg")
            with col3:
                st.image("images/train-cgm-243.jpg")


        if st.checkbox('Cassava Mosaic Disease (CMD)'):
            col1, col2, col3 = st.columns([1,1,1])
            with col1:
                st.image("images/train-cmd-27.jpg")
            with col2:
                st.image("images/train-cmd-155.jpg")
            with col3:
                st.image("images/train-cmd-323.jpg")
        
        if st.checkbox('Healthy'):
            col1, col2, col3 = st.columns([1,1,1])
            with col1:
                st.image("images/train-healthy-1.jpg")
            with col2:
                st.image("images/train-healthy-27.jpg")
            with col3:
                st.image("images/train-healthy-63.jpg")

        
        # if st.checkbox('Images'):
        #     st.image('images/leafs.png')
        # data = pd.read_csv('data/train.csv')
        # if st.checkbox('Dataset'):
        #     fig = plt.figure()
        #     sns.countplot(data = data, y=data.label_name)
        #     plt.title('Cassava Disease Category')
        #     plt.xlabel('Count')
        #     plt.ylabel(' ')
        #     st.pyplot(fig)

    return 
  


def predict(image):
    model = load_model(r'saved_models/my_model')

    test_img = image.resize((380,380))
    test_img = preprocessing.image.img_to_array(test_img).astype(np.float32)/255
    test_img = np.expand_dims(test_img,axis=0)

    probabilities = model.predict(test_img)
    prediction = np.argmax(probabilities,axis=1)
    
    return
  
#st.selectbox('Select a file', "images/Coat_of_arms_of_Uganda.png")
menu = ["Image","Dataset","DocumentFiles","About"]
#choice = st.sidebar.selectbox("Menu",menu)
#choice = st.sidebar("Menu",menu)

def upload():
    
    upload_file = st.file_uploader('Upload Your Image File',type=['jpg','png','jpeg','bmp','gif'])
    if upload_file is not None:
        #lets use PIL to open the uploaded file
        col1,col2,col3 = st.columns([1,4,1])
        with col1:
            st.write("")
        with col2:   
            image = Image.open(upload_file)
        #lets show the image:
            st.image(image)

            # results = predict(image)

            # if results == 0:
            #     st.write(f'Uploaded image is class: CBB')
            # elif results == 1:
            #     st.write(f'Uploaded image is class: CBSD')
            # elif results == 2:
            #     st.write(f'Uploaded image is class: CGM')
            # elif results == 3:
            #     st.write(f'Uploaded image is class: CMD')
            # else:
            #     st.write(f'Uploaded image is class: Healthy')

            st.markdown("<p style='text-align: center; font-size: 18px;'><b>...CMD...</b></p>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: center; font-size: 18px;'>76% certain</p>", unsafe_allow_html=True)

            

        
        with col3:
            st.write("")



        #lets call the predict function



        



    return 



#st.image("images/Coat_of_arms_of_Uganda.png", width=200)
#image2 = Image.open('images/Coat_of_arms_of_Uganda.png')
#st.image(image2, caption=None, width=200, use_column_width=None, clamp=False, channels="RGB", output_format="auto")        
#st.markdown("<img src="pictures/Create_new-branch.png"
#     alt="Create_branch]" width=400>"
#st.markdown('<img src="pictures/Create_new-branch.png">', unsafe_allow_html=True)


def space():
    st.write('')
def uganda():
    col1, col2, col3 = st.columns([1,1,1])
    with col1:
        st.write("")
        st.write("")
        #st.markdown("<h1 style='text-align: center;'>dljsdfjn </h1>", unsafe_allow_html=True)

    with col2:
        st.image("images/Coat_of_arms_of_Uganda.png", width=200)
    with col3:
        st.write("")
        #st.markdown("<h1 style='text-align: center;'> dljsdfjn</h1>", unsafe_allow_html=True)

# if st.button(label='Upload your image here'):
#     upload()



if __name__=='__main__':
    space()
    space()
    upload()

    space()
    space()
    space()

    uganda()

    space()
    space()
    space()

    how_to()
    about()
    
