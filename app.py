"""Main module for the streamlit app"""
import streamlit as st
import src.image_upload
import src.image_classification

res_mode = {1: "Classifier", 2: "Comparison"}
res_model = {1: "VGG16", 2: "ResNet50"}

def main():
    """Main function of the App"""
    st.title("Image Classifier")
    st.sidebar.title("Classifiers")
    mode_selection = st.sidebar.radio("Please choose a mode", (res_mode[1], res_mode[2]))

    if mode_selection == res_mode[1]:
        model_selection = st.sidebar.selectbox("Please choose a model", (res_model[1], res_model[2]))
        # Upload an image and run a classifier
        image_pil = src.image_upload.upload_image()
        if image_pil is not None:
            src.image_classification.image_classifier(model_selection, image_pil)
    if mode_selection == res_mode[2]:
        model_selection = st.sidebar.multiselect("Please choose two models", [res_model[1], res_model[2]])
        # Upload an image and run a classifier
        image_pil = src.image_upload.upload_image()
        for model in model_selection:
            if image_pil is not None:
                src.image_classification.image_classifier(model, image_pil)

if __name__ == '__main__':
    main()

