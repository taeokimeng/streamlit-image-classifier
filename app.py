"""Main module for the streamlit app"""
import streamlit as st
import src.image_upload
import src.image_classification

def main():
    """Main function of the App"""
    st.title("Image Classifier")
    st.sidebar.title("Classifiers")
    selection = st.sidebar.radio("Please choose a model", ("VGG16", "ResNet50"))

    if selection == "VGG16":
        image_pil = src.image_upload.upload_image()
        if image_pil is not None:
            src.image_classification.image_classifier(selection, image_pil)
    else: # ResNet50
        image_pil = src.image_upload.upload_image()
        if image_pil is not None:
            src.image_classification.image_classifier(selection, image_pil)

if __name__ == '__main__':
    main()

