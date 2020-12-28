import numpy as np
import streamlit as st

def image_classifier(selected_model, loaded_image):
    # Which model did you choose?
    if selected_model == "VGG16":
        from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
        from keras.preprocessing import image
        classifier = VGG16()
    else:
        from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
        from keras.preprocessing import image
        classifier = ResNet50()

    new_image = loaded_image.resize((224, 224)) # You should make the size to the expected size (224, 224)

    transformed_image = image.img_to_array(new_image)
    transformed_image = np.expand_dims(transformed_image, axis=0)
    transformed_image = preprocess_input(transformed_image)

    y_pred = classifier.predict(transformed_image)
    decode_predictions(y_pred, top=5)
    label = decode_predictions(y_pred)
    # Retrieve the most likely result, e.g. highest probability
    decoded_label = label[0][0]
    # The result (prediction) from the classifier
    st.write("This is a %s (%.2f%%)" % (decoded_label[1], decoded_label[2]*100))
