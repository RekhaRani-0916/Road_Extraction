import streamlit as st
import torch
import tensorflow as tf
import pandas as pd
import os
import cv2
import albumentations as album
import numpy as np
from PIL import Image
import segmentation_models_pytorch as smp

H = 256
W = 256
select_classes = ['background', 'road']
class_dict = pd.read_csv("class_dict.csv")
class_names = class_dict['name'].tolist()
class_rgb_values = class_dict[['r','g','b']].values.tolist()
select_class_indices = [class_names.index(cls.lower()) for cls in select_classes]
select_class_rgb_values =  np.array(class_rgb_values)[select_class_indices]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_deeplab_model():
    ENCODER = 'resnet50'
    ENCODER_WEIGHTS = 'imagenet'

    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
    best_model = torch.load('best_model.pth', map_location=DEVICE)
    print('Loaded DeepLabV3+ model from this run.')
    return best_model, preprocessing_fn

def get_preprocessing(preprocessing_fn=None):
    _transform = []
    if preprocessing_fn:
        _transform.append(album.Lambda(image=preprocessing_fn))
    _transform.append(album.Lambda(image=to_tensor, mask=to_tensor))
    return album.Compose(_transform)

def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')

def colour_code_segmentation(image, label_values):
    colour_codes = np.array(label_values)
    x = colour_codes[image.astype(int)]
    return x

def reverse_one_hot(image):
    x = np.argmax(image, axis=-1)
    return x

def iou(y_true, y_pred):
    def f(y_true, y_pred):
        intersection = (y_true * y_pred).sum()
        union = y_true.sum() + y_pred.sum() - intersection
        
        x = (intersection + 1e-15) / (union + 1e-15)
        x = x.astype(np.float32)
        return x
    return tf.numpy_function(f, [y_true, y_pred], tf.float32)

smooth = 1e-15
def dice_coef(y_true, y_pred):
    y_true = tf.keras.layers.Flatten()(y_true)
    y_pred = tf.keras.layers.Flatten()(y_pred)
    
    intersection = tf.reduce_sum(y_true*y_pred)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred))


def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)

def load_unet_model():
    model = tf.keras.models.load_model('unet_model.h5', custom_objects={'iou': iou, 'dice_coef': dice_coef, 'dice_loss': dice_loss})
    print("unet model loaded")
    return model

def read_image(path):
    img = Image.open(path)
    img = img.resize((W, H))
    x = np.array(img, dtype=np.float32)
    x = x / 255.0
    return x

def main():
    st.title("Road Extraction")
    model_selection = st.radio("Select Model:", ["DeepLabV3+", "UNet"])
    
    if model_selection == "DeepLabV3+":
        st.header("DeepLabV3+ Model")
        best_model, preprocessing_fn = load_deeplab_model()

    elif model_selection == "UNet":
        st.header("UNet Model")
        unet_model = load_unet_model()
        
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        st.success("Image uploaded successfully")
        if model_selection == "DeepLabV3+":
            with st.spinner("Processing..."):
                image = cv2.cvtColor(np.array(Image.open(uploaded_file)), cv2.COLOR_BGR2RGB)
                preprocess = get_preprocessing(preprocessing_fn)
                transformed_image = preprocess(image=image)["image"]
                # print("hi")
                x_tensor = torch.from_numpy(transformed_image).to(DEVICE).unsqueeze(0)
                pred_mask = best_model(x_tensor)
                pred_mask = pred_mask.detach().squeeze().cpu().numpy()
                pred_mask = np.transpose(pred_mask, (1, 2, 0))
                pred_mask = colour_code_segmentation(reverse_one_hot(pred_mask), select_class_rgb_values)
                # print("hi2")
                st.image(
                    [image, pred_mask],
                    caption=["Original Image", "Predicted Mask"],
                    use_column_width=True,
                )
            
        elif model_selection == "UNet":
            with st.spinner("Processing..."):
                image = read_image(uploaded_file)

                image1 = np.expand_dims(image, axis=0)
                pred = unet_model.predict(image1)
                st.image(
                        [image, pred[0,...]],
                        caption=["Original Image", "Predicted Mask"],
                        use_column_width=True,
                    )

if __name__ == "__main__":
    main()
