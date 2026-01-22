import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("dr_mobilenetv2.h5")

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    conv_output, predictions = grad_model(img_array)
    if pred_index is None:
        pred_index = tf.argmax(predictions[0])
    grads = tf.gradients(predictions[:, pred_index], conv_output)[0]
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))
    conv_output = conv_output[0]
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
    return heatmap.numpy()

# Example usage
img_path = "datasets/train_images/000c1434d8d7.png"  # replace with any image from your dataset
img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224,224))
img_array = tf.keras.preprocessing.image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0

heatmap = make_gradcam_heatmap(img_array, model, "Conv_1")  # last conv layer in MobileNetV2
plt.matshow(heatmap)
plt.show()