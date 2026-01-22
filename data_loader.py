import pandas as pd
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Paths (update to match your folder)
train_csv = "datasets/train.csv"
train_img_dir = "datasets/train_images"

# Load CSV
df = pd.read_csv(train_csv)

# Add full path to images
df['id_code'] = df['id_code'].apply(lambda x: os.path.join(train_img_dir, x + ".png"))

# Convert labels to string
df['diagnosis'] = df['diagnosis'].astype(str)

# Split train/validation
train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['diagnosis'], random_state=42)

# Image generators
datagen = ImageDataGenerator(rescale=1./255,
                             rotation_range=20,
                             zoom_range=0.2,
                             horizontal_flip=True)

train_generator = datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col="id_code",
    y_col="diagnosis",
    target_size=(224,224),
    batch_size=32,
    class_mode="categorical"
)

val_generator = datagen.flow_from_dataframe(
    dataframe=val_df,
    x_col="id_code",
    y_col="diagnosis",
    target_size=(224,224),
    batch_size=32,
    class_mode="categorical"
)

print("Train samples:", train_generator.samples)
print("Validation samples:", val_generator.samples)