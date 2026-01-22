import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Paths
train_csv = "datasets/train.csv"
train_img_dir = "datasets/train_images"

# Load CSV
df = pd.read_csv(train_csv)
df['id_code'] = df['id_code'].apply(lambda x: os.path.join(train_img_dir, x + ".png"))
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

# Build MobileNetV2 model
base_model = MobileNetV2(input_shape=(224,224,3), include_top=False, weights='imagenet')
base_model.trainable = False  # freeze base layers

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(5, activation='softmax')  # 5 DR classes
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train Phase 1
history = model.fit(train_generator, validation_data=val_generator, epochs=5)

# Fine-tune Phase 2
base_model.trainable = True
model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history_ft = model.fit(train_generator, validation_data=val_generator, epochs=5)

# Save model
model.save("dr_mobilenetv2.h5")

# Plot accuracy
plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()