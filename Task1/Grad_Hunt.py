import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

# Paths to data folders
TRAIN_DIR = "C:/Users/HP/Documents/Programs/Cynaptics/Data/Train"
TEST_DIR = "C:/Users/HP/Documents/Programs/Cynaptics/Data/Test"

# Function to prepare the training dataframe
def prepare_train_data(train_dir):
    image_paths = []
    labels = []
    for label in os.listdir(train_dir):
        for img_name in os.listdir(os.path.join(train_dir, label)):
            image_paths.append(os.path.join(train_dir, label, img_name))
            labels.append(label)
    return pd.DataFrame({'image_path': image_paths, 'label': labels})

# Function to prepare the training dataframe with additional images
def prepare_train_data(train_dir, additional_dir=None):
    image_paths = []
    labels = []
    for label in os.listdir(train_dir):
        for img_name in os.listdir(os.path.join(train_dir, label)):
            image_paths.append(os.path.join(train_dir, label, img_name))
            labels.append(label)
    
    # If additional_dir is provided, append its images to the dataset
    if additional_dir:
        for label in os.listdir(additional_dir):
            for img_name in os.listdir(os.path.join(additional_dir, label)):
                image_paths.append(os.path.join(additional_dir, label, img_name))
                labels.append(label)
    
    return pd.DataFrame({'image_path': image_paths, 'label': labels})

# Gradiant_Hunt2 Dataset
ADDITIONAL_DIR = "C:/Users/HP/Documents/Programs/Cynaptics/New_Data"
train_df = prepare_train_data(TRAIN_DIR, additional_dir=ADDITIONAL_DIR)

# Function to extract features (image loading and resizing)
def extract_features(image_paths, target_size=(236, 236)):
    features = []
    for image_path in tqdm(image_paths, desc="Processing Images"):
        try:
            img = load_img(image_path, target_size=target_size)
            img = np.array(img)
            features.append(img)
        except Exception as e:
            print(f"Error with file {image_path}: {e}")
    return np.array(features)

# Prepare training and test dataframes
train_df = prepare_train_data(TRAIN_DIR)
test_df = prepare_test_data(TEST_DIR)

# Extract features and normalize pixel values
train_features = extract_features(train_df['image_path'])
X_train = train_features / 255.0

# Label encoding for training labels
le = LabelEncoder()
y_train = le.fit_transform(train_df['label'])
y_train = to_categorical(y_train, num_classes=2)  # One-hot encoding for binary classification

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
datagen.fit(X_train)

# Build the CNN model
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(236, 236, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu', kernel_regularizer=l2(0.001)),  # L2 regularization
    Dropout(0.5),
    Dense(2, activation='softmax')
])

# Print model summary
model.summary()

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

from sklearn.model_selection import train_test_split

# Split the data into training and validation sets
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Train the model with data augmentation
history = model.fit(
    datagen.flow(X_train_split, y_train_split, batch_size=32),
    epochs=50,
    validation_data=(X_val_split, y_val_split),
    callbacks=[reduce_lr, early_stopping]
)

# Plot training & validation accuracy values
plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

def prepare_test_data(test_dir):
    image_paths = sorted(os.listdir(test_dir))  # Sort file names alphabetically
    full_paths = [os.path.join(test_dir, img_name) for img_name in image_paths]
    return pd.DataFrame({'image_path': full_paths})
  
test_df = prepare_test_data("C:/Users/HP/Documents/Programs/Cynaptics/Test_Images")

X_test = extract_features(test_df['image_path']) / 255.0  # Normalize Test dataset

predictions = model.predict(X_test)
predicted_labels = ["AI" if np.argmax(p) == 0 else "Real" for p in predictions]

# Save predictions to CSV
submission = pd.DataFrame({
    'Id': [os.path.basename(img) for img in test_df['image_path']],  # File names
    'Label': predicted_labels
})
submission.to_csv('submission.csv', index=False)

# Load the CSV file
file_path = "C:/Users/HP/Documents/Programs/Cynaptics/submission.csv"
data = pd.read_csv(file_path)

# Sort the data by the 'ID' column in ascending order
data_sorted = data.sort_values(by='Id', key=lambda x: x.str.extract('(\d+)', expand=False).astype(int))

# Save the sorted data to a new CSV file
sorted_file_path = "C:/Users/HP/Documents/Programs/Cynaptics/sorted_submission.csv"
data_sorted.to_csv(sorted_file_path, index=False)