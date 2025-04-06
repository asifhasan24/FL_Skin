import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler

# ---------------------------
# Parameters and Paths
# ---------------------------
IMAGE_SIZE = 224
NUM_CLIENTS = 5
metadata_csv = "HAM10000_metadata.csv"
# Images are stored in two subfolders:
image_subfolders = ["HAM10000_images_part_1", "HAM10000_images_part_2"]

# ---------------------------
# Load the Metadata CSV
# ---------------------------
metadata = pd.read_csv(metadata_csv)
print("Original metadata shape:", metadata.shape)

# ---------------------------
# Map Diagnosis to Numeric Labels
# ---------------------------
label_map = {"nv": 4, "mel": 6, "bkl": 2, "bcc": 1, "vasc": 5, "akiec": 0, "df": 3}
metadata['label'] = metadata['dx'].map(label_map)
metadata = metadata.dropna(subset=['label'])
metadata['label'] = metadata['label'].astype(int)

# ---------------------------
# Visualize the Distribution of Classes
# ---------------------------
sns.countplot(x='dx', data=metadata)
plt.xlabel('Disease', size=12)
plt.ylabel('Frequency', size=16)
plt.title('Frequency Distribution of Classes', size=12)
plt.show()

# ---------------------------
# Load and Process Images
# ---------------------------
images = []
labels = []
for idx, row in metadata.iterrows():
    image_id = row['image_id']
    label = row['label']
    image_loaded = False
    for folder in image_subfolders:
        image_path = os.path.join(folder, image_id + ".jpg")
        if os.path.exists(image_path):
            try:
                img = load_img(image_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
                img_array = img_to_array(img)  # shape: (224,224,3)
                img_array = img_array.astype("float32") / 255.0
                images.append(img_array)
                labels.append(label)
                image_loaded = True
                break
            except Exception as e:
                print(f"Error loading image {image_path}: {e}")
    if not image_loaded:
        print(f"Could not load image for id {image_id} from any subfolder.")

images = np.array(images)
labels = np.array(labels)
print("Loaded images shape:", images.shape)
print("Loaded labels shape:", labels.shape)

# ---------------------------
# Oversample to Handle Class Imbalance
# ---------------------------
num_samples, h, w, c = images.shape
images_flat = images.reshape(num_samples, -1)
oversample = RandomOverSampler()
images_flat_res, labels_res = oversample.fit_resample(images_flat, labels)
print("After oversampling, data shape:", images_flat_res.shape)
images_res = images_flat_res.reshape(-1, IMAGE_SIZE, IMAGE_SIZE, c)

# ---------------------------
# Convert Labels to Categorical
# ---------------------------
num_classes = 7
labels_cat = to_categorical(labels_res, num_classes)
print("Labels converted to categorical with", num_classes, "classes.")

# ---------------------------
# Split into Training and Test Sets
# ---------------------------
X_train, X_test, Y_train, Y_test = train_test_split(images_res, labels_cat, test_size=0.2, random_state=1)
print(f"X_train shape: {X_train.shape}\nX_test shape: {X_test.shape}")
print(f"Y_train shape: {Y_train.shape}\nY_test shape: {Y_test.shape}")

# ---------------------------
# (Optional) Normalize the Data
# ---------------------------
X_train = (X_train - np.mean(X_train)) / np.std(X_train)
X_test  = (X_test  - np.mean(X_test)) / np.std(X_test)

# ---------------------------
# Split Training Data Among Clients
# ---------------------------
samples_per_client = X_train.shape[0] // NUM_CLIENTS
for client_id in range(1, NUM_CLIENTS + 1):
    start_idx = (client_id - 1) * samples_per_client
    end_idx = client_id * samples_per_client if client_id < NUM_CLIENTS else X_train.shape[0]
    X_client = X_train[start_idx:end_idx]
    Y_client = Y_train[start_idx:end_idx]
    np.savez(f'client_data_{client_id}.npz', X=X_client, Y=Y_client)
    print(f"Saved data for client {client_id}: {X_client.shape}")

# Optionally, save the test set for offline evaluation.
np.savez('server_test_data.npz', X=X_test, Y=Y_test)
print("Saved global test data for offline evaluation.")
