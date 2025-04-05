import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from tensorflow.keras.utils import to_categorical
import tensorflow as tf

# ---------------------------
# Load the CSV Data
# ---------------------------
csv_path = "skin-cancer-mnist-ham10000/hmnist_28_28_RGB.csv"
data = pd.read_csv(csv_path)
print("Original data shape:", data.shape)  # (10015, 2353)

# ---------------------------
# Define Classes for Reference
# ---------------------------
# This dictionary maps numerical labels to class abbreviations and full names.
classes = {
    4: ('nv',' melanocytic nevi'),
    6: ('mel','melanoma'),
    2: ('bkl','benign keratosis-like lesions'),
    1: ('bcc',' basal cell carcinoma'),
    5: ('vasc',' pyogenic granulomas and hemorrhage'),
    0: ('akiec','Actinic keratoses and intraepithelial carcinomae'),
    3: ('df','dermatofibroma')
}

# Visualize the distribution of classes.
# Here we assume that the CSV originally used a different column name for plotting (e.g., "dx")
# If your CSV does not include such a column, you can skip this step.
# sns.countplot(x='dx', data=data)
# plt.xlabel('Disease', size=12)
# plt.ylabel('Frequency', size=16)
# plt.title('Frequency Distribution of Classes', size=12)
# plt.show()

# ---------------------------
# Split Data and Labels
# ---------------------------
y = data['label']
x = data.drop(columns=['label'])

# ---------------------------
# Handle Imbalanced Dataset with Oversampling
# ---------------------------
oversample = RandomOverSampler()
x_res, y_res = oversample.fit_resample(x, y)
print("After oversampling, data shape:", x_res.shape)

# ---------------------------
# Convert Pixels to Float and Reshape to Images
# ---------------------------
# The CSV contains 2352 pixel values per sample (28*28*3)
# First, convert to float and scale the pixel values to [0,1]
x_res = x_res.astype("float32") / 255.0
# Reshape to (num_samples, 28, 28, 3)
images = np.array(x_res).reshape(-1, 28, 28, 3)
print("Shape after reshaping to 28x28:", images.shape)

# ---------------------------
# Convert Labels to Categorical
# ---------------------------
num_classes = len(np.unique(y_res))
labels_cat = to_categorical(y_res, num_classes)
print("Converted labels to categorical with", num_classes, "classes.")

# ---------------------------
# Split into Training and Test Sets
# ---------------------------
X_train, X_test, Y_train, Y_test = train_test_split(images, labels_cat, test_size=0.2, random_state=1)
print(f"X_train shape: {X_train.shape}\nX_test shape: {X_test.shape}")
print(f"Y_train shape: {Y_train.shape}\nY_test shape: {Y_test.shape}")

# ---------------------------
# Upscale Images to 224x224 and Normalize
# ---------------------------
IMAGE_SIZE = 224
# Upscale training and test images
X_train_resized = tf.image.resize(X_train, (IMAGE_SIZE, IMAGE_SIZE)).numpy()
X_test_resized  = tf.image.resize(X_test,  (IMAGE_SIZE, IMAGE_SIZE)).numpy()
print("After resizing:")
print("X_train_resized shape:", X_train_resized.shape)
print("X_test_resized shape:", X_test_resized.shape)

# Normalize: (x - mean) / std
X_train_resized = (X_train_resized - np.mean(X_train_resized)) / np.std(X_train_resized)
X_test_resized  = (X_test_resized - np.mean(X_test_resized)) / np.std(X_test_resized)

# ---------------------------
# Save Data for Federated Learning
# ---------------------------
# Split the training data among 5 clients
NUM_CLIENTS = 5
samples_per_client = X_train_resized.shape[0] // NUM_CLIENTS

for client_id in range(1, NUM_CLIENTS + 1):
    start_idx = (client_id - 1) * samples_per_client
    end_idx = client_id * samples_per_client if client_id < NUM_CLIENTS else X_train_resized.shape[0]
    X_client = X_train_resized[start_idx:end_idx]
    Y_client = Y_train[start_idx:end_idx]
    np.savez(f'client_data_{client_id}.npz', X=X_client, Y=Y_client)
    print(f"Saved data for client {client_id}: {X_client.shape}")

# Save the test set for the server
np.savez('server_test_data.npz', X=X_test_resized, Y=Y_test)
print("Saved global test data for the server.")
