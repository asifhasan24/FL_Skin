import sys
import numpy as np
import matplotlib.pyplot as plt
import shap
import tensorflow as tf

# Mapping dictionary for skin lesion types.
mapping = {
    0: 'Melanocytic nevi',
    1: 'Melanoma',
    2: 'Benign keratosis-like lesions',
    3: 'Basal cell carcinoma',
    4: 'Actinic keratoses',
    5: 'Vascular lesions',
    6: 'Dermatofibroma'
}

shap.initjs()

def main(client_id):
    # Load client's local data.
    data = np.load(f'client_data_{client_id}.npz')
    X_client = data["X"]
    Y_client = data["Y"]

    # Load the client's local model.
    local_model_filename = f'client_local_model_{client_id}.h5'
    try:
        model = tf.keras.models.load_model(local_model_filename)
        print(f"Client {client_id}: Loaded local model from '{local_model_filename}'.")
    except Exception as e:
        print(f"Client {client_id}: Failed to load local model. Error: {e}")
        return

    # Convert one-hot labels to indices.
    y_client_indices = np.argmax(Y_client, axis=1)
    unique_classes = np.unique(y_client_indices)

    # Select one sample per class.
    one_sample_per_class = {}
    for class_label in unique_classes:
        idx = np.where(y_client_indices == class_label)[0][0]
        one_sample_per_class[class_label] = (X_client[idx], mapping[class_label])

    for class_label, (image, true_label) in one_sample_per_class.items():
        pred_probs = model.predict(image.reshape((1, 224, 224, 3)))
        pred_label_idx = np.argmax(pred_probs)
        pred_label = mapping[pred_label_idx]

        plt.figure(figsize=(8, 8))
        plt.imshow(image)
        plt.title(f"True Label: {true_label}\nPredicted Label: {pred_label}")
        plt.axis('off')
        plt.show()

        masker = shap.maskers.Image("inpaint_telea", X_client[0].shape)
        explainer = shap.Explainer(model, masker, output_names=[mapping[i] for i in range(len(mapping))])
        shap_values = explainer(image.reshape((1, 224, 224, 3)))
        shap.image_plot(shap_values)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python client_explain.py <client_id>")
    else:
        client_id = sys.argv[1]
        main(client_id)
