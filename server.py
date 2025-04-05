import os
import time
import pickle
import numpy as np
import tensorflow as tf
import tenseal as ts

# Global server configuration
NUM_CLIENTS = 5
EPOCHS = 50
UPDATE_FILENAME_TEMPLATE = "client_update_{}.pkl"

# ---------------------------
# Tenseal Helper Functions
# ---------------------------
def create_tenseal_context():
    poly_modulus_degree = 16384  # Adjust as needed
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=poly_modulus_degree,
        coeff_mod_bit_sizes=[60, 40, 40, 40, 40, 60]
    )
    context.global_scale = 2**40
    context.generate_galois_keys()
    context.generate_relin_keys()
    return context, poly_modulus_degree

def decrypt_weights(context, encrypted_weights, shapes):
    decrypted_weights = []
    for enc_layer, shape in zip(encrypted_weights, shapes):
        if isinstance(enc_layer, list):  # Handle layers stored in chunks
            decrypted_chunks = []
            for chunk in enc_layer:
                decrypted_chunks.extend(chunk.decrypt())
            decrypted_array = np.array(decrypted_chunks[:np.prod(shape)])
        else:
            decrypted_array = np.array(enc_layer.decrypt())[:np.prod(shape)]
        decrypted_weights.append(decrypted_array.reshape(shape))
    return decrypted_weights

def aggregate_encrypted_weights(encrypted_updates, poly_modulus_degree):
    aggregated = []
    num_clients = len(encrypted_updates)
    for layer_idx in range(len(encrypted_updates[0])):
        if isinstance(encrypted_updates[0][layer_idx], list):
            aggregated_chunks = []
            num_chunks = len(encrypted_updates[0][layer_idx])
            for chunk_idx in range(num_chunks):
                agg_chunk = encrypted_updates[0][layer_idx][chunk_idx]
                for client_idx in range(1, num_clients):
                    agg_chunk += encrypted_updates[client_idx][layer_idx][chunk_idx]
                aggregated_chunks.append(agg_chunk.mul(1.0 / num_clients))
            aggregated.append(aggregated_chunks)
        else:
            agg_layer = encrypted_updates[0][layer_idx]
            for client_idx in range(1, num_clients):
                agg_layer += encrypted_updates[client_idx][layer_idx]
            aggregated.append(agg_layer.mul(1.0 / num_clients))
    return aggregated

# ---------------------------
# Global Model Creation Function
# ---------------------------
def create_global_model():
    from tensorflow.keras import layers, models, regularizers
    model = models.Sequential()
    model.add(layers.Input(shape=(224, 224, 3)))
    
    # Block 1
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal'))
    model.add(layers.MaxPooling2D())
    model.add(layers.BatchNormalization())
    
    # Block 2
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal'))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal'))
    model.add(layers.MaxPooling2D())
    model.add(layers.BatchNormalization())
    
    # Block 3
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal'))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal'))
    model.add(layers.MaxPooling2D())
    model.add(layers.BatchNormalization())
    
    # Block 4
    model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal'))
    model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal'))
    model.add(layers.MaxPooling2D())
    
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(256, activation='relu', kernel_initializer='he_normal'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(128, activation='relu', kernel_initializer='he_normal'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(64, activation='relu', kernel_initializer='he_normal'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(32, activation='relu', kernel_initializer='he_normal',
                           kernel_regularizer=regularizers.L1L2()))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(7, activation='softmax', kernel_initializer='glorot_uniform', name='classifier'))
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# ---------------------------
# Main Server Code
# ---------------------------
def main():
    # Load or initialize the global model.
    try:
        model = tf.keras.models.load_model('global_model.h5')
        print("Server: Loaded global model from 'global_model.h5'.")
    except Exception as e:
        print("Server: Global model not found. Creating a new model.")
        model = create_global_model()
        model.save('global_model.h5')
        print("Server: New global model created and saved as 'global_model.h5'.")

    # Create Tenseal context.
    tenseal_context, poly_modulus_degree = create_tenseal_context()

    for epoch in range(EPOCHS):
        print(f"\nServer: Starting epoch {epoch + 1}/{EPOCHS}")
        encrypted_updates = []

        # Wait until updates from all clients are available.
        while True:
            update_files = [UPDATE_FILENAME_TEMPLATE.format(i) for i in range(1, NUM_CLIENTS + 1)]
            if all(os.path.exists(f) for f in update_files):
                print("Server: Received all client updates.")
                break
            else:
                print("Server: Waiting for client updates...")
                time.sleep(5)

        # Load and remove each client’s update.
        for client_id in range(1, NUM_CLIENTS + 1):
            with open(UPDATE_FILENAME_TEMPLATE.format(client_id), "rb") as f:
                update = pickle.load(f)
                encrypted_updates.append(update)
            os.remove(UPDATE_FILENAME_TEMPLATE.format(client_id))
            print(f"Server: Loaded and removed update from client {client_id}.")

        # Aggregate the encrypted updates.
        aggregated_encrypted_weights = aggregate_encrypted_weights(encrypted_updates, poly_modulus_degree)

        # Decrypt aggregated weights and update the global model.
        model_shapes = [w.shape for w in model.get_weights()]
        new_weights = decrypt_weights(tenseal_context, aggregated_encrypted_weights, model_shapes)
        model.set_weights(new_weights)
        print(f"Server: Updated global model for epoch {epoch + 1}.")

        # Save updated global model for clients to load next.
        model.save("global_model.h5")
        print("Server: Global model saved as 'global_model.h5'.")

    print("Server: Federated training completed.")

if __name__ == "__main__":
    main()
