import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import tenseal as ts
import pickle

# Global client configuration
BATCH_SIZE = 20
LOCAL_EPOCHS = 1
LEARNING_RATE = 0.0001

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

def encrypt_weights(context, weights, poly_modulus_degree):
    encrypted_weights = []
    for layer in weights:
        flat_weights = layer.flatten().tolist()
        max_size = poly_modulus_degree // 2 - 1
        if len(flat_weights) > max_size:
            chunks = [flat_weights[i:i + max_size] for i in range(0, len(flat_weights), max_size)]
            encrypted_chunks = [ts.ckks_vector(context, chunk) for chunk in chunks]
            encrypted_weights.append(encrypted_chunks)
        else:
            encrypted_weights.append(ts.ckks_vector(context, flat_weights))
    return encrypted_weights

# ---------------------------
# Main Client Code
# ---------------------------
def main(client_id):
    # Load local client data (images are 224x224x3)
    data = np.load(f'client_data_{client_id}.npz')
    X_client = data["X"]
    y_client = data["Y"]
    
    # Load the latest global model.
    try:
        model = tf.keras.models.load_model('global_model.h5')
        print(f"Client {client_id}: Loaded global model from 'global_model.h5'.")
    except Exception as e:
        print(f"Client {client_id}: Failed to load global model. Error: {e}")
        return

    # Compile the model.
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
                  loss='categorical_crossentropy')
    
    # Train locally for one epoch.
    model.fit(X_client, y_client, batch_size=BATCH_SIZE, epochs=LOCAL_EPOCHS, verbose=1)
    print(f"Client {client_id}: Finished local training.")

    # Create Tenseal encryption context and encrypt model weights.
    tenseal_context, poly_modulus_degree = create_tenseal_context()
    weights = model.get_weights()
    encrypted_update = encrypt_weights(tenseal_context, weights, poly_modulus_degree)

    # Save the encrypted update.
    with open(f'client_update_{client_id}.pkl', 'wb') as f:
        pickle.dump(encrypted_update, f)
    print(f"Client {client_id}: Encrypted update saved as 'client_update_{client_id}.pkl'.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python client.py <client_id>")
    else:
        client_id = sys.argv[1]
        main(client_id)
