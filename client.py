import sys
import os
import time
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import tenseal as ts

# Global client configuration
BATCH_SIZE = 20
LOCAL_EPOCHS = 1  # Local training per round
ROUNDS = 50      # Total federated rounds
LEARNING_RATE = 0.0001

HE_CONTEXT_FILE = "he_context.pkl"

# ---------------------------
# HE Helper Functions
# ---------------------------
def generate_he_context():
    poly_modulus_degree = 16384
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=poly_modulus_degree,
        coeff_mod_bit_sizes=[60, 40, 40, 40, 40, 60]
    )
    context.global_scale = 2**40
    context.generate_galois_keys()
    context.generate_relin_keys()
    return context, poly_modulus_degree

def save_he_context(context):
    with open(HE_CONTEXT_FILE, "wb") as f:
        pickle.dump(context.serialize(), f)

def load_he_context():
    with open(HE_CONTEXT_FILE, "rb") as f:
        full_context = pickle.load(f)
    context = ts.context_from(full_context)
    poly_modulus_degree = context.poly_modulus_degree()
    return context, poly_modulus_degree

def get_he_context(client_id):
    # Client 1 generates and saves the HE context; others load it.
    if client_id == "1":
        context, poly_modulus_degree = generate_he_context()
        save_he_context(context)
        print("Client 1: Generated and shared HE context (full keys).")
        return context, poly_modulus_degree
    else:
        while not os.path.exists(HE_CONTEXT_FILE):
            print(f"Client {client_id}: Waiting for HE context...")
            time.sleep(3)
        context, poly_modulus_degree = load_he_context()
        print(f"Client {client_id}: Loaded shared HE context.")
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

def decrypt_weights(context, encrypted_weights, shapes):
    decrypted_weights = []
    for enc_layer, shape in zip(encrypted_weights, shapes):
        if isinstance(enc_layer, list):
            decrypted_chunks = []
            for chunk in enc_layer:
                decrypted_chunks.extend(chunk.decrypt())
            decrypted_array = np.array(decrypted_chunks[:np.prod(shape)])
        else:
            decrypted_array = np.array(enc_layer.decrypt())[:np.prod(shape)]
        decrypted_weights.append(decrypted_array.reshape(shape))
    return decrypted_weights

# ---------------------------
# Main Client Code (Multiple Rounds)
# ---------------------------
def main(client_id):
    # Load local client data.
    data = np.load(f'client_data_{client_id}.npz')
    X_client = data["X"]
    y_client = data["Y"]
    
    # Load the current global model (for initialization).
    try:
        model = tf.keras.models.load_model('global_model.h5')
        print(f"Client {client_id}: Loaded global model from 'global_model.h5'.")
    except Exception as e:
        print(f"Client {client_id}: Failed to load global model. Error: {e}")
        return

    # Get the shared HE context (client 1 generates; others load).
    context, poly_modulus_degree = get_he_context(client_id)

    # Run federated rounds.
    for round_num in range(ROUNDS):
        print(f"\nClient {client_id}: Starting round {round_num+1}/{ROUNDS}")
        # Local training for one epoch.
        model.fit(X_client, y_client, batch_size=BATCH_SIZE, epochs=LOCAL_EPOCHS, verbose=1)
        print(f"Client {client_id}: Finished local training for round {round_num+1}.")

        # Save local model for explainability.
        local_model_filename = f'client_local_model_{client_id}.h5'
        model.save(local_model_filename)
        print(f"Client {client_id}: Saved local model as '{local_model_filename}'.")

        # Encrypt the updated model weights.
        weights = model.get_weights()
        encrypted_update = encrypt_weights(context, weights, poly_modulus_degree)
        # Save encrypted update for the aggregator.
        with open(f'client_update_{client_id}.pkl', 'wb') as f:
            pickle.dump(encrypted_update, f)
        print(f"Client {client_id}: Saved encrypted update for round {round_num+1}.")

        # Wait for the aggregated encrypted update from the server.
        print(f"Client {client_id}: Waiting for aggregated update from server...")
        while not os.path.exists("aggregated_update.pkl"):
            time.sleep(3)
        with open("aggregated_update.pkl", "rb") as f:
            aggregated_encrypted = pickle.load(f)
        # Decrypt the aggregated update.
        new_weights = decrypt_weights(context, aggregated_encrypted, [w.shape for w in model.get_weights()])
        model.set_weights(new_weights)
        print(f"Client {client_id}: Updated local model from aggregated update for round {round_num+1}.")
        # Save updated global model locally.
        model.save("global_model.h5")
        print(f"Client {client_id}: Saved updated global model for round {round_num+1}.")
        # Wait a short time before next round.
        time.sleep(5)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python client.py <client_id>")
    else:
        client_id = sys.argv[1]
        main(client_id)
