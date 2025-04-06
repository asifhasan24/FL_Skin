import os
import time
import pickle
import numpy as np
import tenseal as ts

# Global server configuration
NUM_CLIENTS = 5
EPOCHS = 50
UPDATE_FILENAME_TEMPLATE = "client_update_{}.pkl"
AGGREGATED_UPDATE_FILE = "aggregated_update.pkl"

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

def main():
    for epoch in range(EPOCHS):
        print(f"\nServer: Starting aggregation round {epoch+1}/{EPOCHS}")
        encrypted_updates = []
        # Wait until updates from all clients are available.
        while True:
            update_files = [UPDATE_FILENAME_TEMPLATE.format(i) for i in range(1, NUM_CLIENTS+1)]
            if all(os.path.exists(f) for f in update_files):
                print("Server: Received all client updates.")
                break
            else:
                print("Server: Waiting for client updates...")
                time.sleep(5)
        # Load and remove each client's update.
        for client_id in range(1, NUM_CLIENTS+1):
            with open(UPDATE_FILENAME_TEMPLATE.format(client_id), "rb") as f:
                update = pickle.load(f)
                encrypted_updates.append(update)
            os.remove(UPDATE_FILENAME_TEMPLATE.format(client_id))
            print(f"Server: Loaded and removed update from client {client_id}.")
        # Here, the server uses the public key information implicitly from the ciphertexts.
        poly_modulus_degree = 16384  # Known parameter.
        aggregated_encrypted = aggregate_encrypted_weights(encrypted_updates, poly_modulus_degree)
        # Publish the aggregated encrypted update.
        with open(AGGREGATED_UPDATE_FILE, "wb") as f:
            pickle.dump(aggregated_encrypted, f)
        print(f"Server: Published aggregated encrypted update as '{AGGREGATED_UPDATE_FILE}'.")
        # Wait before next round.
        time.sleep(10)
    print("Server: Federated aggregation rounds completed.")

if __name__ == "__main__":
    main()
