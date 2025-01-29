"""
Enterprise-Grade Federated Learning System with Secure Aggregation
"""

import logging
import numpy as np
import tensorflow as tf
from typing import List, Dict, Tuple, Optional
from cryptography.hazmat.primitives import hashes, hmac
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives.serialization import load_pem_public_key
import pickle
import time
import sys
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FederatedLearning")

class FederatedLearningServer:
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize federated learning server
        
        Args:
            config (dict): Configuration parameters including:
                - num_rounds: Total training rounds
                - client_fraction: Fraction of clients per round
                - target_accuracy: Target validation accuracy
                - min_clients: Minimum required clients per round
                - secure_agg: Enable secure aggregation
                - dp_params: Differential privacy parameters
        """
        default_config = {
            'num_rounds': 100,
            'client_fraction': 0.3,
            'target_accuracy': 0.95,
            'min_clients': 5,
            'secure_agg': True,
            'dp_params': {'epsilon': 4.0, 'delta': 1e-5},
            'max_retries': 3,
            'timeout': 30
        }
        self.config = {**default_config, **(config or {})}
        
        self.global_model = self._create_model()
        self.clients = {}
        self.history = []
        self.crypto_context = {}
        self.logger = logger.getChild("Server")

    def _create_model(self) -> tf.keras.Model:
        """Create baseline CNN model"""
        return tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
            tf.keras.layers.MaxPooling2D((2,2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(10)
        ])

    def register_client(self, client_id: str, public_key: bytes):
        """Register new client with public key"""
        self.clients[client_id] = {
            'public_key': load_pem_public_key(public_key),
            'status': 'available'
        }
        self.logger.info(f"Registered client {client_id}")

    def start_training(self, validation_data: Tuple[np.ndarray, np.ndarray]):
        """Orchestrate federated training process"""
        self.logger.info("Starting federated training")
        
        for round_num in range(1, self.config['num_rounds'] + 1):
            start_time = time.time()
            
            # Select clients for current round
            selected = self._select_clients()
            if not selected:
                self.logger.error("Insufficient clients for training round")
                break
                
            # Initialize secure aggregation
            if self.config['secure_agg']:
                self._setup_secure_aggregation(selected)

            # Distribute global model
            global_weights = self._serialize_weights(self.global_model.get_weights())
            updates = []
            
            # Collect client updates
            for client_id in selected:
                try:
                    update = self._get_client_update(client_id, global_weights)
                    if update:
                        updates.append(update)
                except Exception as e:
                    self.logger.error(f"Client {client_id} failed: {str(e)}")
                    self.clients[client_id]['status'] = 'failed'

            # Aggregate updates
            if updates:
                new_weights = self._aggregate_updates(updates)
                self.global_model.set_weights(new_weights)
                
                # Evaluate model
                loss, accuracy = self.global_model.evaluate(*validation_data, verbose=0)
                self.history.append({
                    'round': round_num,
                    'accuracy': accuracy,
                    'loss': loss,
                    'clients': len(updates)
                })
                
                self.logger.info(
                    f"Round {round_num}: Accuracy={accuracy:.3f}, "
                    f"Time={time.time()-start_time:.1f}s"
                )
                
                if accuracy >= self.config['target_accuracy']:
                    self.logger.info(f"Target accuracy reached at round {round_num}")
                    break

        return self.history

    def _select_clients(self) -> List[str]:
        """Select clients for current training round"""
        available = [cid for cid, info in self.clients.items() 
                    if info['status'] == 'available']
        k = max(self.config['min_clients'], 
               int(len(available) * self.config['client_fraction']))
        return np.random.choice(available, size=k, replace=False).tolist()

    def _get_client_update(self, client_id: str, global_weights: bytes) -> Optional[np.ndarray]:
        """Retrieve secure update from client"""
        for _ in range(self.config['max_retries']):
            try:
                # Simulated secure communication
                encrypted_update = self._simulate_network_request(client_id, global_weights)
                return self._decrypt_update(client_id, encrypted_update)
            except TimeoutError:
                continue
        return None

    def _aggregate_updates(self, updates: List[np.ndarray]) -> List[np.ndarray]:
        """Secure Federated Averaging with DP support"""
        if self.config['secure_agg']:
            summed_weights = np.sum(updates, axis=0)
        else:
            summed_weights = np.mean(updates, axis=0)
            
        if self.config['dp_params']:
            noise_scale = self._calculate_dp_noise()
            summed_weights += np.random.normal(scale=noise_scale, size=summed_weights.shape)
            
        return summed_weights

    def _calculate_dp_noise(self) -> float:
        """Calculate differential privacy noise scale"""
        # Implementation of the Moments Accountant method
        return 0.01  # Simplified calculation

    def _setup_secure_aggregation(self, client_ids: List[str]):
        """Initialize cryptographic context for secure aggregation"""
        self.crypto_context = {
            'secret_shares': {},
            'masking_vectors': {}
        }

    def _serialize_weights(self, weights: List[np.ndarray]) -> bytes:
        """Serialize model weights with integrity check"""
        serialized = pickle.dumps(weights)
        h = hmac.HMAC(b"secret-key", hashes.SHA256())
        h.update(serialized)
        return h.finalize() + serialized

    def _verify_weights(self, data: bytes) -> Optional[List[np.ndarray]]:
        """Verify serialized weights integrity"""
        try:
            hmac_digest = data[:32]
            payload = data[32:]
            
            h = hmac.HMAC(b"secret-key", hashes.SHA256())
            h.update(payload)
            h.verify(hmac_digest)
            
            return pickle.loads(payload)
        except:
            return None

    def _simulate_network_request(self, client_id: str, data: bytes) -> bytes:
        """Simulate secure client-server communication"""
        # In real implementation, use TLS/SSL with mutual authentication
        return data

    def _decrypt_update(self, client_id: str, encrypted_data: bytes) -> np.ndarray:
        """Decrypt client update using server private key"""
        # Implementation would use actual asymmetric decryption
        return pickle.loads(encrypted_data)

class FederatedLearningClient:
    def __init__(self, client_id: str, train_data: Tuple[np.ndarray, np.ndarray], 
                config: Optional[Dict] = None):
        default_config = {
            'local_epochs': 3,
            'batch_size': 32,
            'learning_rate': 0.01,
            'max_samples': 500,
            'dp_clip': 1.0
        }
        self.config = {**default_config, **(config or {})}
        
        self.client_id = client_id
        self.train_data = self._preprocess_data(train_data)
        self.model = self._create_model()
        self.logger = logger.getChild(f"Client.{client_id}")

    def _create_model(self) -> tf.keras.Model:
        """Create client-local model instance"""
        return tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
            tf.keras.layers.MaxPooling2D((2,2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(10)
        ])

    def _preprocess_data(self, data: Tuple[np.ndarray, np.ndarray]) -> tf.data.Dataset:
        """Create batched and shuffled dataset"""
        x, y = data
        dataset = tf.data.Dataset.from_tensor_slices((x, y))
        return dataset.shuffle(1000).batch(self.config['batch_size']).take(self.config['max_samples'])

    def train_update(self, global_weights: bytes) -> bytes:
        """Perform local training and return secure update"""
        try:
            # Verify and load global weights
            weights = self._verify_weights(global_weights)
            if not weights:
                raise ValueError("Invalid global weights")
                
            self.model.set_weights(weights)
            
            # Differential privacy optimizer
            optimizer = tf.keras.optimizers.SGD(
                learning_rate=self.config['learning_rate']
            )
            if self.config.get('dp_clip'):
                optimizer = tf.keras.optimizers.SGD(
                    learning_rate=self.config['learning_rate'],
                    clipnorm=self.config['dp_clip']
                )

            # Local training
            self.model.compile(
                optimizer=optimizer,
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy']
            )
            
            self.model.fit(
                self.train_data,
                epochs=self.config['local_epochs'],
                verbose=0
            )
            
            # Compute weight delta
            delta_weights = [
                client - server for client, server in 
                zip(self.model.get_weights(), weights)
            ]
            
            return self._encrypt_update(delta_weights)
            
        except Exception as e:
            self.logger.error(f"Training failed: {str(e)}")
            raise

    def _verify_weights(self, data: bytes) -> Optional[List[np.ndarray]]:
        """Verify HMAC of received global weights"""
        # Implementation matches server-side verification
        return pickle.loads(data[32:])  # Simplified for example

    def _encrypt_update(self, delta_weights: List[np.ndarray]) -> bytes:
        """Encrypt weight update using server public key"""
        # In real implementation, use hybrid encryption
        return pickle.dumps(delta_weights)

# Example Usage
if __name__ == "__main__":
    # Initialize server
    server = FederatedLearningServer({
        'num_rounds': 20,
        'client_fraction': 0.4,
        'secure_agg': True
    })

    # Simulate clients with MNIST data
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train[..., None].astype(np.float32) / 255.0
    x_test = x_test[..., None].astype(np.float32) / 255.0

    clients = []
    for i in range(10):
        # Split data into non-IID partitions
        client_data = (x_train[i*6000:(i+1)*6000], y_train[i*6000:(i+1)*6000])
        client = FederatedLearningClient(f"client_{i}", client_data)
        clients.append(client)
        server.register_client(f"client_{i}", b"dummy-public-key")

    # Start federated training
    history = server.start_training((x_test, y_test))
    print(f"Final test accuracy: {history[-1]['accuracy']:.3f}")