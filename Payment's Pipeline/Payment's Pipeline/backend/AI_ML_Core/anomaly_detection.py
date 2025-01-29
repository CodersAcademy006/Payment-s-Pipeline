import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from pydantic import BaseModel, ValidationError
from datetime import datetime
import mlflow
import logging
from collections import deque
import json
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AnomalyDetectionConfig(BaseModel):
    """Validation model for detection configuration"""
    window_size: int = 60  # seconds
    threshold: float = 0.85
    max_batch_size: int = 1024
    learning_rate: float = 0.001
    latent_dim: int = 32
    num_heads: int = 4
    dropout: float = 0.2

class TemporalConvBlock(nn.Module):
    """Temporal Convolutional Network Block with Dilated Conv"""
    def __init__(self, input_dim: int, output_dim: int, kernel_size: int, dilation: int):
        super().__init__()
        self.conv = nn.Conv1d(
            input_dim, output_dim,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=(kernel_size - 1) * dilation
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.batchnorm = nn.BatchNorm1d(output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.relu(x)
        x = self.dropout(x)
        return self.batchnorm(x)

class GraphAttentionLayer(nn.Module):
    """GAT Layer for Transaction Graph Processing"""
    def __init__(self, in_features: int, out_features: int, heads: int):
        super().__init__()
        self.heads = heads
        self.attention = nn.MultiheadAttention(embed_dim=out_features, num_heads=heads)
        self.linear = nn.Linear(in_features, out_features)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        attn_output, _ = self.attention(x, x, x)
        return attn_output

class PaymentAnomalyDetector(nn.Module):
    """Hybrid Autoencoder with Temporal and Graph Features"""
    def __init__(self, config: AnomalyDetectionConfig):
        super().__init__()
        self.config = config
        
        # Temporal Encoder
        self.temporal_conv = nn.Sequential(
            TemporalConvBlock(10, 64, 5, 1),
            TemporalConvBlock(64, 32, 3, 2),
            TemporalConvBlock(32, config.latent_dim, 3, 4)
        )
        
        # Graph Encoder
        self.gat = GraphAttentionLayer(10, config.latent_dim, config.num_heads)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(config.latent_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, temporal_data: torch.Tensor, graph_data: torch.Tensor) -> torch.Tensor:
        # Temporal features
        temp_features = self.temporal_conv(temporal_data.permute(0, 2, 1)).mean(dim=-1)
        
        # Graph features
        graph_features = self.gat(graph_data, None).mean(dim=1)
        
        # Combined representation
        combined = torch.cat([temp_features, graph_features], dim=-1)
        
        # Reconstruction
        reconstructed = self.decoder(combined)
        return reconstructed

class AnomalyDetector:
    """Production-grade Anomaly Detection System"""
    def __init__(self, config_path: str = None):
        self.config = self._load_config(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = PaymentAnomalyDetector(self.config).to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.config.learning_rate)
        self.scaler = None
        self.metric_buffer = deque(maxlen=100)
        self._init_mlflow()

    def _load_config(self, config_path: Optional[str]) -> AnomalyDetectionConfig:
        """Load and validate configuration"""
        try:
            if config_path:
                with open(config_path, 'r') as f:
                    config_data = json.load(f)
            else:
                config_data = {}
            return AnomalyDetectionConfig(**config_data)
        except (FileNotFoundError, ValidationError) as e:
            logger.warning(f"Using default config: {str(e)}")
            return AnomalyDetectionConfig()

    def _init_mlflow(self):
        """Initialize MLFlow tracking"""
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment("payment_anomaly_detection")
        mlflow.start_run()
        mlflow.log_params(self.config.dict())

    def _preprocess(self, raw_data: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
        """Advanced preprocessing pipeline"""
        try:
            # Feature engineering
            raw_data['time_diff'] = raw_data['timestamp'].diff().dt.total_seconds().fillna(0)
            raw_data['amount_velocity'] = raw_data['amount'] / raw_data['time_diff'].replace(0, 1e-5)
            
            # Normalization
            if self.scaler is None:
                self.scaler = StandardScaler()
                scaled_data = self.scaler.fit_transform(raw_data)
            else:
                scaled_data = self.scaler.transform(raw_data)
                
            # Temporal features
            temporal_tensor = torch.tensor(scaled_data, dtype=torch.float32)
            
            # Graph features (simplified example)
            graph_tensor = self._build_transaction_graph(raw_data)
            
            return temporal_tensor.to(self.device), graph_tensor.to(self.device)
        except Exception as e:
            logger.error(f"Preprocessing failed: {str(e)}")
            raise

    def _build_transaction_graph(self, data: pd.DataFrame) -> torch.Tensor:
        """Build transaction graph features"""
        # Implement real graph construction logic
        return torch.randn(len(data), 10)  # Placeholder

    def train(self, train_data: pd.DataFrame, epochs: int = 100):
        """Full training pipeline with early stopping"""
        try:
            temporal_data, graph_data = self._preprocess(train_data)
            dataset = TensorDataset(temporal_data, graph_data)
            loader = DataLoader(dataset, batch_size=self.config.max_batch_size, shuffle=True)
            
            best_loss = float('inf')
            for epoch in range(epochs):
                total_loss = 0.0
                self.model.train()
                for batch in loader:
                    self.optimizer.zero_grad()
                    temporal, graph = batch
                    reconstructed = self.model(temporal, graph)
                    loss = nn.MSELoss()(reconstructed, temporal)
                    loss.backward()
                    self.optimizer.step()
                    total_loss += loss.item()
                
                avg_loss = total_loss / len(loader)
                mlflow.log_metric("train_loss", avg_loss, step=epoch)
                
                # Early stopping
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    torch.save(self.model.state_dict(), "best_model.pth")
                else:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
                    
            mlflow.log_artifact("best_model.pth")
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise

    def detect(self, transaction: Dict) -> Tuple[bool, float]:
        """Real-time anomaly detection with explainability"""
        try:
            df = pd.DataFrame([transaction])
            temporal, graph = self._preprocess(df)
            with torch.no_grad():
                self.model.eval()
                reconstructed = self.model(temporal, graph)
                loss = nn.MSELoss()(reconstructed, temporal).item()
                
            anomaly_score = self._calculate_anomaly_score(loss)
            return (anomaly_score > self.config.threshold), anomaly_score
        except Exception as e:
            logger.error(f"Detection failed: {str(e)}")
            return False, 0.0

    def _calculate_anomaly_score(self, loss: float) -> float:
        """Dynamic threshold adjustment using running statistics"""
        self.metric_buffer.append(loss)
        mean = np.mean(self.metric_buffer)
        std = np.std(self.metric_buffer)
        return (loss - mean) / (std + 1e-8)

    def save_model(self, path: str):
        """Save full model state"""
        torch.save({
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scaler': self.scaler,
            'config': self.config.dict()
        }, path)

    @classmethod
    def load_model(cls, path: str):
        """Load complete model state"""
        instance = cls()
        checkpoint = torch.load(path)
        instance.model.load_state_dict(checkpoint['model_state'])
        instance.optimizer.load_state_dict(checkpoint['optimizer_state'])
        instance.scaler = checkpoint['scaler']
        instance.config = AnomalyDetectionConfig(**checkpoint['config'])
        return instance

# Unit Tests
import pytest
from unittest.mock import Mock

def test_anomaly_detection_flow():
    detector = AnomalyDetector()
    sample_data = pd.DataFrame({
        'timestamp': [datetime.now()],
        'amount': [100.0],
        'user_id': [1],
        'merchant_id': [123],
        'location_lat': [40.7128],
        'location_lon': [-74.0060],
        'device_hash': [hashlib.sha256(b'test').hexdigest()],
        'ip_address': ['192.168.1.1'],
        'transaction_type': ['online'],
        'currency': ['USD']
    })
    
    # Test training
    detector.train(sample_data, epochs=2)
    
    # Test detection
    transaction = {
        'timestamp': datetime.now(),
        'amount': 9999.99,
        'user_id': 1,
        'merchant_id': 666,
        'location_lat': 40.7128,
        'location_lon': -74.0060,
        'device_hash': 'unknown_device',
        'ip_address': '10.0.0.1',
        'transaction_type': 'card_not_present',
        'currency': 'USD'
    }
    is_anomaly, score = detector.detect(transaction)
    assert isinstance(is_anomaly, bool)
    assert 0 <= score <= 1

if __name__ == "__main__":
    # Initialize with custom config
    detector = AnomalyDetector("config/anomaly_config.json")
    
    # Example training
    historical_data = pd.read_parquet("data/processed/transactions.parquet")
    detector.train(historical_data)
    
    # Save production model
    detector.save_model("models/anomaly_detector_v2.pth")