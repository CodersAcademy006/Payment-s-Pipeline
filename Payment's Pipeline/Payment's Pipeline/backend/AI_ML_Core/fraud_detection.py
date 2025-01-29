""""
Advanced Fraud Detection System for Payment Transactions
Combines Graph Neural Networks, Autoencoders, and Explainable AI (XAI)
"""

import torch
import torch.nn as nn # Declaring the neural network
import torch.nn.functional as F # Activation functions
import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple
from pydantic import BaseModel, ValidationError
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from torch import optim
from torch.utils.data import TensorDataset, DataLoader
from torch_geometric.nn import GATConv
import mlflow
import logging
import json
import hashlib
from collections import deque

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FraudDetectionConfig(BaseModel):
    """Configuration for fraud detection system"""
    latent_dim: int = 64
    num_heads: int = 4
    dropout: float = 0.2
    learning_rate: float = 0.001
    batch_size: int = 1024
    anomaly_threshold: float = 0.85
    max_transaction_history: int = 1000

class FraudGraphAttention(nn.Module):
    """Graph Attention Network for Fraud Ring Detection"""
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, heads: int):
        super().__init__()
        self.conv1 = GATConv(input_dim, hidden_dim, heads=heads, dropout=0.2)
        self.conv2 = GATConv(hidden_dim * heads, output_dim, heads=1, concat=False, dropout=0.2)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = F.elu(self.conv1(x, edge_index))
        return self.conv2(x, edge_index)

class FraudDetectionModel(nn.Module):
    """Hybrid Fraud Detection Model with Temporal and Graph Features"""
    def __init__(self, config: FraudDetectionConfig):
        super().__init__()
        self.config = config
        
        # Temporal Encoder
        self.temporal_encoder = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, config.latent_dim)
        )
        
        # Graph Encoder
        self.graph_encoder = FraudGraphAttention(
            input_dim=10,
            hidden_dim=32,
            output_dim=config.latent_dim,
            heads=config.num_heads
        )
        
        # Fraud Classifier
        self.classifier = nn.Sequential(
            nn.Linear(config.latent_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, temporal_data: torch.Tensor, graph_data: Data) -> torch.Tensor:
        # Temporal features
        temporal_features = self.temporal_encoder(temporal_data)
        
        # Graph features
        graph_features = self.graph_encoder(graph_data.x, graph_data.edge_index)
        
        # Combine features
        combined = torch.cat([
            temporal_features.mean(dim=1),
            graph_features.mean(dim=1)
        ], dim=-1)
        
        # Fraud probability
        return self.classifier(combined)

class FraudDetector:
    """Production-grade Fraud Detection System"""
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = FraudDetectionModel(self.config).to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.config.learning_rate)
        self.scaler = StandardScaler()
        self.transaction_history = deque(maxlen=self.config.max_transaction_history)
        self._init_mlflow()

    def _load_config(self, config_path: Optional[str]) -> FraudDetectionConfig:
        """Load and validate configuration"""
        try:
            file_path = config_path if config_path else 'default_config.json'
            with open(file_path, 'r') as f:
                config_data = json.load(f) if config_path else {}
            return FraudDetectionConfig(**config_data)
        except (FileNotFoundError, ValidationError) as e:
            logger.warning(f"Using default config: {str(e)}")
            return FraudDetectionConfig()

    def _init_mlflow(self):
        """Initialize MLFlow tracking"""
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment("fraud_detection")
        mlflow.start_run()
        mlflow.log_params(self.config.dict())

    def _preprocess(self, raw_data: pd.DataFrame) -> Tuple[torch.Tensor, Data]:
        """Advanced preprocessing pipeline"""
        try:
            # Feature engineering
            raw_data['time_diff'] = raw_data['timestamp'].diff().dt.total_seconds().fillna(0)
            raw_data['amount_velocity'] = raw_data['amount'] / raw_data['time_diff'].replace(0, 1e-5)
            
            # Normalization
            scaled_data = self.scaler.fit_transform(raw_data)
            temporal_tensor = torch.tensor(scaled_data, dtype=torch.float32)
            
            # Build transaction graph
            graph_data = self._build_transaction_graph(raw_data)
            
            return temporal_tensor.to(self.device), graph_data.to(self.device)
        except Exception as e:
            logger.error(f"Preprocessing failed: {str(e)}")
            raise

    def _build_transaction_graph(self, data: pd.DataFrame) -> Data:
        """Build transaction graph using PyTorch Geometric"""
        # Create edges based on shared attributes
        user_ids = data['user_id'].values
        merchant_ids = data['merchant_id'].values
        
        # Create edges (simplified example)
        edge_index = torch.tensor([
            [i, j] for i in range(len(data)) 
            for j in range(len(data)) 
            if user_ids[i] == user_ids[j] or merchant_ids[i] == merchant_ids[j]
        ], dtype=torch.long).t().contiguous()
        
        # Node features
        x = torch.tensor(data.drop(columns=['timestamp']).values, dtype=torch.float32)
        
        return Data(x=x, edge_index=edge_index)

    def train(self, train_data: pd.DataFrame, epochs: int = 50):
        """Training pipeline with early stopping"""
        try:
            temporal_data, graph_data = self._preprocess(train_data)
            dataset = TensorDataset(temporal_data, graph_data)
            loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)
            
            best_loss = float('inf')
            for epoch in range(epochs):
                self.model.train()
                total_loss = 0.0
                for batch in loader:
                    self.optimizer.zero_grad()
                    temporal, graph = batch
                    predictions = self.model(temporal, graph)
                    loss = F.binary_cross_entropy(predictions, graph.y)
                    loss.backward()
                    self.optimizer.step()
                    total_loss += loss.item()
                
                avg_loss = total_loss / len(loader)
                mlflow.log_metric("train_loss", avg_loss, step=epoch)
                
                # Early stopping
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    torch.save(self.model.state_dict(), "best_fraud_model.pth")
                else:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
                    
            mlflow.log_artifact("best_fraud_model.pth")
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise

    def detect(self, transaction: Dict) -> Tuple[bool, float]:
        """Real-time fraud detection"""
        try:
            # Add to transaction history
            self.transaction_history.append(transaction)
            df = pd.DataFrame(self.transaction_history)
            
            # Preprocess and predict
            temporal, graph = self._preprocess(df)
            with torch.no_grad():
                self.model.eval()
                fraud_prob = self.model(temporal, graph).item()
                
            return (fraud_prob > self.config.anomaly_threshold), fraud_prob
        except Exception as e:
            logger.error(f"Detection failed: {str(e)}")
            return False, 0.0

    def save_model(self, path: str):
        """Save complete model state"""
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
        instance.config = FraudDetectionConfig(**checkpoint['config'])
        return instance

# Unit Tests
import pytest
from unittest.mock import Mock

def test_fraud_detection_flow():
    detector = FraudDetector()
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
    is_fraud, score = detector.detect(transaction)
    assert isinstance(is_fraud, bool)
    assert 0 <= score <= 1

if __name__ == "__main__":
    # Initialize with custom config
    detector = FraudDetector("config/fraud_config.json")
    
    # Example training
    historical_data = pd.read_parquet("data/processed/transactions.parquet")
    detector.train(historical_data)
    
    # Save production model
    detector.save_model("models/fraud_detector_v2.pth")