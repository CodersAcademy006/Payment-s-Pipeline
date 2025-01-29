"""
Enterprise Feature Store Implementation for Payment Processing
Handles feature engineering, storage, versioning, and serving for ML pipelines
"""

import os
import logging
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from redis import Redis
import pyarrow as pa
import pyarrow.parquet as pq
from prometheus_client import Summary, Counter, start_http_server
from dataclasses import dataclass
import json
import concurrent.futures

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("feature_store.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Metrics configuration
REQUEST_TIME = Summary('feature_retrieval_latency_seconds', 'Time spent processing features')
FEATURE_COUNTER = Counter('feature_requests_total', 'Total feature requests', ['feature_set', 'version'])

@dataclass
class FeatureStoreConfig:
    """Configuration for feature store operations"""
    offline_store_path: str = "s3://feature-store/offline"
    online_store_host: str = "redis://localhost:6379"
    default_feature_set: str = "payment_transactions"
    versioning_enabled: bool = True
    cache_ttl: int = 3600  # 1 hour
    monitoring_port: int = 9090
    max_workers: int = 10
    feature_validation: bool = True
    backfill_window: int = 365  # days

class FeatureStoreManager:
    """Main class for managing feature storage and retrieval"""
    
    def __init__(self, config: FeatureStoreConfig):
        self.config = config
        self.registry = FeatureRegistry()
        self.online_store = OnlineStoreFactory.create(config.online_store_host)
        self.offline_store = OfflineStoreFactory.create(config.offline_store_path)
        self.validator = FeatureValidator() if config.feature_validation else None
        self._init_monitoring()
        
    def _init_monitoring(self):
        """Initialize metrics server"""
        start_http_server(self.config.monitoring_port)
        logger.info(f"Metrics server started on port {self.config.monitoring_port}")
    
    @REQUEST_TIME.time()
    def get_online_features(self, entity_ids: List[str], feature_set: str, version: str = "latest") -> Dict[str, Any]:
        """Retrieve features from online store with real-time serving"""
        feature_set = self.registry.resolve_feature_set(feature_set, version)
        FEATURE_COUNTER.labels(feature_set=feature_set.name, version=feature_set.version).inc()
        
        features = self.online_store.batch_get(
            keys=entity_ids,
            feature_names=feature_set.feature_names
        )
        
        if self.validator:
            self.validator.validate_online_features(features, feature_set.schema)
            
        return features
    
    def get_offline_features(self, start_date: datetime, end_date: datetime, 
                            feature_set: str, version: str = "latest") -> pd.DataFrame:
        """Retrieve historical features for training and batch processing"""
        feature_set = self.registry.resolve_feature_set(feature_set, version)
        return self.offline_store.load_features(
            feature_set=feature_set,
            start_date=start_date,
            end_date=end_date
        )
    
    def ingest_features(self, data: pd.DataFrame, feature_set: str, 
                       is_backfill: bool = False) -> str:
        """Main ingestion pipeline with version control"""
        feature_set_obj = self.registry.get_or_create_feature_set(feature_set)
        
        # Data validation
        if self.validator:
            self.validator.validate_ingestion_data(data, feature_set_obj.schema)
        
        # Feature transformation
        transformed_data = self._transform_features(data, feature_set_obj)
        
        # Version management
        version_hash = self._generate_version_hash(transformed_data)
        feature_set_obj.register_version(version_hash)
        
        # Dual write to offline and online stores
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            offline_future = executor.submit(
                self.offline_store.write_features,
                transformed_data,
                feature_set_obj,
                version_hash
            )
            online_future = executor.submit(
                self.online_store.batch_put,
                transformed_data,
                feature_set_obj
            )
            
        # Handle backfill logic
        if not is_backfill and feature_set_obj.needs_backfill():
            self._trigger_backfill_job(feature_set_obj)
            
        return version_hash
    
    def _transform_features(self, data: pd.DataFrame, feature_set) -> pd.DataFrame:
        """Apply feature engineering transformations"""
        # Add temporal features
        data['transaction_hour'] = data['timestamp'].dt.hour
        data['is_weekend'] = data['timestamp'].dt.weekday >= 5
        
        # Payment-specific features
        data['amount_log'] = np.log1p(data['amount'])
        data['amount_to_balance_ratio'] = data['amount'] / data['account_balance']
        
        # Rolling window features
        data.sort_values('timestamp', inplace=True)
        data['rolling_3d_avg'] = data.groupby('user_id')['amount'].transform(
            lambda x: x.rolling('3D', closed='left').mean()
        )
        
        return data
    
    def _generate_version_hash(self, data: pd.DataFrame) -> str:
        """Create deterministic version hash for data versioning"""
        content_hash = hashlib.sha256(
            pd.util.hash_pandas_object(data).values.tobytes()
        ).hexdigest()
        return f"v{content_hash[:8]}"
    
    def _trigger_backfill_job(self, feature_set):
        """Initialize backfill process for historical data"""
        backfill_job = BackfillJob(
            feature_store=self,
            feature_set=feature_set,
            window_days=self.config.backfill_window
        )
        backfill_job.execute()
    
class FeatureSet:
    """Logical grouping of related features with version control"""
    
    def __init__(self, name: str, schema: Dict[str, type]):
        self.name = name
        self.schema = schema
        self.versions = []
        self.current_version = None
        self.metadata = {
            'created_at': datetime.utcnow(),
            'statistics': {},
            'data_source': None,
            'owner': None
        }
        
    def register_version(self, version_hash: str):
        """Track new feature version"""
        if version_hash not in self.versions:
            self.versions.append(version_hash)
            self.current_version = version_hash
            logger.info(f"Registered new version {version_hash} for {self.name}")
    
    def needs_backfill(self) -> bool:
        """Determine if historical backfill is required"""
        return len(self.versions) == 1  # Backfill only for first version
    
class FeatureRegistry:
    """Central registry for managing feature set metadata"""
    
    def __init__(self):
        self.feature_sets = {}
        self.lock = threading.Lock()
        
    def get_or_create_feature_set(self, name: str, schema: Optional[Dict] = None) -> FeatureSet:
        with self.lock:
            if name not in self.feature_sets:
                if not schema:
                    raise ValueError(f"Schema required for new feature set {name}")
                self.feature_sets[name] = FeatureSet(name, schema)
                logger.info(f"Created new feature set: {name}")
            return self.feature_sets[name]
    
    def resolve_feature_set(self, name: str, version: str) -> FeatureSet:
        feature_set = self.feature_sets.get(name)
        if not feature_set:
            raise KeyError(f"Feature set {name} not found")
            
        if version == "latest":
            return feature_set
        
        if version not in feature_set.versions:
            raise ValueError(f"Version {version} not available for {name}")
            
        return feature_set

class OnlineStore(ABC):
    """Interface for online feature serving"""
    
    @abstractmethod
    def batch_get(self, keys: List[str], feature_names: List[str]) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    def batch_put(self, data: pd.DataFrame, feature_set: FeatureSet):
        pass

class RedisOnlineStore(OnlineStore):
    """Redis implementation for low-latency feature serving"""
    
    def __init__(self, connection_str: str):
        self.client = Redis.from_url(connection_str)
        self.pipeline = self.client.pipeline()
        
    def batch_get(self, keys: List[str], feature_names: List[str]) -> Dict[str, Any]:
        results = {}
        with self.client.pipeline() as pipe:
            for key in keys:
                pipe.hmget(f"features:{key}", feature_names)
            responses = pipe.execute()
            
        for key, response in zip(keys, responses):
            results[key] = dict(zip(feature_names, response))
        return results
    
    def batch_put(self, data: pd.DataFrame, feature_set: FeatureSet):
        with self.client.pipeline() as pipe:
            for _, row in data.iterrows():
                key = f"{feature_set.name}:{row['entity_id']}"
                feature_data = {
                    f.name: row[f.name] for f in feature_set.schema
                }
                pipe.hmset(key, feature_data)
                pipe.expire(key, feature_set.metadata.get('ttl', 3600))
            pipe.execute()

class OfflineStore(ABC):
    """Interface for historical feature storage"""
    
    @abstractmethod
    def write_features(self, data: pd.DataFrame, feature_set: FeatureSet, version: str):
        pass
    
    @abstractmethod
    def load_features(self, feature_set: FeatureSet, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        pass

class ParquetOfflineStore(OfflineStore):
    """Parquet-based implementation for historical features"""
    
    def __init__(self, storage_path: str):
        self.fs = fsspec.filesystem(storage_path.split("://")[0])
        self.base_path = storage_path
        
    def write_features(self, data: pd.DataFrame, feature_set: FeatureSet, version: str):
        path = self._get_feature_path(feature_set.name, version)
        table = pa.Table.from_pandas(data)
        
        with self.fs.open(path, 'wb') as f:
            pq.write_table(table, f, compression='snappy')
            
        logger.info(f"Wrote {len(data)} rows to {path}")
    
    def load_features(self, feature_set: FeatureSet, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        paths = self._discover_partitions(feature_set.name, start_date, end_date)
        return self._read_parquet_files(paths)
    
    def _get_feature_path(self, feature_set: str, version: str) -> str:
        return f"{self.base_path}/{feature_set}/v{version}/data.parquet"
    
class FeatureValidator:
    """Data quality and schema validation for features"""
    
    def validate_ingestion_data(self, data: pd.DataFrame, schema: Dict[str, type]):
        self._validate_schema(data, schema)
        self._check_missing_values(data)
        self._validate_distributions(data)
    
    def validate_online_features(self, features: Dict[str, Any], schema: Dict[str, type]):
        for entity_id, feature_values in features.items():
            self._validate_feature_types(feature_values, schema)
            self._check_feature_freshness(feature_values)
    
    def _validate_schema(self, data: pd.DataFrame, schema: Dict[str, type]):
        # Implementation using Great Expectations or custom validation
        pass

class BackfillJob:
    """Historical feature backfill management"""
    
    def __init__(self, feature_store: FeatureStoreManager, feature_set: FeatureSet, window_days: int):
        self.feature_store = feature_store
        self.feature_set = feature_set
        self.window_days = window_days
        
    def execute(self):
        """Execute backfill across historical data"""
        logger.info(f"Starting backfill for {self.feature_set.name}")
        
        date_range = pd.date_range(
            end=datetime.utcnow(),
            periods=self.window_days,
            freq='D'
        )
        
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = []
            for date in date_range:
                futures.append(
                    executor.submit(
                        self._process_date,
                        date.date()
                    )
                )
                
            for future in concurrent.futures.as_completed(futures):
                future.result()

class StoreFactory:
    """Factory pattern for creating storage implementations"""
    
    @staticmethod
    def create_online_store(connection_str: str) -> OnlineStore:
        if connection_str.startswith("redis"):
            return RedisOnlineStore(connection_str)
        # Add other implementations
        raise ValueError("Unsupported online store type")
    
    @staticmethod
    def create_offline_store(storage_path: str) -> OfflineStore:
        if storage_path.startswith(("s3", "gs", "file")):
            return ParquetOfflineStore(storage_path)
        # Add other implementations
        raise ValueError("Unsupported offline store type")

# Example usage
if __name__ == "__main__":
    config = FeatureStoreConfig(
        offline_store_path="s3://payment-features/offline",
        online_store_host="redis://prod-redis:6379",
        backfill_window=180
    )
    
    # Initialize feature store
    fs = FeatureStoreManager(config)
    
    # Example transaction data
    transactions = pd.DataFrame({
        "transaction_id": ["txn_001", "txn_002"],
        "user_id": ["user_1", "user_2"],
        "amount": [150.0, 200.0],
        "currency": ["USD", "EUR"],
        "timestamp": [datetime.utcnow(), datetime.utcnow()],
        "account_balance": [5000.0, 7500.0]
    })
    
    # Ingest features
    version = fs.ingest_features(
        data=transactions,
        feature_set="payment_transactions"
    )
    
    # Retrieve online features
    features = fs.get_online_features(
        entity_ids=["user_1", "user_2"],
        feature_set="payment_transactions",
        version=version
    )
    
    # Retrieve offline features
    historical_data = fs.get_offline_features(
        start_date=datetime.utcnow() - timedelta(days=7),
        end_date=datetime.utcnow(),
        feature_set="payment_transactions"
    )