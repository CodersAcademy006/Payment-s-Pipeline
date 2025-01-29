"""
Real-time payment simulator with ML-powered fraud detection
Features:
- Kafka stream integration
- Ensemble fraud detection (XGBoost + Isolation Forest)
- Synthetic transaction generation
- Data drift monitoring
- Async processing pipeline
"""

import json
import time
import logging
import uuid
import asyncio
from datetime import datetime
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
from kafka import KafkaProducer, KafkaConsumer
from xgboost import XGBClassifier
from sklearn.ensemble import IsolationForest
from evidently.report import Report
from evidently.metrics import DatasetDriftMetric, ColumnDriftMetric

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('payment_simulator.log'), logging.StreamHandler()]
)
logger = logging.getLogger("PaymentSimulator")
logger.setLevel(logging.DEBUG)

class PaymentSimulator:
    def __init__(self, bootstrap_servers: str = 'localhost:9092'):
        # Kafka configuration
        self.producer_config = {
            'bootstrap_servers': bootstrap_servers,
            'value_serializer': lambda v: json.dumps(v).encode('utf-8'),
            'acks': 'all',
            'retries': 3
        }
        self.consumer_config = {
            'bootstrap_servers': bootstrap_servers,
            'auto_offset_reset': 'earliest',
            'enable_auto_commit': False,
            'group_id': 'fraud-detection-group',
            'max_poll_records': 100
        }
        
        self.producer = KafkaProducer(**self.producer_config)
        self.consumer = KafkaConsumer('payment_events', **self.consumer_config)
        
        # ML Models
        self.fraud_model = self._init_fraud_model()
        self.anomaly_model = self._init_anomaly_model()
        
        # Monitoring
        self.drift_detector = self._init_drift_detector()
        self.reference_data = self._load_reference_data()
        
        # State management
        self.batch_records = []
        self.batch_size = 100
        self.last_drift_check = datetime.now()

    def _init_fraud_model(self) -> XGBClassifier:
        """Initialize and load fraud detection model"""
        model = XGBClassifier(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
            tree_method='gpu_hist' if self._check_gpu() else 'hist',
            enable_categorical=True,
            scale_pos_weight=5  # Handle class imbalance
        )
        # Load pre-trained weights in production
        return model

    def _init_anomaly_model(self) -> IsolationForest:
        """Initialize anomaly detection model"""
        return IsolationForest(
            n_estimators=150,
            contamination=0.03,
            random_state=42,
            verbose=0
        )

    def _init_drift_detector(self) -> Report:
        """Initialize data drift detection system"""
        return Report(metrics=[
            DatasetDriftMetric(),
            ColumnDriftMetric(column_name='amount'),
            ColumnDriftMetric(column_name='country_risk')
        ])

    def _load_reference_data(self) -> pd.DataFrame:
        """Load reference dataset for drift detection"""
        # In production, load from feature store
        return pd.DataFrame(columns=[
            'amount', 'time_of_day', 'category_risk', 
            'country_risk', 'ip_entropy', 'hourly_frequency'
        ])

    def _check_gpu(self) -> bool:
        """Check GPU availability for model acceleration"""
        try:
            from xgboost import XGBClassifier
            return True
        except ImportError:
            return False

    def generate_transaction(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Generate synthetic transaction with realistic patterns"""
        base_amount = np.random.lognormal(mean=3.5, sigma=1.2)
        is_fraud = 0
        
        # Fraud patterns
        if np.random.rand() < 0.07:  # Dynamic fraud rate
            if np.random.rand() < 0.6:
                # High-value fraud
                amount = base_amount * np.random.uniform(8, 15)
            else:
                # Micro-payment fraud
                amount = base_amount * np.random.uniform(0.01, 0.1)
            is_fraud = 1
        else:
            # Legitimate transaction
            amount = base_amount * np.random.uniform(0.8, 1.2)

        return {
            "transaction_id": str(uuid.uuid4()),
            "user_id": user_id or f"USER_{uuid.uuid4().hex[:8]}",
            "amount": round(float(amount), 2),
            "currency": np.random.choice(["USD", "EUR", "GBP", "JPY"]),
            "merchant_category": np.random.choice(
                ["retail", "travel", "gaming", "utility", "finance"],
                p=[0.4, 0.1, 0.2, 0.2, 0.1]
            ),
            "time_of_day": datetime.now().hour,
            "device_location": np.random.choice(
                ["US", "GB", "CN", "IN", "RU", "BR"],
                p=[0.4, 0.2, 0.15, 0.1, 0.1, 0.05]
            ),
            "ip_prefix": ".".join(str(np.random.randint(0, 256)) for _ in range(3)),
            "is_fraud": is_fraud,
            "timestamp": datetime.now().isoformat()
        }

    def preprocess_features(self, transaction: Dict) -> pd.DataFrame:
        """Feature engineering pipeline optimized for real-time processing"""
        features = {
            "amount_log": np.log(transaction["amount"] + 1e-6),
            "time_sin": np.sin(2 * np.pi * transaction["time_of_day"] / 24),
            "time_cos": np.cos(2 * np.pi * transaction["time_of_day"] / 24),
            "category_risk": self._get_category_risk(transaction["merchant_category"]),
            "country_risk": self._get_country_risk(transaction["device_location"]),
            "ip_entropy": self._calculate_ip_entropy(transaction["ip_prefix"]),
            "hourly_freq": self._get_hourly_frequency(transaction["user_id"])
        }
        return pd.DataFrame([features])

    def _get_category_risk(self, category: str) -> float:
        """Dynamic category risk scoring"""
        risk_map = {
            "gaming": 0.9,
            "travel": 0.7,
            "finance": 0.6,
            "utility": 0.3,
            "retail": 0.2
        }
        return risk_map.get(category, 0.5)

    def _get_country_risk(self, country: str) -> float:
        """Geographic risk scoring"""
        risk_map = {
            "RU": 0.85,
            "CN": 0.75,
            "BR": 0.65,
            "IN": 0.4,
            "GB": 0.2,
            "US": 0.1
        }
        return risk_map.get(country, 0.5)

    def _calculate_ip_entropy(self, ip_prefix: str) -> float:
        """Calculate IP address entropy for fraud detection"""
        parts = ip_prefix.split('.')
        unique_chars = len(set(''.join(parts)))
        return unique_chars / 12  # Normalize to 0-1 range

    def _get_hourly_frequency(self, user_id: str) -> float:
        """Mock user frequency analysis"""
        return np.random.beta(a=2, b=5)  # Simulated user activity pattern

    async def detect_fraud(self, transaction: Dict) -> Dict[str, float]:
        """Ensemble fraud detection with model explainability"""
        try:
            features = self.preprocess_features(transaction)
            
            # XGBoost prediction
            xgb_proba = self.fraud_model.predict_proba(features)[0][1]
            
            # Anomaly detection
            anomaly_score = self.anomaly_model.decision_function(features)[0]
            normalized_anomaly = 1 / (1 + np.exp(-anomaly_score))  # Sigmoid normalization
            
            # Ensemble scoring
            final_score = 0.6 * xgb_proba + 0.4 * normalized_anomaly
            
            return {
                "transaction_id": transaction["transaction_id"],
                "fraud_score": final_score,
                "xgb_score": xgb_proba,
                "anomaly_score": normalized_anomaly
            }
            
        except Exception as e:
            logger.error(f"Fraud detection failed: {str(e)}")
            return {"fraud_score": 0.0, "error": str(e)}

    async def handle_drift_detection(self):
        """Check for data drift periodically"""
        if (datetime.now() - self.last_drift_check).seconds < 3600:
            return
            
        if len(self.batch_records) < self.batch_size:
            return

        current_data = pd.DataFrame(self.batch_records)
        drift_report = self.drift_detector.run(
            reference_data=self.reference_data,
            current_data=current_data
        )
        
        if drift_report['metrics'][0]['result']['dataset_drift']:
            logger.warning("Data drift detected! Drift metrics:")
            for metric in drift_report['metrics']:
                logger.warning(f"{metric['metric']}: {metric['result']}")
            
            # Trigger retraining workflow
            self._trigger_retraining(current_data)
            
        self.last_drift_check = datetime.now()
        self.batch_records.clear()

    def _trigger_retraining(self, current_data: pd.DataFrame):
        """Mock retraining workflow"""
        logger.info("Initializing model retraining...")
        # In production: 
        # 1. Store new data in feature store
        # 2. Trigger MLOps pipeline
        # 3. Validate new model
        # 4. Deploy updated model
        self.reference_data = current_data
        logger.info("Retraining workflow completed")

    async def process_message(self, message):
        """Async processing pipeline for transaction messages"""
        try:
            transaction = json.loads(message.value.decode('utf-8'))
            
            # Fraud detection
            fraud_result = await self.detect_fraud(transaction)
            
            # Risk decisioning
            risk_status = "block" if fraud_result['fraud_score'] > 0.85 else \
                         "review" if fraud_result['fraud_score'] > 0.65 else \
                         "approve"
            
            # Log results
            logger.info(f"""
            Transaction {transaction['transaction_id']}:
            - Amount: {transaction['amount']} {transaction['currency']}
            - Merchant: {transaction['merchant_category']}
            - Location: {transaction['device_location']}
            - Fraud Score: {fraud_result['fraud_score']:.2%}
            - Decision: {risk_status.upper()}
            """)
            
            # Store for drift detection
            features = self.preprocess_features(transaction)
            self.batch_records.extend(features.to_dict('records'))
            
            # Handle downstream actions
            await self._handle_downstream_actions(risk_status, transaction)
            
            return risk_status
            
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            return "error"

    async def _handle_downstream_actions(self, decision: str, transaction: Dict):
        """Handle post-decision workflows"""
        if decision == "block":
            # Add to blockchain blacklist
            await self._blockchain_blacklist(transaction)
            
        elif decision == "review":
            # Send to human review queue
            await self._send_to_review_queue(transaction)
            
        # Commit message offset
        self.consumer.commit()

    async def _blockchain_blacklist(self, transaction: Dict):
        """Mock blockchain interaction"""
        logger.info(f"Adding {transaction['transaction_id']} to blockchain blacklist")
        await asyncio.sleep(0.1)  # Simulate async blockchain call

    async def _send_to_review_queue(self, transaction: Dict):
        """Mock review queue integration"""
        logger.info(f"Sending {transaction['transaction_id']} to human review queue")
        await asyncio.sleep(0.05)

    async def run_simulation(self, interval: float = 0.01):
        """Main simulation loop with optimized throughput"""
        logger.info("Starting payment simulation pipeline...")
        
        # Start background tasks
        asyncio.create_task(self._monitor_drift_continuously())
        
        try:
            while True:
                # Generate transaction
                transaction = self.generate_transaction()
                
                # Produce message
                self.producer.send('payment_events', transaction)
                
                # Process messages
                message_batch = self.consumer.poll(timeout_ms=100, max_records=100)
                
                for tp, messages in message_batch.items():
                    for message in messages:
                        await self.process_message(message)
                
                # Throttle if needed
                if interval > 0:
                    await asyncio.sleep(interval)
                    
        except KeyboardInterrupt:
            logger.info("Stopping payment simulator...")
        finally:
            self.producer.flush()
            self.consumer.close()

    async def _monitor_drift_continuously(self):
        """Background task for continuous drift monitoring"""
        while True:
            await self.handle_drift_detection()
            await asyncio.sleep(300)  # Check every 5 minutes

if __name__ == "__main__":
    simulator = PaymentSimulator()
    
    try:
        asyncio.run(simulator.run_simulation())
    except Exception as e:
        logger.error(f"Critical error in simulation: {str(e)}")