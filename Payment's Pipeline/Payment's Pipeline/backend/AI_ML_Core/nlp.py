# F:\Github Projects\Python Projects\Python Advanced Projects\Payment's Pipeline\Payment's Pipeline\backend\AI_ML_Core\nlp.py

"""
Advanced NLP Pipeline for Payment Systems
Integrates Transaction Understanding, Sentiment Analysis, and Fraud Pattern Detection
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from pydantic import BaseModel, ValidationError
import mlflow
import logging
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    pipeline,
    BertForSequenceClassification,
    BertTokenizer
)
import spacy
from spacy import Language
from spacy.tokens import Doc
import rasa
from rasa.core.agent import Agent
import asyncio
import json
from datetime import datetime
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NLPConfig(BaseModel):
    """Configuration for NLP pipeline components"""
    model_version: str = "bert-base-uncased"
    rasa_model_path: str = "models/rasa"
    sentiment_threshold: float = 0.85
    fraud_keywords: List[str] = ["refund", "chargeback", "dispute", "unauthorized"]
    entity_labels: Dict[str, str] = {
        "AMOUNT": "MONEY",
        "MERCHANT": "ORG",
        "DATE": "DATE"
    }
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 32
    max_length: int = 128

class PaymentNLP:
    """Production-grade NLP system for payment processing"""
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self._load_models()
        self._init_mlflow()
        self.rasa_agent = self._load_rasa_agent()

    def _load_config(self, config_path: Optional[str]) -> NLPConfig:
        """Load and validate NLP configuration"""
        try:
            if config_path:
                with open(config_path, 'r') as f:
                    config_data = json.load(f)
            else:
                config_data = {}
            return NLPConfig(**config_data)
        except (FileNotFoundError, ValidationError) as e:
            logger.warning(f"Using default config: {str(e)}")
            return NLPConfig()

    def _load_models(self):
        """Load all NLP models with MLflow tracking"""
        mlflow.set_experiment("payment_nlp")
        
        # Load sentiment analysis model
        with mlflow.start_run(run_name="sentiment_model"):
            self.sentiment_tokenizer = AutoTokenizer.from_pretrained(self.config.model_version)
            self.sentiment_model = AutoModelForSequenceClassification.from_pretrained(
                self.config.model_version,
                num_labels=2
            ).to(self.config.device)
            mlflow.log_param("model_type", "bert-sentiment")

        # Load NER model
        with mlflow.start_run(run_name="ner_model"):
            self.ner_model = spacy.load("en_core_web_trf")
            mlflow.log_param("ner_model", "spacy_trf")

        # Load fraud detection model
        with mlflow.start_run(run_name="fraud_classifier"):
            self.fraud_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.fraud_model = BertForSequenceClassification.from_pretrained(
                'bert-base-uncased',
                num_labels=2
            ).to(self.config.device)
            mlflow.log_param("fraud_model", "bert-fraud")

    def _load_rasa_agent(self) -> Agent:
        """Load RASA conversation AI model"""
        return Agent.load(self.config.rasa_model_path)

    def _init_mlflow(self):
        """Initialize MLFlow tracking"""
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.start_run()
        mlflow.log_params(self.config.dict())

    async def process_query(self, text: str) -> Dict:
        """End-to-end processing of customer query"""
        try:
            # Parallel processing
            results = await asyncio.gather(
                self.detect_intent(text),
                self.analyze_sentiment(text),
                self.extract_entities(text),
                self.detect_fraud_pattern(text)
            )
            
            return {
                "intent": results[0],
                "sentiment": results[1],
                "entities": results[2],
                "fraud_risk": results[3]
            }
        except Exception as e:
            logger.error(f"Query processing failed: {str(e)}")
            raise

    async def detect_intent(self, text: str) -> Dict:
        """Identify payment-related intents using RASA"""
        response = await self.rasa_agent.parse_message(text)
        return {
            "intent": response["intent"]["name"],
            "confidence": response["intent"]["confidence"],
            "entities": response["entities"]
        }

    async def analyze_sentiment(self, text: str) -> Dict:
        """Real-time sentiment analysis with risk scoring"""
        inputs = self.sentiment_tokenizer(
            text,
            return_tensors="pt",
            max_length=self.config.max_length,
            truncation=True
        ).to(self.config.device)
        
        with torch.no_grad():
            outputs = self.sentiment_model(**inputs)
        
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        return {
            "negative": probs[0][0].item(),
            "positive": probs[0][1].item(),
            "risk_flag": probs[0][0].item() > self.config.sentiment_threshold
        }

    async def extract_entities(self, text: str) -> List[Dict]:
        """Transaction entity extraction with custom rules"""
        doc = self.ner_model(text)
        return [{
            "text": ent.text,
            "label": ent.label_,
            "start": ent.start_char,
            "end": ent.end_char
        } for ent in doc.ents if ent.label_ in self.config.entity_labels.values()]

    async def detect_fraud_pattern(self, text: str) -> Dict:
        """BERT-based fraud pattern detection"""
        inputs = self.fraud_tokenizer(
            text,
            return_tensors="pt",
            max_length=self.config.max_length,
            truncation=True
        ).to(self.config.device)
        
        with torch.no_grad():
            outputs = self.fraud_model(**inputs)
        
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        return {
            "fraud_prob": probs[0][1].item(),
            "keywords_found": [kw for kw in self.config.fraud_keywords if kw in text.lower()]
        }

    def batch_process(self, texts: List[str]) -> List[Dict]:
        """Batch processing for customer support logs"""
        results = []
        for i in range(0, len(texts), self.config.batch_size):
            batch = texts[i:i+self.config.batch_size]
            results.extend(asyncio.run(self._process_batch(batch)))
        return results

    async def _process_batch(self, batch: List[str]) -> List[Dict]:
        """Process batch of texts asynchronously"""
        return await asyncio.gather(*[self.process_query(text) for text in batch])

    def save_models(self, output_dir: str = "models/nlp"):
        """Save all NLP models"""
        self.sentiment_model.save_pretrained(f"{output_dir}/sentiment")
        self.sentiment_tokenizer.save_pretrained(f"{output_dir}/sentiment")
        self.fraud_model.save_pretrained(f"{output_dir}/fraud")
        self.fraud_tokenizer.save_pretrained(f"{output_dir}/fraud")
        mlflow.log_artifacts(output_dir)

# Unit Tests
import pytest
from unittest.mock import AsyncMock, patch

@pytest.mark.asyncio
async def test_nlp_pipeline():
    nlp = PaymentNLP()
    test_text = "I need to dispute a $500 charge from Amazon on 2023-07-15"
    
    result = await nlp.process_query(test_text)
    
    assert "intent" in result
    assert "sentiment" in result
    assert "entities" in result
    assert "fraud_risk" in result
    assert any(ent["label"] == "MONEY" for ent in result["entities"])

if __name__ == "__main__":
    # Example usage
    nlp_system = PaymentNLP("config/nlp_config.json")
    
    # Real-time processing example
    sample_query = "Why was I charged $299 for Netflix on March 15th?"
    result = asyncio.run(nlp_system.process_query(sample_query))
    print(json.dumps(result, indent=2))
    
    # Batch processing example
    customer_logs = [
        "Request refund for unauthorized transaction ID #4567",
        "How do I update my payment method?",
        "Dispute resolution for December charge"
    ]
    batch_results = nlp_system.batch_process(customer_logs)
    
    # Save models
    nlp_system.save_models()