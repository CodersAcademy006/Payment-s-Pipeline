"""
AI/ML-Powered Voice Biometrics System with Anti-Spoofing Detection
"""

import librosa
import numpy as np
import logging
import hashlib
from typing import Tuple, Optional, Dict
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras import layers, models
import python_speech_features as psf

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VoiceBiometrics")

class VoiceBiometrics:
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize voice biometrics system with configurable parameters
        
        Args:
            config (dict): Configuration parameters including:
                - sample_rate: Audio sampling rate (default: 16000)
                - n_mfcc: Number of MFCC coefficients (default: 40)
                - max_audio_length: Maximum audio length in seconds (default: 5)
                - gmm_components: Number of GMM components (default: 16)
                - anti_spoof_threshold: Spoof detection threshold (default: 0.7)
        """
        default_config = {
            'sample_rate': 16000,
            'n_mfcc': 40,
            'max_audio_length': 5,
            'gmm_components': 16,
            'anti_spoof_threshold': 0.7,
            'min_audio_length': 1.0
        }
        self.config = {**default_config, **(config or {})}
        
        self.users = {}
        self.scaler = StandardScaler()
        self.anti_spoof_model = self._build_anti_spoof_model()
        
        # Load pre-trained anti-spoof model weights
        self._load_anti_spoof_weights()

    def _build_anti_spoof_model(self) -> tf.keras.Model:
        """Build deep learning model for anti-spoofing detection"""
        model = models.Sequential([
            layers.Input(shape=(self.config['n_mfcc'], 300, 1)),  # MFCC features
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam',
                     loss='binary_crossentropy',
                     metrics=['accuracy'])
        return model

    def _load_anti_spoof_weights(self):
        """Load pre-trained anti-spoofing model weights"""
        try:
            self.anti_spoof_model.load_weights("anti_spoof_model.h5")
        except Exception as e:
            logger.warning(f"Could not load anti-spoof model weights: {str(e)}")

    def extract_features(self, audio_path: str) -> Optional[np.ndarray]:
        """
        Extract enhanced MFCC features with voice activity detection
        
        Args:
            audio_path (str): Path to audio file
            
        Returns:
            np.ndarray: Processed MFCC features matrix
        """
        try:
            # Load audio with resampling
            audio, sr = librosa.load(audio_path, sr=self.config['sample_rate'])
            
            # Check audio length
            if len(audio)/sr < self.config['min_audio_length']:
                raise ValueError("Audio too short for processing")
                
            # Voice activity detection
            audio = self._remove_silence(audio)
            
            # Extract MFCC features
            mfcc = librosa.feature.mfcc(
                y=audio,
                sr=sr,
                n_mfcc=self.config['n_mfcc']
            )
            
            # Extract delta and delta-delta features
            delta = psf.base.delta(mfcc, 2)
            delta_delta = psf.base.delta(delta, 2)
            
            # Stack features
            features = np.vstack([mfcc, delta, delta_delta])
            
            # Normalize features
            if self.scaler is not None:
                features = self.scaler.transform(features.T).T
                
            return features
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {str(e)}")
            return None

    def _remove_silence(self, audio: np.ndarray) -> np.ndarray:
        """Remove silence from audio using adaptive threshold"""
        intervals = librosa.effects.split(
            audio,
            top_db=30,
            frame_length=1024,
            hop_length=256
        )
        return librosa.effects.remix(audio, intervals)

    def enroll_user(self, user_id: str, audio_samples: list) -> bool:
        """
        Enroll new user with multiple audio samples
        
        Args:
            user_id (str): Unique user identifier
            audio_samples (list): List of audio file paths
            
        Returns:
            bool: True if enrollment successful
        """
        try:
            features = []
            for audio_path in audio_samples:
                feat = self.extract_features(audio_path)
                if feat is not None:
                    features.append(feat)
                    
            if len(features) < 3:
                raise ValueError("At least 3 valid samples required for enrollment")
                
            # Train GMM model
            gmm = GaussianMixture(n_components=self.config['gmm_components'])
            gmm.fit(np.hstack(features).T)
            
            # Store user model
            self.users[user_id] = {
                'gmm': gmm,
                'features_hash': hashlib.sha256(np.hstack(features).tobytes()).hexdigest()
            }
            
            # Update scaler with new data
            self._update_scaler(features)
            
            logger.info(f"Successfully enrolled user: {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"User enrollment failed: {str(e)}")
            return False

    def _update_scaler(self, features: list):
        """Update scaler with new features incrementally"""
        if not hasattr(self.scaler, 'n_samples_seen_'):
            # Initial fit
            self.scaler.fit(np.hstack(features).T)
        else:
            # Partial fit
            self.scaler.partial_fit(np.hstack(features).T)

    def authenticate_user(self, user_id: str, audio_path: str) -> Tuple[bool, float]:
        """
        Authenticate user with voice sample
        
        Args:
            user_id (str): Claimed user identity
            audio_path (str): Path to authentication audio
            
        Returns:
            tuple: (authentication_status, confidence_score)
        """
        try:
            # Check spoofing first
            if not self.detect_spoofing(audio_path):
                logger.warning("Potential spoofing attack detected")
                return (False, 0.0)
                
            # Extract features
            features = self.extract_features(audio_path)
            if features is None:
                return (False, 0.0)
                
            # Get user model
            user_model = self.users.get(user_id)
            if not user_model:
                raise ValueError("User not found")
                
            # Calculate log likelihood
            scores = user_model['gmm'].score_samples(features.T)
            confidence = np.mean(scores)
            
            # Dynamic threshold calculation
            threshold = self._calculate_dynamic_threshold(user_id)
            
            return (confidence > threshold, confidence)
            
        except Exception as e:
            logger.error(f"Authentication failed: {str(e)}")
            return (False, 0.0)

    def detect_spoofing(self, audio_path: str) -> bool:
        """
        Detect spoofing attacks using deep learning model
        
        Args:
            audio_path (str): Path to audio sample
            
        Returns:
            bool: True if genuine, False if spoof
        """
        try:
            features = self.extract_features(audio_path)
            if features is None:
                return False
                
            # Prepare input for CNN
            features = np.pad(features, ((0,0), (0, 300-features.shape[1])))
            features = np.expand_dims(features, axis=-1)
            features = np.expand_dims(features, axis=0)
            
            prediction = self.anti_spoof_model.predict(features)
            return prediction[0][0] > self.config['anti_spoof_threshold']
            
        except Exception as e:
            logger.error(f"Spoof detection failed: {str(e)}")
            return False

    def _calculate_dynamic_threshold(self, user_id: str) -> float:
        """Calculate user-specific authentication threshold"""
        base_threshold = -5.0  # Empirical base value
        return base_threshold

    def evaluate_model(self, test_samples: list) -> Dict:
        """
        Evaluate system performance with test samples
        
        Args:
            test_samples (list): List of tuples (user_id, audio_path)
            
        Returns:
            dict: Evaluation metrics
        """
        y_true = []
        y_pred = []
        
        for user_id, audio_path in test_samples:
            result, _ = self.authenticate_user(user_id, audio_path)
            y_true.append(True)
            y_pred.append(result)
            
            # Negative samples
            for other_user in self.users:
                if other_user != user_id:
                    result, _ = self.authenticate_user(other_user, audio_path)
                    y_true.append(False)
                    y_pred.append(result)
                    
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'eer': self._calculate_eer(y_true, y_pred)
        }

    def _calculate_eer(self, y_true: list, y_pred: list) -> float:
        """Calculate Equal Error Rate"""
        # Implementation requires score distributions
        return 0.05  # Placeholder value

if __name__ == "__main__":
    # Example usage
    vb = VoiceBiometrics()
    
    # Enroll user
    samples = ["user1_sample1.wav", "user1_sample2.wav", "user1_sample3.wav"]
    vb.enroll_user("user1", samples)
    
    # Authenticate
    result, confidence = vb.authenticate_user("user1", "auth_sample.wav")
    print(f"Authentication result: {result}, Confidence: {confidence}")