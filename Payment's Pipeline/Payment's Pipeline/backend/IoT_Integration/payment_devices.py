"""
IoT Payment Devices Integration Module
Robust, secure, and optimized payment processing pipeline for IoT devices
"""

import logging
import sys
import time
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Optional, Tuple, Any
import json
import hashlib
from functools import wraps
import ssl
import socket
import uuid


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('payment_processor.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


# Custom Exceptions
class PaymentProcessingError(Exception):
    """Base exception for payment processing errors"""


class DeviceConnectionError(PaymentProcessingError):
    """Exception raised for device connection issues"""


class PaymentValidationError(PaymentProcessingError):
    """Exception raised for invalid payment data"""


class PaymentMethodNotSupported(PaymentProcessingError):
    """Exception raised for unsupported payment methods"""


# Security Constants
ALLOWED_CURRENCIES = {'USD', 'EUR', 'GBP', 'JPY'}
MIN_TLS_VERSION = ssl.TLSVersion.TLSv1_2


def mask_sensitive_data(func):
    """Decorator to mask sensitive information in logs"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        log_data = {
            k: '***MASKED***' if 'card_number' in k or 'cvv' in k else v
            for k, v in result.items()
        }
        logger.debug(f"Processed payment data: {log_data}")
        return result
    return wrapper


class PaymentMethodType(Enum):
    CREDIT_CARD = 'credit_card'
    NFC = 'nfc'
    QR_CODE = 'qr_code'


class PaymentDeviceType(Enum):
    SERIAL = 'serial'
    NETWORK = 'network'
    SIMULATED = 'simulated'


class PaymentMethod(ABC):
    """Abstract base class for payment methods"""
    
    def __init__(self, amount: float, currency: str):
        self.amount = amount
        self.currency = currency
        self.transaction_id = str(uuid.uuid4())
        self.timestamp = time.time()
        self._validate_currency()

    def _validate_currency(self):
        if self.currency not in ALLOWED_CURRENCIES:
            raise PaymentValidationError(
                f"Unsupported currency: {self.currency}"
            )

    @abstractmethod
    def process(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    def validate(self) -> bool:
        pass


class CreditCardPayment(PaymentMethod):
    """Credit card payment processing implementation"""
    
    def __init__(self, amount: float, currency: str, 
                 card_number: str, expiry: str, cvv: str):
        super().__init__(amount, currency)
        self.card_number = card_number
        self.expiry = expiry
        self.cvv = cvv
        self.validate()

    def validate(self):
        if not all([self.card_number, self.expiry, self.cvv]):
            raise PaymentValidationError("Missing credit card details")
        if len(self.cvv) not in {3, 4}:
            raise PaymentValidationError("Invalid CVV length")

    @mask_sensitive_data
    def process(self) -> Dict[str, Any]:
        return {
            'transaction_id': self.transaction_id,
            'amount': self.amount,
            'currency': self.currency,
            'card_number': self.card_number,
            'status': 'processed'
        }


class NFCPayment(PaymentMethod):
    """NFC payment processing implementation"""
    
    def __init__(self, amount: float, currency: str, nfc_token: str):
        super().__init__(amount, currency)
        self.nfc_token = nfc_token
        self.validate()

    def validate(self):
        if not self.nfc_token:
            raise PaymentValidationError("Missing NFC token")
        if len(self.nfc_token) != 64:
            raise PaymentValidationError("Invalid NFC token format")

    def process(self) -> Dict[str, Any]:
        return {
            'transaction_id': self.transaction_id,
            'amount': self.amount,
            'currency': self.currency,
            'nfc_token': self.nfc_token,
            'status': 'processed'
        }


class PaymentMethodFactory:
    """Factory class for creating payment method instances"""
    
    @staticmethod
    def create(payment_type: PaymentMethodType, **kwargs) -> PaymentMethod:
        if payment_type == PaymentMethodType.CREDIT_CARD:
            return CreditCardPayment(**kwargs)
        elif payment_type == PaymentMethodType.NFC:
            return NFCPayment(**kwargs)
        elif payment_type == PaymentMethodType.QR_CODE:
            # Implement QRCodePayment similarly
            raise NotImplementedError("QR Code payment not implemented")
        else:
            raise PaymentMethodNotSupported(
                f"Unsupported payment type: {payment_type}"
            )


class PaymentDevice(ABC):
    """Abstract base class for payment devices"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.is_connected = False

    @abstractmethod
    def connect(self) -> None:
        pass

    @abstractmethod
    def disconnect(self) -> None:
        pass

    @abstractmethod
    def send_command(self, command: str) -> str:
        pass

    @abstractmethod
    def process_payment(self, payment_data: Dict) -> Dict[str, Any]:
        pass


class NetworkPaymentDevice(PaymentDevice):
    """Network-connected payment device implementation"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.socket = None
        self.ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        self.ssl_context.minimum_version = MIN_TLS_VERSION
        self.ssl_context.verify_mode = ssl.CERT_REQUIRED

    def connect(self) -> None:
        try:
            raw_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket = self.ssl_context.wrap_socket(
                raw_socket,
                server_hostname=self.config['hostname']
            )
            self.socket.connect(
                (self.config['hostname'], self.config['port'])
            )
            self.is_connected = True
        except (socket.error, ssl.SSLError) as e:
            raise DeviceConnectionError(
                f"Connection failed: {str(e)}"
            ) from e

    def disconnect(self) -> None:
        if self.socket:
            self.socket.close()
            self.is_connected = False

    def send_command(self, command: str) -> str:
        if not self.is_connected:
            raise DeviceConnectionError("Device not connected")
        try:
            self.socket.sendall(command.encode())
            return self.socket.recv(1024).decode()
        except socket.error as e:
            raise DeviceConnectionError(
                f"Communication error: {str(e)}"
            ) from e

    def process_payment(self, payment_data: Dict) -> Dict[str, Any]:
        # Implement device-specific payment processing
        command = f"PAY {payment_data['amount']} {payment_data['currency']}"
        response = self.send_command(command)
        return {
            'status': 'success' if 'APPROVED' in response else 'failed',
            'response': response
        }


class PaymentProcessor:
    """Main payment processing orchestrator"""
    
    def __init__(self, device: PaymentDevice):
        self.device = device
        self.transaction_retries = 3
        self.retry_delay = 0.5

    def process_transaction(self, payment_method: PaymentMethod) -> Dict:
        try:
            if not self.device.is_connected:
                self.device.connect()
            
            payment_data = payment_method.process()
            
            for attempt in range(self.transaction_retries):
                try:
                    result = self.device.process_payment(payment_data)
                    if result['status'] == 'success':
                        return result
                except PaymentProcessingError as e:
                    if attempt == self.transaction_retries - 1:
                        raise
                    time.sleep(self.retry_delay * (attempt + 1))
            
            raise PaymentProcessingError("Transaction failed after retries")
        
        finally:
            self.device.disconnect()


# Example Usage
if __name__ == "__main__":
    device_config = {
        'type': PaymentDeviceType.NETWORK,
        'hostname': 'payment-gateway.example.com',
        'port': 443,
        'timeout': 10
    }
    
    payment_data = {
        'payment_type': PaymentMethodType.CREDIT_CARD,
        'amount': 100.0,
        'currency': 'USD',
        'card_number': '4111111111111111',
        'expiry': '12/25',
        'cvv': '123'
    }
    
    try:
        device = NetworkPaymentDevice(device_config)
        payment_method = PaymentMethodFactory.create(
            PaymentMethodType.CREDIT_CARD,
            **payment_data
        )
        processor = PaymentProcessor(device)
        result = processor.process_transaction(payment_method)
        print(f"Payment Result: {result}")
    except PaymentProcessingError as e:
        logger.error(f"Payment failed: {str(e)}")
        sys.exit(1)