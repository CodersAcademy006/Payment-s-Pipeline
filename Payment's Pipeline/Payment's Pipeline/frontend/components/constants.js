// constants.js
export const TRANSACTION_STATUS = {
    COMPLETED: 'completed',
    FAILED: 'failed',
    PENDING: 'pending',
    REFUNDED: 'refunded',
    FLAGGED: 'flagged'
  };
  
  export const PAYMENT_METHODS = {
    CREDIT_CARD: 'Credit Card',
    PAYPAL: 'PayPal',
    BANK_TRANSFER: 'Bank Transfer',
    CRYPTO: 'Cryptocurrency'
  };
  
  export const NOTIFICATION_TYPES = {
    ERROR: 'error',
    WARNING: 'warning',
    INFO: 'info',
    SUCCESS: 'success'
  };
  
  export const API_RATE_LIMIT_CONFIG = {
    MAX_REQUESTS: 100,
    WINDOW_MS: 15 * 60 * 1000 // 15 minutes
  };