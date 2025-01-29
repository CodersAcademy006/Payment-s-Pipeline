// Notification.js
import React from 'react';
import PropTypes from 'prop-types';
import { Snackbar, Alert } from '@mui/material';

const Notification = ({ open, message, severity, onClose }) => {
  return (
    <Snackbar
      open={open}
      autoHideDuration={6000}
      onClose={onClose}
      anchorOrigin={{ vertical: 'top', horizontal: 'center' }}
    >
      <Alert onClose={onClose} severity={severity} sx={{ width: '100%' }}>
        {message}
      </Alert>
    </Snackbar>
  );
};

Notification.propTypes = {
  open: PropTypes.bool.isRequired,
  message: PropTypes.string.isRequired,
  severity: PropTypes.oneOf(Object.values(NOTIFICATION_TYPES)),
  onClose: PropTypes.func.isRequired
};

export default Notification;