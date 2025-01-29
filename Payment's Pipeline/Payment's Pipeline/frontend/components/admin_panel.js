import React, { useState, useEffect, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';
import { 
  Box,
  Typography,
  Container,
  Grid,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  TablePagination,
  TextField,
  Button,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  LinearProgress,
  Alert,
  IconButton,
  Tooltip,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle
} from '@mui/material';
import { 
  Search, 
  FilterList, 
  Refresh, 
  CloudDownload,
  Visibility,
  Block,
  MonetizationOn
} from '@mui/icons-material';
import { makeStyles } from '@mui/styles';
import { format, parseISO } from 'date-fns';
import jwtDecode from 'jwt-decode';
import { debounce } from 'lodash';
import { TRANSACTION_STATUS, PAYMENT_METHODS } from '../../constants';
import Notification from '../Notification';

const useStyles = makeStyles((theme) => ({
  root: {
    padding: theme.spacing(4),
    backgroundColor: '#f9fafb',
    minHeight: '100vh'
  },
  filterSection: {
    marginBottom: theme.spacing(4),
    padding: theme.spacing(3),
    backgroundColor: '#ffffff',
    borderRadius: '12px',
    boxShadow: '0px 4px 24px rgba(0, 0, 0, 0.08)'
  },
  tableHeader: {
    backgroundColor: theme.palette.primary.main,
    '& .MuiTableCell-head': {
      color: '#ffffff',
      fontWeight: 600
    }
  },
  actionButton: {
    marginLeft: theme.spacing(1),
    transition: 'transform 0.2s',
    '&:hover': {
      transform: 'scale(1.1)'
    }
  },
  loadingContainer: {
    padding: theme.spacing(4),
    textAlign: 'center'
  }
}));

const AdminPanel = () => {
  const classes = useStyles();
  const navigate = useNavigate();
  const [transactions, setTransactions] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [page, setPage] = useState(0);
  const [rowsPerPage, setRowsPerPage] = useState(10);
  const [totalCount, setTotalCount] = useState(0);
  const [filters, setFilters] = useState({
    search: '',
    status: '',
    paymentMethod: '',
    startDate: '',
    endDate: '',
    minAmount: '',
    maxAmount: ''
  });
  const [selectedTransaction, setSelectedTransaction] = useState(null);
  const [notification, setNotification] = useState({ open: false, message: '', severity: 'info' });

  const validateAdminAccess = useCallback(() => {
    const token = localStorage.getItem('paymentPipelineToken');
    if (!token) {
      navigate('/login');
      return;
    }

    try {
      const decoded = jwtDecode(token);
      if (decoded.role !== 'admin') {
        navigate('/unauthorized');
      }
    } catch (error) {
      localStorage.removeItem('paymentPipelineToken');
      navigate('/login');
    }
  }, [navigate]);

  const fetchTransactions = useCallback(async () => {
    try {
      setLoading(true);
      const response = await axios.get('/api/admin/transactions', {
        headers: {
          Authorization: `Bearer ${localStorage.getItem('paymentPipelineToken')}`
        },
        params: {
          page: page + 1,
          pageSize: rowsPerPage,
          ...filters
        }
      });

      setTransactions(response.data.transactions);
      setTotalCount(response.data.totalCount);
      setError(null);
    } catch (err) {
      setError(err.response?.data?.message || 'Failed to fetch transactions');
      setNotification({
        open: true,
        message: err.response?.data?.message || 'Failed to fetch transactions',
        severity: 'error'
      });
    } finally {
      setLoading(false);
    }
  }, [page, rowsPerPage, filters]);

  useEffect(() => {
    validateAdminAccess();
    fetchTransactions();
  }, [validateAdminAccess, fetchTransactions]);

  const handleFilterChange = (name, value) => {
    setFilters(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const handleSearchDebounced = debounce(value => {
    handleFilterChange('search', value);
  }, 500);

  const handleExportData = async () => {
    try {
      const response = await axios.get('/api/admin/transactions/export', {
        headers: {
          Authorization: `Bearer ${localStorage.getItem('paymentPipelineToken')}`,
          'Content-Type': 'blob'
        },
        responseType: 'blob'
      });

      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', `transactions_${Date.now()}.csv`);
      document.body.appendChild(link);
      link.click();
    } catch (err) {
      setNotification({
        open: true,
        message: err.response?.data?.message || 'Export failed',
        severity: 'error'
      });
    }
  };

  const handleTransactionAction = async (transactionId, action) => {
    try {
      await axios.post(`/api/admin/transactions/${transactionId}/${action}`, {}, {
        headers: {
          Authorization: `Bearer ${localStorage.getItem('paymentPipelineToken')}`
        }
      });
      
      setNotification({
        open: true,
        message: `Transaction ${action} successful`,
        severity: 'success'
      });
      fetchTransactions();
    } catch (err) {
      setNotification({
        open: true,
        message: err.response?.data?.message || `Action failed`,
        severity: 'error'
      });
    }
  };

  const renderStatusBadge = (status) => {
    const statusConfig = {
      [TRANSACTION_STATUS.COMPLETED]: { color: '#4caf50', label: 'Completed' },
      [TRANSACTION_STATUS.FAILED]: { color: '#f44336', label: 'Failed' },
      [TRANSACTION_STATUS.PENDING]: { color: '#ff9800', label: 'Pending' },
      [TRANSACTION_STATUS.REFUNDED]: { color: '#9c27b0', label: 'Refunded' },
      [TRANSACTION_STATUS.FLAGGED]: { color: '#e91e63', label: 'Flagged' }
    };

    return (
      <Box
        sx={{
          backgroundColor: statusConfig[status]?.color + '20',
          color: statusConfig[status]?.color,
          padding: '4px 12px',
          borderRadius: '20px',
          display: 'inline-block',
          fontWeight: 500
        }}
      >
        {statusConfig[status]?.label}
      </Box>
    );
  };

  return (
    <div className={classes.root}>
      <Container maxWidth="xl">
        <Box mb={4}>
          <Typography variant="h4" component="h1" gutterBottom>
            Transaction Management
          </Typography>
          <Box display="flex" justifyContent="space-between" alignItems="center">
            <Button
              variant="contained"
              color="primary"
              startIcon={<CloudDownload />}
              onClick={handleExportData}
            >
              Export Data
            </Button>
            <Box>
              <Button
                variant="outlined"
                startIcon={<Refresh />}
                onClick={fetchTransactions}
                className={classes.actionButton}
              >
                Refresh
              </Button>
            </Box>
          </Box>
        </Box>

        <Paper className={classes.filterSection}>
          <Grid container spacing={3}>
            <Grid item xs={12} md={4}>
              <TextField
                fullWidth
                label="Search Transactions"
                variant="outlined"
                InputProps={{
                  startAdornment: <Search />
                }}
                onChange={(e) => handleSearchDebounced(e.target.value)}
              />
            </Grid>
            <Grid item xs={6} md={2}>
              <FormControl fullWidth>
                <InputLabel>Status</InputLabel>
                <Select
                  value={filters.status}
                  onChange={(e) => handleFilterChange('status', e.target.value)}
                  label="Status"
                >
                  <MenuItem value="">All</MenuItem>
                  {Object.values(TRANSACTION_STATUS).map(status => (
                    <MenuItem key={status} value={status}>
                      {status}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={6} md={2}>
              <FormControl fullWidth>
                <InputLabel>Payment Method</InputLabel>
                <Select
                  value={filters.paymentMethod}
                  onChange={(e) => handleFilterChange('paymentMethod', e.target.value)}
                  label="Payment Method"
                >
                  <MenuItem value="">All</MenuItem>
                  {Object.values(PAYMENT_METHODS).map(method => (
                    <MenuItem key={method} value={method}>
                      {method}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>
          </Grid>
        </Paper>

        {loading ? (
          <Box className={classes.loadingContainer}>
            <LinearProgress />
            <Typography variant="body1" mt={2}>
              Loading transactions...
            </Typography>
          </Box>
        ) : error ? (
          <Alert severity="error" variant="outlined">
            {error}
          </Alert>
        ) : (
          <>
            <TableContainer component={Paper}>
              <Table>
                <TableHead className={classes.tableHeader}>
                  <TableRow>
                    <TableCell>Transaction ID</TableCell>
                    <TableCell>Amount</TableCell>
                    <TableCell>Status</TableCell>
                    <TableCell>Payment Method</TableCell>
                    <TableCell>Date</TableCell>
                    <TableCell>Actions</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {transactions.map(transaction => (
                    <TableRow key={transaction.id}>
                      <TableCell>{transaction.shortId}</TableCell>
                      <TableCell>
                        {new Intl.NumberFormat('en-US', {
                          style: 'currency',
                          currency: transaction.currency
                        }).format(transaction.amount)}
                      </TableCell>
                      <TableCell>{renderStatusBadge(transaction.status)}</TableCell>
                      <TableCell>{transaction.paymentMethod}</TableCell>
                      <TableCell>
                        {format(parseISO(transaction.createdAt), 'PPpp')}
                      </TableCell>
                      <TableCell>
                        <Tooltip title="View Details">
                          <IconButton
                            onClick={() => setSelectedTransaction(transaction)}
                          >
                            <Visibility />
                          </IconButton>
                        </Tooltip>
                        {transaction.status === TRANSACTION_STATUS.COMPLETED && (
                          <Tooltip title="Initiate Refund">
                            <IconButton
                              onClick={() => handleTransactionAction(transaction.id, 'refund')}
                            >
                              <MonetizationOn color="secondary" />
                            </IconButton>
                          </Tooltip>
                        )}
                        <Tooltip title="Flag Transaction">
                          <IconButton
                            onClick={() => handleTransactionAction(transaction.id, 'flag')}
                          >
                            <Block color="error" />
                          </IconButton>
                        </Tooltip>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>

            <TablePagination
              rowsPerPageOptions={[10, 25, 50]}
              component="div"
              count={totalCount}
              rowsPerPage={rowsPerPage}
              page={page}
              onPageChange={(_, newPage) => setPage(newPage)}
              onRowsPerPageChange={(e) => {
                setRowsPerPage(parseInt(e.target.value, 10));
                setPage(0);
              }}
            />
          </>
        )}
      </Container>

      <Notification
        open={notification.open}
        message={notification.message}
        severity={notification.severity}
        onClose={() => setNotification(prev => ({ ...prev, open: false }))}
      />

      <Dialog
        open={!!selectedTransaction}
        onClose={() => setSelectedTransaction(null)}
        maxWidth="md"
      >
        <DialogTitle>Transaction Details</DialogTitle>
        <DialogContent>
          {selectedTransaction && (
            <Box>
              <pre>{JSON.stringify(selectedTransaction, null, 2)}</pre>
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setSelectedTransaction(null)}>Close</Button>
        </DialogActions>
      </Dialog>
    </div>
  );
};

export default AdminPanel;