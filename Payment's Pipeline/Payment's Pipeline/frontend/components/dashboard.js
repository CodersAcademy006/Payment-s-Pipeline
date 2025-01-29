import React, { useState, useEffect, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';
import {
  Box,
  Typography,
  Container,
  Grid,
  Paper,
  LinearProgress,
  Alert,
  IconButton,
  Skeleton,
  Divider,
  MenuItem,
  FormControl,
  InputLabel,
  Select
} from '@mui/material';
import { Refresh, DateRange, ArrowUpward, ArrowDownward } from '@mui/icons-material';
import { makeStyles } from '@mui/styles';
import { LineChart, Line, BarChart, Bar, PieChart, Pie, ResponsiveContainer } from 'recharts';
import { format, subDays, startOfDay, endOfDay } from 'date-fns';
import jwtDecode from 'jwt-decode';
import { DatePicker } from '@mui/x-date-pickers';
import Notification from './Notification';
import { TRANSACTION_STATUS, PAYMENT_METHODS, CHART_COLORS } from '../../constants';

const useStyles = makeStyles((theme) => ({
  root: {
    padding: theme.spacing(4),
    backgroundColor: '#f8fafc',
    minHeight: '100vh'
  },
  metricCard: {
    padding: theme.spacing(3),
    borderRadius: '12px',
    transition: 'transform 0.3s, box-shadow 0.3s',
    '&:hover': {
      transform: 'translateY(-4px)',
      boxShadow: theme.shadows[4]
    }
  },
  chartContainer: {
    height: '400px',
    marginTop: theme.spacing(4),
    padding: theme.spacing(3),
    backgroundColor: '#ffffff',
    borderRadius: '12px'
  },
  recentTransactions: {
    marginTop: theme.spacing(4),
    padding: theme.spacing(3),
    backgroundColor: '#ffffff',
    borderRadius: '12px'
  }
}));

const Dashboard = () => {
  const classes = useStyles();
  const navigate = useNavigate();
  const [metrics, setMetrics] = useState({
    totalVolume: 0,
    successfulTransactions: 0,
    failureRate: 0,
    avgProcessingTime: 0,
    activeDevices: 0
  });
  const [chartData, setChartData] = useState([]);
  const [paymentDistribution, setPaymentDistribution] = useState([]);
  const [recentTransactions, setRecentTransactions] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [dateRange, setDateRange] = useState({
    start: subDays(new Date(), 7),
    end: new Date()
  });
  const [notification, setNotification] = useState({ open: false, message: '', severity: 'info' });

  const validateAccess = useCallback(() => {
    const token = localStorage.getItem('paymentPipelineToken');
    if (!token) {
      navigate('/login');
      return;
    }

    try {
      const decoded = jwtDecode(token);
      if (!['admin', 'analyst'].includes(decoded.role)) {
        navigate('/unauthorized');
      }
    } catch (error) {
      localStorage.removeItem('paymentPipelineToken');
      navigate('/login');
    }
  }, [navigate]);

  const fetchDashboardData = useCallback(async () => {
    try {
      setLoading(true);
      const [metricsRes, chartRes, distributionRes, recentRes] = await Promise.all([
        axios.get('/api/dashboard/metrics', {
          headers: {
            Authorization: `Bearer ${localStorage.getItem('paymentPipelineToken')}`
          },
          params: {
            startDate: startOfDay(dateRange.start).toISOString(),
            endDate: endOfDay(dateRange.end).toISOString()
          }
        }),
        axios.get('/api/dashboard/hourly-transactions', {
          headers: {
            Authorization: `Bearer ${localStorage.getItem('paymentPipelineToken')}`
          },
          params: {
            startDate: startOfDay(dateRange.start).toISOString(),
            endDate: endOfDay(dateRange.end).toISOString()
          }
        }),
        axios.get('/api/dashboard/payment-distribution', {
          headers: {
            Authorization: `Bearer ${localStorage.getItem('paymentPipelineToken')}`
          }
        }),
        axios.get('/api/dashboard/recent-transactions', {
          headers: {
            Authorization: `Bearer ${localStorage.getItem('paymentPipelineToken')}`
          }
        })
      ]);

      setMetrics(metricsRes.data);
      setChartData(chartRes.data);
      setPaymentDistribution(distributionRes.data);
      setRecentTransactions(recentRes.data.transactions);
      setError(null);
    } catch (err) {
      setError(err.response?.data?.message || 'Failed to load dashboard data');
      setNotification({
        open: true,
        message: err.response?.data?.message || 'Dashboard load failed',
        severity: 'error'
      });
    } finally {
      setLoading(false);
    }
  }, [dateRange]);

  useEffect(() => {
    validateAccess();
    fetchDashboardData();
  }, [validateAccess, fetchDashboardData]);

  const handleDateChange = (date, type) => {
    setDateRange(prev => ({
      ...prev,
      [type]: date
    }));
  };

  const MetricCard = ({ title, value, trend, subtitle }) => (
    <Paper className={classes.metricCard}>
      <Typography variant="subtitle2" color="textSecondary" gutterBottom>
        {title}
      </Typography>
      <Box display="flex" alignItems="center">
        <Typography variant="h4" component="div">
          {typeof value === 'number' ? value.toLocaleString() : value}
        </Typography>
        {trend && (
          <Box ml={1} display="flex" alignItems="center" color={trend > 0 ? '#4caf50' : '#f44336'}>
            {trend > 0 ? <ArrowUpward fontSize="small" /> : <ArrowDownward fontSize="small" />}
            <Typography variant="body2" style={{ marginLeft: 4 }}>
              {Math.abs(trend)}%
            </Typography>
          </Box>
        )}
      </Box>
      {subtitle && (
        <Typography variant="caption" color="textSecondary">
          {subtitle}
        </Typography>
      )}
    </Paper>
  );

  return (
    <div className={classes.root}>
      <Container maxWidth="xl">
        <Box display="flex" justifyContent="space-between" alignItems="center" mb={4}>
          <Typography variant="h4" component="h1">
            Payment Analytics Dashboard
          </Typography>
          <Box display="flex" alignItems="center">
            <FormControl variant="outlined" size="small" sx={{ minWidth: 120, mr: 2 }}>
              <InputLabel>Date Range</InputLabel>
              <Select
                value="custom"
                label="Date Range"
                onChange={() => {}}
              >
                <MenuItem value="24h">Last 24 hours</MenuItem>
                <MenuItem value="7d">Last 7 days</MenuItem>
                <MenuItem value="30d">Last 30 days</MenuItem>
                <MenuItem value="custom">Custom Range</MenuItem>
              </Select>
            </FormControl>
            <DatePicker
              label="Start Date"
              value={dateRange.start}
              onChange={(date) => handleDateChange(date, 'start')}
              maxDate={dateRange.end}
              renderInput={(params) => <TextField {...params} size="small" sx={{ width: 150, mr: 2 }} />}
            />
            <DatePicker
              label="End Date"
              value={dateRange.end}
              onChange={(date) => handleDateChange(date, 'end')}
              minDate={dateRange.start}
              renderInput={(params) => <TextField {...params} size="small" sx={{ width: 150, mr: 2 }} />}
            />
            <IconButton onClick={fetchDashboardData} color="primary">
              <Refresh />
            </IconButton>
          </Box>
        </Box>

        {loading ? (
          <Grid container spacing={3}>
            {[1, 2, 3, 4, 5].map((item) => (
              <Grid item xs={12} sm={6} md={4} lg={2.4} key={item}>
                <Skeleton variant="rectangular" height={120} />
              </Grid>
            ))}
          </Grid>
        ) : error ? (
          <Alert severity="error" variant="outlined">
            {error}
          </Alert>
        ) : (
          <>
            <Grid container spacing={3}>
              <Grid item xs={12} sm={6} md={4} lg={2.4}>
                <MetricCard
                  title="Total Volume"
                  value={`$${metrics.totalVolume.toLocaleString()}`}
                  trend={12.5}
                  subtitle="Last 7 days"
                />
              </Grid>
              <Grid item xs={12} sm={6} md={4} lg={2.4}>
                <MetricCard
                  title="Success Rate"
                  value={`${(metrics.successfulTransactions * 100).toFixed(1)}%`}
                  trend={-2.3}
                />
              </Grid>
              <Grid item xs={12} sm={6} md={4} lg={2.4}>
                <MetricCard
                  title="Avg Processing Time"
                  value={`${metrics.avgProcessingTime}ms`}
                  trend={-8.1}
                />
              </Grid>
              <Grid item xs={12} sm={6} md={4} lg={2.4}>
                <MetricCard
                  title="Active Devices"
                  value={metrics.activeDevices}
                  trend={4.7}
                />
              </Grid>
              <Grid item xs={12} sm={6} md={4} lg={2.4}>
                <MetricCard
                  title="Failure Rate"
                  value={`${(metrics.failureRate * 100).toFixed(1)}%`}
                  trend={3.2}
                />
              </Grid>
            </Grid>

            <Grid container spacing={3} mt={2}>
              <Grid item xs={12} md={8}>
                <Paper className={classes.chartContainer}>
                  <Typography variant="h6" gutterBottom>
                    Transaction Volume Timeline
                  </Typography>
                  <ResponsiveContainer width="100%" height="90%">
                    <LineChart data={chartData}>
                      <Line
                        type="monotone"
                        dataKey="count"
                        stroke={CHART_COLORS.primary}
                        strokeWidth={2}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </Paper>
              </Grid>
              <Grid item xs={12} md={4}>
                <Paper className={classes.chartContainer}>
                  <Typography variant="h6" gutterBottom>
                    Payment Method Distribution
                  </Typography>
                  <ResponsiveContainer width="100%" height="90%">
                    <PieChart>
                      <Pie
                        data={paymentDistribution}
                        dataKey="value"
                        nameKey="name"
                        cx="50%"
                        cy="50%"
                        outerRadius={80}
                        fill={CHART_COLORS.secondary}
                      />
                    </PieChart>
                  </ResponsiveContainer>
                </Paper>
              </Grid>
            </Grid>

            <Paper className={classes.recentTransactions}>
              <Typography variant="h6" gutterBottom>
                Recent Transactions
              </Typography>
              <Grid container spacing={2}>
                {recentTransactions.map((transaction) => (
                  <Grid item xs={12} key={transaction.id}>
                    <Box display="flex" justifyContent="space-between" alignItems="center">
                      <Box>
                        <Typography variant="body2">
                          {transaction.merchant}
                        </Typography>
                        <Typography variant="caption" color="textSecondary">
                          {format(new Date(transaction.timestamp), 'PPpp')}
                        </Typography>
                      </Box>
                      <Box textAlign="right">
                        <Typography variant="body2">
                          {new Intl.NumberFormat('en-US', {
                            style: 'currency',
                            currency: transaction.currency
                          }).format(transaction.amount)}
                        </Typography>
                        <Typography
                          variant="caption"
                          style={{
                            color: transaction.status === TRANSACTION_STATUS.COMPLETED
                              ? '#4caf50'
                              : transaction.status === TRANSACTION_STATUS.FAILED
                              ? '#f44336'
                              : '#ff9800'
                          }}
                        >
                          {transaction.status}
                        </Typography>
                      </Box>
                    </Box>
                    <Divider sx={{ mt: 1 }} />
                  </Grid>
                ))}
              </Grid>
            </Paper>
          </>
        )}
      </Container>

      <Notification
        open={notification.open}
        message={notification.message}
        severity={notification.severity}
        onClose={() => setNotification(prev => ({ ...prev, open: false }))}
      />
    </div>
  );
};

export default Dashboard;