import React, { useState, useEffect } from 'react';
import {
  Grid, Card, CardContent, Typography, Box, Button,
  LinearProgress, Alert, Table, TableBody, TableCell,
  TableContainer, TableHead, TableRow, Paper, Chip
} from '@mui/material';
import {
  Assessment, Refresh, TrendingUp, CheckCircle
} from '@mui/icons-material';
import axios from 'axios';

const API_BASE = 'http://localhost:8000';

function ModelAccuracy() {
  const [metrics, setMetrics] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    fetchAccuracyMetrics();
  }, []);

  const fetchAccuracyMetrics = async () => {
    setLoading(true);
    setError('');

    try {
      const response = await axios.get(`${API_BASE}/accuracy/metrics`);
      setMetrics(response.data);
    } catch (error) {
      setError('Failed to fetch accuracy metrics');
      console.error('Accuracy fetch error:', error);
    } finally {
      setLoading(false);
    }
  };

  const runAccuracyTest = async () => {
    setLoading(true);
    setError('');

    try {
      const response = await axios.post(`${API_BASE}/accuracy/evaluate`);
      setMetrics(response.data);
    } catch (error) {
      setError('Failed to run accuracy evaluation');
      console.error('Accuracy test error:', error);
    } finally {
      setLoading(false);
    }
  };

  const getAccuracyColor = (accuracy) => {
    if (accuracy >= 0.9) return '#4caf50';
    if (accuracy >= 0.8) return '#ff9800';
    return '#f44336';
  };

  const getAccuracyLabel = (accuracy) => {
    if (accuracy >= 0.9) return 'Excellent';
    if (accuracy >= 0.8) return 'Good';
    if (accuracy >= 0.7) return 'Fair';
    return 'Needs Improvement';
  };

  if (loading && !metrics) {
    return (
      <Box>
        <Typography variant="h4" gutterBottom>
          Model Accuracy
        </Typography>
        <LinearProgress />
        <Typography variant="body1" sx={{ mt: 2, textAlign: 'center' }}>
          Loading accuracy metrics...
        </Typography>
      </Box>
    );
  }

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        AI Model Accuracy
      </Typography>

      {/* Controls */}
      <Card sx={{ mb: 4 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Accuracy Evaluation
          </Typography>
          <Box display="flex" gap={2} alignItems="center">
            <Button
              variant="contained"
              onClick={runAccuracyTest}
              disabled={loading}
              startIcon={<Assessment />}
            >
              Run Accuracy Test
            </Button>
            <Button
              variant="outlined"
              onClick={fetchAccuracyMetrics}
              disabled={loading}
              startIcon={<Refresh />}
            >
              Refresh Metrics
            </Button>
            {loading && <LinearProgress sx={{ flexGrow: 1 }} />}
          </Box>

          {error && (
            <Alert severity="error" sx={{ mt: 2 }}>
              {error}
            </Alert>
          )}
        </CardContent>
      </Card>

      {/* Overall Metrics */}
      {metrics && (
        <>
          <Grid container spacing={3} sx={{ mb: 4 }}>
            <Grid item xs={12} md={4}>
              <Card>
                <CardContent>
                  <Typography color="textSecondary" variant="h6">
                    Overall Accuracy
                  </Typography>
                  <Typography
                    variant="h3"
                    sx={{ color: getAccuracyColor(metrics.overall_accuracy), fontWeight: 'bold' }}
                  >
                    {(metrics.overall_accuracy * 100).toFixed(1)}%
                  </Typography>
                  <Chip
                    label={getAccuracyLabel(metrics.overall_accuracy)}
                    color={metrics.overall_accuracy >= 0.8 ? 'success' : 'warning'}
                    size="small"
                  />
                </CardContent>
              </Card>
            </Grid>

            <Grid item xs={12} md={4}>
              <Card>
                <CardContent>
                  <Typography color="textSecondary" variant="h6">
                    Total Test Cases
                  </Typography>
                  <Typography variant="h3" sx={{ fontWeight: 'bold' }}>
                    {metrics.total_cases}
                  </Typography>
                  <Typography variant="body2" color="textSecondary">
                    Evaluated samples
                  </Typography>
                </CardContent>
              </Card>
            </Grid>

            <Grid item xs={12} md={4}>
              <Card>
                <CardContent>
                  <Typography color="textSecondary" variant="h6">
                    Correct Predictions
                  </Typography>
                  <Typography variant="h3" sx={{ color: '#4caf50', fontWeight: 'bold' }}>
                    {metrics.correct_predictions}
                  </Typography>
                  <Typography variant="body2" color="textSecondary">
                    Out of {metrics.total_cases}
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
          </Grid>

          {/* Model-specific Accuracy */}
          {metrics.model_accuracies && (
            <Card sx={{ mb: 4 }}>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Model Performance Breakdown
                </Typography>
                <TableContainer component={Paper} variant="outlined">
                  <Table>
                    <TableHead>
                      <TableRow>
                        <TableCell>Model</TableCell>
                        <TableCell align="right">Accuracy</TableCell>
                        <TableCell align="right">Test Cases</TableCell>
                        <TableCell align="center">Status</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {Object.entries(metrics.model_accuracies).map(([model, data]) => (
                        <TableRow key={model}>
                          <TableCell component="th" scope="row">
                            {model}
                          </TableCell>
                          <TableCell align="right">
                            <Typography sx={{ color: getAccuracyColor(data.accuracy) }}>
                              {(data.accuracy * 100).toFixed(1)}%
                            </Typography>
                          </TableCell>
                          <TableCell align="right">{data.cases}</TableCell>
                          <TableCell align="center">
                            <Chip
                              label={getAccuracyLabel(data.accuracy)}
                              color={data.accuracy >= 0.8 ? 'success' : 'warning'}
                              size="small"
                            />
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>
              </CardContent>
            </Card>
          )}

          {/* Test Results Summary */}
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Test Summary
              </Typography>
              <Grid container spacing={2}>
                <Grid item xs={12} md={6}>
                  <Typography variant="body2" color="textSecondary">
                    Last Evaluation: {new Date().toLocaleString()}
                  </Typography>
                  <Typography variant="body2" color="textSecondary">
                    Test Duration: {metrics.test_duration || 'N/A'}
                  </Typography>
                </Grid>
                <Grid item xs={12} md={6}>
                  <Typography variant="body2" color="textSecondary">
                    Average Confidence: {metrics.avg_confidence ? (metrics.avg_confidence * 100).toFixed(1) + '%' : 'N/A'}
                  </Typography>
                  <Typography variant="body2" color="textSecondary">
                    Benchmark Status:
                    <Chip
                      label="Active"
                      color="success"
                      size="small"
                      icon={<CheckCircle />}
                      sx={{ ml: 1 }}
                    />
                  </Typography>
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        </>
      )}
    </Box>
  );
}

export default ModelAccuracy;
