import React, { useState, useEffect } from 'react';
import {
  Grid, Card, CardContent, Typography, Box,
  LinearProgress, Chip, Alert, Button
} from '@mui/material';
import {
  People, Assignment, TrendingUp, Warning,
  CheckCircle, Error, Info, Assessment, Mic
} from '@mui/icons-material';
import axios from 'axios';

const API_BASE = 'http://localhost:8000';

function DashboardHome() {
  const [systemHealth, setSystemHealth] = useState(null);
  const [stats, setStats] = useState({
    totalPatients: 0,
    activeCases: 0,
    accuracy: 0,
    alerts: 0
  });
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchSystemData();
  }, []);

  const fetchSystemData = async () => {
    try {
      // Fetch system health
      const healthResponse = await axios.get(`${API_BASE}/health`);
      setSystemHealth(healthResponse.data);

      // Fetch accuracy metrics
      const accuracyResponse = await axios.get(`${API_BASE}/accuracy/metrics`);
      setStats(prev => ({
        ...prev,
        accuracy: accuracyResponse.data.overall_accuracy * 100,
        totalPatients: accuracyResponse.data.total_cases,
        activeCases: Math.floor(accuracyResponse.data.total_cases * 0.3)
      }));

      setLoading(false);
    } catch (error) {
      console.error('Failed to fetch system data:', error);
      setLoading(false);
    }
  };

  const StatCard = ({ title, value, icon, color, subtitle }) => (
    <Card sx={{ height: '100%' }}>
      <CardContent>
        <Box display="flex" alignItems="center" justifyContent="space-between">
          <Box>
            <Typography color="textSecondary" variant="h6">
              {title}
            </Typography>
            <Typography variant="h4" sx={{ color: color, fontWeight: 'bold' }}>
              {value}
            </Typography>
            {subtitle && (
              <Typography variant="body2" color="textSecondary">
                {subtitle}
              </Typography>
            )}
          </Box>
          <Box sx={{ color: color }}>
            {icon}
          </Box>
        </Box>
      </CardContent>
    </Card>
  );

  if (loading) {
    return (
      <Box sx={{ width: '100%', mt: 2 }}>
        <LinearProgress />
        <Typography variant="h6" sx={{ mt: 2, textAlign: 'center' }}>
          Loading Healthcare System Dashboard...
        </Typography>
      </Box>
    );
  }

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Healthcare AI Dashboard
      </Typography>

      {/* System Status Alert */}
      <Alert
        severity={systemHealth?.status === 'healthy' ? 'success' : 'warning'}
        sx={{ mb: 3 }}
        icon={systemHealth?.status === 'healthy' ? <CheckCircle /> : <Warning />}
      >
        System Status: {systemHealth?.status || 'Unknown'} -
        AI Model Accuracy: {systemHealth?.model_accuracy || 'Not evaluated'}
      </Alert>

      {/* Stats Cards */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} sm={6} md={3}>
          <StatCard
            title="Total Patients"
            value={stats.totalPatients}
            icon={<People sx={{ fontSize: 40 }} />}
            color="#1976d2"
            subtitle="Registered in system"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <StatCard
            title="Active Cases"
            value={stats.activeCases}
            icon={<Assignment sx={{ fontSize: 40 }} />}
            color="#ff9800"
            subtitle="Under treatment"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <StatCard
            title="AI Accuracy"
            value={`${stats.accuracy.toFixed(1)}%`}
            icon={<TrendingUp sx={{ fontSize: 40 }} />}
            color="#4caf50"
            subtitle="Diagnostic precision"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <StatCard
            title="System Alerts"
            value={stats.alerts}
            icon={<Warning sx={{ fontSize: 40 }} />}
            color="#f44336"
            subtitle="Require attention"
          />
        </Grid>
      </Grid>

      {/* Quick Actions */}
      <Card sx={{ mb: 4 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Quick Actions
          </Typography>
          <Box display="flex" gap={2} flexWrap="wrap">
            <Button variant="contained" startIcon={<People />}>
              Add New Patient
            </Button>
            <Button variant="contained" startIcon={<Assessment />}>
              Run Diagnostics
            </Button>
            <Button variant="outlined" startIcon={<Mic />}>
              Voice Input
            </Button>
            <Button variant="outlined" startIcon={<TrendingUp />}>
              View Analytics
            </Button>
          </Box>
        </CardContent>
      </Card>

      {/* System Information */}
      <Card>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            System Information
          </Typography>
          <Grid container spacing={2}>
            <Grid item xs={12} md={6}>
              <Typography variant="body2" color="textSecondary">
                Version: {systemHealth?.version || 'Unknown'}
              </Typography>
              <Typography variant="body2" color="textSecondary">
                Service: {systemHealth?.service || 'Healthcare AI System'}
              </Typography>
              <Typography variant="body2" color="textSecondary">
                Last Updated: {new Date().toLocaleString()}
              </Typography>
            </Grid>
            <Grid item xs={12} md={6}>
              <Typography variant="body2" color="textSecondary">
                Active Agents:
              </Typography>
              <Box display="flex" gap={1} mt={1} flexWrap="wrap">
                {systemHealth?.agents?.map((agent, index) => (
                  <Chip
                    key={index}
                    label={agent}
                    size="small"
                    color="primary"
                    icon={<CheckCircle sx={{ fontSize: 16 }} />}
                  />
                ))}
              </Box>
            </Grid>
          </Grid>
        </CardContent>
      </Card>
    </Box>
  );
}

export default DashboardHome;
