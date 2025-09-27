import React, { useState, useEffect } from 'react';
import {
  Grid, Card, CardContent, Typography, Box, Button,
  TextField, Select, MenuItem, FormControl, InputLabel,
  Alert, CircularProgress, Chip, Divider
} from '@mui/material';
import {
  Assignment, CloudUpload, Visibility, Mic,
  CheckCircle, Warning, Error
} from '@mui/icons-material';
import axios from 'axios';

const API_BASE = 'http://localhost:8000';

function DiagnosticCenter() {
  const [selectedTest, setSelectedTest] = useState('');
  const [patientId, setPatientId] = useState('');
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const diagnosticTests = [
    { id: 'general', name: 'General Health Assessment', description: 'Comprehensive health evaluation' },
    { id: 'vision', name: 'Vision Analysis', description: 'Medical image analysis' },
    { id: 'voice', name: 'Voice Processing', description: 'Speech pattern analysis' },
    { id: 'symptoms', name: 'Symptom Analysis', description: 'Symptom-based diagnosis' }
  ];

  const runDiagnostic = async () => {
    if (!selectedTest || !patientId) {
      setError('Please select a test type and enter patient ID');
      return;
    }

    setLoading(true);
    setError('');

    try {
      const response = await axios.post(`${API_BASE}/diagnostic/run`, {
        test_type: selectedTest,
        patient_id: patientId
      });
      setResults(response.data);
    } catch (error) {
      setError('Failed to run diagnostic test');
      console.error('Diagnostic error:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Diagnostic Center
      </Typography>

      {/* Test Selection */}
      <Card sx={{ mb: 4 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Run Diagnostic Test
          </Typography>

          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Patient ID"
                value={patientId}
                onChange={(e) => setPatientId(e.target.value)}
                placeholder="Enter patient identifier"
              />
            </Grid>

            <Grid item xs={12} md={6}>
              <FormControl fullWidth>
                <InputLabel>Test Type</InputLabel>
                <Select
                  value={selectedTest}
                  onChange={(e) => setSelectedTest(e.target.value)}
                  label="Test Type"
                >
                  {diagnosticTests.map((test) => (
                    <MenuItem key={test.id} value={test.id}>
                      {test.name}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>

            <Grid item xs={12}>
              {selectedTest && (
                <Alert severity="info">
                  {diagnosticTests.find(t => t.id === selectedTest)?.description}
                </Alert>
              )}
            </Grid>

            <Grid item xs={12}>
              <Button
                variant="contained"
                onClick={runDiagnostic}
                disabled={loading || !selectedTest || !patientId}
                startIcon={loading ? <CircularProgress size={20} /> : <Assignment />}
              >
                {loading ? 'Running Diagnostic...' : 'Run Diagnostic'}
              </Button>
            </Grid>
          </Grid>

          {error && (
            <Alert severity="error" sx={{ mt: 2 }}>
              {error}
            </Alert>
          )}
        </CardContent>
      </Card>

      {/* Available Tests */}
      <Card sx={{ mb: 4 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Available Diagnostic Tests
          </Typography>

          <Grid container spacing={2}>
            {diagnosticTests.map((test) => (
              <Grid item xs={12} md={6} key={test.id}>
                <Card variant="outlined">
                  <CardContent>
                    <Typography variant="h6">{test.name}</Typography>
                    <Typography variant="body2" color="textSecondary">
                      {test.description}
                    </Typography>
                    <Box mt={2}>
                      <Chip
                        label="Available"
                        color="success"
                        size="small"
                        icon={<CheckCircle />}
                      />
                    </Box>
                  </CardContent>
                </Card>
              </Grid>
            ))}
          </Grid>
        </CardContent>
      </Card>

      {/* Results */}
      {results && (
        <Card>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              Diagnostic Results
            </Typography>

            <Grid container spacing={2}>
              <Grid item xs={12} md={6}>
                <Typography variant="subtitle2">Patient ID:</Typography>
                <Typography variant="body1">{results.patient_id}</Typography>
              </Grid>

              <Grid item xs={12} md={6}>
                <Typography variant="subtitle2">Test Type:</Typography>
                <Typography variant="body1">{results.test_type}</Typography>
              </Grid>

              <Grid item xs={12}>
                <Divider sx={{ my: 2 }} />
                <Typography variant="subtitle2">Results:</Typography>
                <Alert
                  severity={results.confidence > 0.8 ? 'success' : results.confidence > 0.6 ? 'warning' : 'error'}
                  sx={{ mt: 1 }}
                >
                  Confidence: {(results.confidence * 100).toFixed(1)}%
                  <br />
                  Diagnosis: {results.diagnosis || 'No specific diagnosis'}
                  {results.recommendations && (
                    <>
                      <br />
                      Recommendations: {results.recommendations}
                    </>
                  )}
                </Alert>
              </Grid>
            </Grid>
          </CardContent>
        </Card>
      )}
    </Box>
  );
}

export default DiagnosticCenter;
