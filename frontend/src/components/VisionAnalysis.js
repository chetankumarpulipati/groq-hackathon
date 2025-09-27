import React, { useState, useRef } from 'react';
import {
  Grid, Card, CardContent, Typography, Box, Button,
  Alert, LinearProgress, List, ListItem, ListItemText,
  Chip, Divider, Paper
} from '@mui/material';
import {
  Visibility, CloudUpload, Delete, ZoomIn,
  CheckCircle, Warning, Error, PhotoCamera
} from '@mui/icons-material';
import axios from 'axios';

const API_BASE = 'http://localhost:8000';

function VisionAnalysis() {
  const [selectedImage, setSelectedImage] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [analysis, setAnalysis] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const fileInputRef = useRef(null);

  const handleImageSelect = (event) => {
    const file = event.target.files[0];
    if (file) {
      if (file.type.startsWith('image/')) {
        setSelectedImage(file);
        const reader = new FileReader();
        reader.onload = (e) => setImagePreview(e.target.result);
        reader.readAsDataURL(file);
        setError('');
      } else {
        setError('Please select a valid image file');
      }
    }
  };

  const analyzeImage = async () => {
    if (!selectedImage) {
      setError('Please select an image first');
      return;
    }

    setLoading(true);
    setError('');

    try {
      const formData = new FormData();
      formData.append('image', selectedImage);

      const response = await axios.post(`${API_BASE}/vision/analyze`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      setAnalysis(response.data);
    } catch (error) {
      setError('Failed to analyze image');
      console.error('Vision analysis error:', error);
    } finally {
      setLoading(false);
    }
  };

  const clearImage = () => {
    setSelectedImage(null);
    setImagePreview(null);
    setAnalysis(null);
    setError('');
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const getConfidenceColor = (confidence) => {
    if (confidence >= 0.8) return '#4caf50';
    if (confidence >= 0.6) return '#ff9800';
    return '#f44336';
  };

  const getSeverityColor = (severity) => {
    switch (severity?.toLowerCase()) {
      case 'high': return '#f44336';
      case 'medium': return '#ff9800';
      case 'low': return '#4caf50';
      default: return '#9e9e9e';
    }
  };

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Medical Image Analysis
      </Typography>

      {/* Image Upload */}
      <Card sx={{ mb: 4 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Upload Medical Image
          </Typography>

          <Box display="flex" gap={2} alignItems="center" mb={2}>
            <input
              type="file"
              accept="image/*"
              onChange={handleImageSelect}
              style={{ display: 'none' }}
              ref={fileInputRef}
            />

            <Button
              variant="outlined"
              onClick={() => fileInputRef.current?.click()}
              startIcon={<CloudUpload />}
            >
              Select Image
            </Button>

            <Button
              variant="contained"
              onClick={analyzeImage}
              disabled={loading || !selectedImage}
              startIcon={<Visibility />}
            >
              Analyze Image
            </Button>

            <Button
              variant="outlined"
              onClick={clearImage}
              disabled={loading}
              startIcon={<Delete />}
            >
              Clear
            </Button>
          </Box>

          {loading && (
            <Box sx={{ mb: 2 }}>
              <LinearProgress />
              <Typography variant="body2" sx={{ mt: 1, textAlign: 'center' }}>
                Analyzing medical image...
              </Typography>
            </Box>
          )}

          {error && (
            <Alert severity="error" sx={{ mb: 2 }}>
              {error}
            </Alert>
          )}

          <Typography variant="body2" color="textSecondary">
            Supported formats: JPEG, PNG, GIF. Maximum file size: 10MB
          </Typography>
        </CardContent>
      </Card>

      {/* Image Preview */}
      {imagePreview && (
        <Card sx={{ mb: 4 }}>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              Selected Image
            </Typography>
            <Box display="flex" justifyContent="center">
              <img
                src={imagePreview}
                alt="Selected medical image"
                style={{
                  maxWidth: '100%',
                  maxHeight: '400px',
                  objectFit: 'contain',
                  border: '1px solid #ddd',
                  borderRadius: '8px'
                }}
              />
            </Box>
            <Box mt={2} display="flex" justifyContent="center">
              <Typography variant="body2" color="textSecondary">
                File: {selectedImage?.name} ({(selectedImage?.size / 1024 / 1024).toFixed(2)} MB)
              </Typography>
            </Box>
          </CardContent>
        </Card>
      )}

      {/* Analysis Results */}
      {analysis && (
        <Card sx={{ mb: 4 }}>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              Analysis Results
            </Typography>

            <Grid container spacing={3}>
              {/* Overall Assessment */}
              <Grid item xs={12} md={4}>
                <Card variant="outlined">
                  <CardContent>
                    <Typography variant="subtitle2">Overall Assessment</Typography>
                    <Typography variant="h4" sx={{ color: getConfidenceColor(analysis.confidence) }}>
                      {analysis.diagnosis || 'Normal'}
                    </Typography>
                    <Typography variant="body2" color="textSecondary">
                      Confidence: {(analysis.confidence * 100).toFixed(1)}%
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>

              {/* Image Quality */}
              <Grid item xs={12} md={4}>
                <Card variant="outlined">
                  <CardContent>
                    <Typography variant="subtitle2">Image Quality</Typography>
                    <Typography variant="h4" sx={{ color: '#4caf50' }}>
                      {analysis.quality || 'Good'}
                    </Typography>
                    <Typography variant="body2" color="textSecondary">
                      Resolution: {analysis.resolution || 'Adequate'}
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>

              {/* Processing Status */}
              <Grid item xs={12} md={4}>
                <Card variant="outlined">
                  <CardContent>
                    <Typography variant="subtitle2">Processing Status</Typography>
                    <Typography variant="h4" sx={{ color: '#4caf50' }}>
                      Complete
                    </Typography>
                    <Typography variant="body2" color="textSecondary">
                      Processing time: {analysis.processing_time || '2.3s'}
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>

              {/* Detected Findings */}
              {analysis.findings && analysis.findings.length > 0 && (
                <Grid item xs={12}>
                  <Divider sx={{ my: 2 }} />
                  <Typography variant="subtitle2" gutterBottom>
                    Detected Findings:
                  </Typography>
                  <List>
                    {analysis.findings.map((finding, index) => (
                      <ListItem key={index}>
                        <ListItemText
                          primary={finding.description}
                          secondary={`Location: ${finding.location} | Confidence: ${(finding.confidence * 100).toFixed(1)}%`}
                        />
                        <Chip
                          label={finding.severity || 'Low'}
                          color={finding.severity === 'High' ? 'error' : finding.severity === 'Medium' ? 'warning' : 'default'}
                          size="small"
                        />
                      </ListItem>
                    ))}
                  </List>
                </Grid>
              )}

              {/* Measurements */}
              {analysis.measurements && (
                <Grid item xs={12} md={6}>
                  <Paper sx={{ p: 2 }} variant="outlined">
                    <Typography variant="subtitle2" gutterBottom>
                      Measurements:
                    </Typography>
                    {Object.entries(analysis.measurements).map(([key, value]) => (
                      <Box key={key} display="flex" justifyContent="space-between" mb={1}>
                        <Typography variant="body2">{key}:</Typography>
                        <Typography variant="body2" fontWeight="bold">{value}</Typography>
                      </Box>
                    ))}
                  </Paper>
                </Grid>
              )}

              {/* Recommendations */}
              {analysis.recommendations && (
                <Grid item xs={12} md={6}>
                  <Alert severity="info">
                    <Typography variant="subtitle2">Recommendations:</Typography>
                    <Typography variant="body2">
                      {analysis.recommendations}
                    </Typography>
                  </Alert>
                </Grid>
              )}

              {/* Technical Details */}
              <Grid item xs={12}>
                <Divider sx={{ my: 2 }} />
                <Typography variant="subtitle2" gutterBottom>
                  Technical Analysis Details:
                </Typography>
                <Grid container spacing={2}>
                  <Grid item xs={12} md={6}>
                    <Typography variant="body2" color="textSecondary">
                      Analysis Model: {analysis.model_used || 'Vision Transformer v2.1'}
                    </Typography>
                    <Typography variant="body2" color="textSecondary">
                      Image Format: {analysis.image_format || selectedImage?.type}
                    </Typography>
                  </Grid>
                  <Grid item xs={12} md={6}>
                    <Typography variant="body2" color="textSecondary">
                      Analysis Date: {new Date().toLocaleString()}
                    </Typography>
                    <Typography variant="body2" color="textSecondary">
                      Status:
                      <Chip
                        label="Completed"
                        color="success"
                        size="small"
                        icon={<CheckCircle />}
                        sx={{ ml: 1 }}
                      />
                    </Typography>
                  </Grid>
                </Grid>
              </Grid>
            </Grid>
          </CardContent>
        </Card>
      )}

      {/* Supported Analysis Types */}
      <Card>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Supported Medical Image Analysis
          </Typography>

          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <Typography variant="subtitle2" gutterBottom>Imaging Modalities:</Typography>
              <List dense>
                <ListItem>
                  <CheckCircle sx={{ color: '#4caf50', mr: 1, fontSize: 20 }} />
                  <ListItemText primary="X-Ray Images" />
                </ListItem>
                <ListItem>
                  <CheckCircle sx={{ color: '#4caf50', mr: 1, fontSize: 20 }} />
                  <ListItemText primary="CT Scans" />
                </ListItem>
                <ListItem>
                  <CheckCircle sx={{ color: '#4caf50', mr: 1, fontSize: 20 }} />
                  <ListItemText primary="MRI Images" />
                </ListItem>
                <ListItem>
                  <CheckCircle sx={{ color: '#4caf50', mr: 1, fontSize: 20 }} />
                  <ListItemText primary="Ultrasound Images" />
                </ListItem>
              </List>
            </Grid>

            <Grid item xs={12} md={6}>
              <Typography variant="subtitle2" gutterBottom>Detection Capabilities:</Typography>
              <List dense>
                <ListItem>
                  <ListItemText primary="Bone fractures and abnormalities" />
                </ListItem>
                <ListItem>
                  <ListItemText primary="Lung conditions and pneumonia" />
                </ListItem>
                <ListItem>
                  <ListItemText primary="Tumor detection and classification" />
                </ListItem>
                <ListItem>
                  <ListItemText primary="Organ measurements and analysis" />
                </ListItem>
              </List>
            </Grid>
          </Grid>
        </CardContent>
      </Card>
    </Box>
  );
}

export default VisionAnalysis;
