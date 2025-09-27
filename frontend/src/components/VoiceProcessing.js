import React, { useState, useEffect } from 'react';
import {
  Grid, Card, CardContent, Typography, Box, Button,
  Alert, LinearProgress, List, ListItem, ListItemText,
  Chip, Divider, TextField
} from '@mui/material';
import {
  Mic, Stop, PlayArrow, Upload, VolumeUp,
  CheckCircle, Warning, Error
} from '@mui/icons-material';
import axios from 'axios';

const API_BASE = 'http://localhost:8000';

function VoiceProcessing() {
  const [isRecording, setIsRecording] = useState(false);
  const [audioFile, setAudioFile] = useState(null);
  const [transcription, setTranscription] = useState('');
  const [analysis, setAnalysis] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [mediaRecorder, setMediaRecorder] = useState(null);

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const recorder = new MediaRecorder(stream);
      const chunks = [];

      recorder.ondataavailable = (event) => {
        chunks.push(event.data);
      };

      recorder.onstop = () => {
        const blob = new Blob(chunks, { type: 'audio/wav' });
        setAudioFile(blob);
        stream.getTracks().forEach(track => track.stop());
      };

      recorder.start();
      setMediaRecorder(recorder);
      setIsRecording(true);
      setError('');
    } catch (error) {
      setError('Failed to access microphone. Please check permissions.');
    }
  };

  const stopRecording = () => {
    if (mediaRecorder) {
      mediaRecorder.stop();
      setIsRecording(false);
    }
  };

  const processAudio = async () => {
    if (!audioFile && !transcription) {
      setError('Please record audio or enter text for analysis');
      return;
    }

    setLoading(true);
    setError('');

    try {
      const formData = new FormData();
      if (audioFile) {
        formData.append('audio', audioFile);
      } else {
        formData.append('text', transcription);
      }

      const response = await axios.post(`${API_BASE}/voice/process`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      setAnalysis(response.data);
      if (response.data.transcription) {
        setTranscription(response.data.transcription);
      }
    } catch (error) {
      setError('Failed to process audio');
      console.error('Voice processing error:', error);
    } finally {
      setLoading(false);
    }
  };

  const clearAll = () => {
    setAudioFile(null);
    setTranscription('');
    setAnalysis(null);
    setError('');
  };

  const getSentimentColor = (sentiment) => {
    switch (sentiment?.toLowerCase()) {
      case 'positive': return '#4caf50';
      case 'negative': return '#f44336';
      case 'neutral': return '#ff9800';
      default: return '#9e9e9e';
    }
  };

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Voice Processing & Analysis
      </Typography>

      {/* Recording Controls */}
      <Card sx={{ mb: 4 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Audio Recording
          </Typography>

          <Box display="flex" gap={2} alignItems="center" mb={2}>
            <Button
              variant={isRecording ? "contained" : "outlined"}
              onClick={isRecording ? stopRecording : startRecording}
              startIcon={isRecording ? <Stop /> : <Mic />}
              color={isRecording ? "error" : "primary"}
            >
              {isRecording ? 'Stop Recording' : 'Start Recording'}
            </Button>

            <Button
              variant="outlined"
              onClick={processAudio}
              disabled={loading || (!audioFile && !transcription)}
              startIcon={<VolumeUp />}
            >
              Process Audio
            </Button>

            <Button
              variant="outlined"
              onClick={clearAll}
              disabled={loading}
            >
              Clear All
            </Button>
          </Box>

          {isRecording && (
            <Alert severity="info" sx={{ mb: 2 }}>
              <Box display="flex" alignItems="center" gap={1}>
                <Typography>Recording in progress...</Typography>
                <LinearProgress sx={{ flexGrow: 1, ml: 2 }} />
              </Box>
            </Alert>
          )}

          {audioFile && !isRecording && (
            <Alert severity="success" sx={{ mb: 2 }}>
              Audio recorded successfully. Click "Process Audio" to analyze.
            </Alert>
          )}

          {loading && (
            <Box sx={{ mb: 2 }}>
              <LinearProgress />
              <Typography variant="body2" sx={{ mt: 1, textAlign: 'center' }}>
                Processing audio...
              </Typography>
            </Box>
          )}

          {error && (
            <Alert severity="error" sx={{ mb: 2 }}>
              {error}
            </Alert>
          )}
        </CardContent>
      </Card>

      {/* Text Input Alternative */}
      <Card sx={{ mb: 4 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Text Input (Alternative)
          </Typography>

          <TextField
            fullWidth
            multiline
            rows={4}
            label="Enter text for analysis"
            value={transcription}
            onChange={(e) => setTranscription(e.target.value)}
            placeholder="Enter patient speech or symptoms description..."
            disabled={loading}
          />

          <Box mt={2}>
            <Typography variant="body2" color="textSecondary">
              You can either record audio above or enter text here for voice pattern analysis.
            </Typography>
          </Box>
        </CardContent>
      </Card>

      {/* Analysis Results */}
      {analysis && (
        <Card sx={{ mb: 4 }}>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              Voice Analysis Results
            </Typography>

            <Grid container spacing={3}>
              {/* Transcription */}
              {analysis.transcription && (
                <Grid item xs={12}>
                  <Typography variant="subtitle2" gutterBottom>
                    Transcription:
                  </Typography>
                  <Box sx={{ p: 2, backgroundColor: '#f5f5f5', borderRadius: 1 }}>
                    <Typography variant="body1">
                      "{analysis.transcription}"
                    </Typography>
                  </Box>
                </Grid>
              )}

              {/* Sentiment Analysis */}
              <Grid item xs={12} md={4}>
                <Card variant="outlined">
                  <CardContent>
                    <Typography variant="subtitle2">Sentiment</Typography>
                    <Typography
                      variant="h4"
                      sx={{ color: getSentimentColor(analysis.sentiment) }}
                    >
                      {analysis.sentiment || 'Unknown'}
                    </Typography>
                    {analysis.sentiment_confidence && (
                      <Typography variant="body2" color="textSecondary">
                        Confidence: {(analysis.sentiment_confidence * 100).toFixed(1)}%
                      </Typography>
                    )}
                  </CardContent>
                </Card>
              </Grid>

              {/* Stress Level */}
              <Grid item xs={12} md={4}>
                <Card variant="outlined">
                  <CardContent>
                    <Typography variant="subtitle2">Stress Level</Typography>
                    <Typography variant="h4" sx={{ color: '#ff9800' }}>
                      {analysis.stress_level || 'Normal'}
                    </Typography>
                    {analysis.stress_confidence && (
                      <Typography variant="body2" color="textSecondary">
                        Confidence: {(analysis.stress_confidence * 100).toFixed(1)}%
                      </Typography>
                    )}
                  </CardContent>
                </Card>
              </Grid>

              {/* Speech Quality */}
              <Grid item xs={12} md={4}>
                <Card variant="outlined">
                  <CardContent>
                    <Typography variant="subtitle2">Speech Quality</Typography>
                    <Typography variant="h4" sx={{ color: '#4caf50' }}>
                      {analysis.quality || 'Good'}
                    </Typography>
                    {analysis.clarity_score && (
                      <Typography variant="body2" color="textSecondary">
                        Clarity: {(analysis.clarity_score * 100).toFixed(1)}%
                      </Typography>
                    )}
                  </CardContent>
                </Card>
              </Grid>

              {/* Medical Indicators */}
              {analysis.medical_indicators && (
                <Grid item xs={12}>
                  <Divider sx={{ my: 2 }} />
                  <Typography variant="subtitle2" gutterBottom>
                    Medical Indicators:
                  </Typography>
                  <List>
                    {analysis.medical_indicators.map((indicator, index) => (
                      <ListItem key={index}>
                        <ListItemText
                          primary={indicator.condition}
                          secondary={`Confidence: ${(indicator.confidence * 100).toFixed(1)}%`}
                        />
                        <Chip
                          label={indicator.severity || 'Low'}
                          color={indicator.severity === 'High' ? 'error' : 'default'}
                          size="small"
                        />
                      </ListItem>
                    ))}
                  </List>
                </Grid>
              )}

              {/* Recommendations */}
              {analysis.recommendations && (
                <Grid item xs={12}>
                  <Alert severity="info">
                    <Typography variant="subtitle2">Recommendations:</Typography>
                    <Typography variant="body2">
                      {analysis.recommendations}
                    </Typography>
                  </Alert>
                </Grid>
              )}
            </Grid>
          </CardContent>
        </Card>
      )}

      {/* Feature Information */}
      <Card>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Voice Analysis Features
          </Typography>

          <Grid container spacing={2}>
            <Grid item xs={12} md={6}>
              <Typography variant="subtitle2">Available Analysis:</Typography>
              <List dense>
                <ListItem>
                  <CheckCircle sx={{ color: '#4caf50', mr: 1, fontSize: 20 }} />
                  <ListItemText primary="Speech-to-Text Transcription" />
                </ListItem>
                <ListItem>
                  <CheckCircle sx={{ color: '#4caf50', mr: 1, fontSize: 20 }} />
                  <ListItemText primary="Sentiment Analysis" />
                </ListItem>
                <ListItem>
                  <CheckCircle sx={{ color: '#4caf50', mr: 1, fontSize: 20 }} />
                  <ListItemText primary="Stress Level Detection" />
                </ListItem>
                <ListItem>
                  <CheckCircle sx={{ color: '#4caf50', mr: 1, fontSize: 20 }} />
                  <ListItemText primary="Speech Quality Assessment" />
                </ListItem>
              </List>
            </Grid>

            <Grid item xs={12} md={6}>
              <Typography variant="subtitle2">Medical Applications:</Typography>
              <List dense>
                <ListItem>
                  <ListItemText primary="Depression screening" />
                </ListItem>
                <ListItem>
                  <ListItemText primary="Anxiety detection" />
                </ListItem>
                <ListItem>
                  <ListItemText primary="Cognitive assessment" />
                </ListItem>
                <ListItem>
                  <ListItemText primary="Speech disorder analysis" />
                </ListItem>
              </List>
            </Grid>
          </Grid>
        </CardContent>
      </Card>
    </Box>
  );
}

export default VoiceProcessing;
