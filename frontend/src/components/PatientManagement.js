import React, { useState, useEffect } from 'react';
import {
  Box, Typography, Card, CardContent, Button, Dialog, DialogTitle,
  DialogContent, DialogActions, TextField, Grid, Chip, Alert,
  Table, TableBody, TableCell, TableContainer, TableHead, TableRow,
  Paper, IconButton, MenuItem, FormControl, InputLabel, Select
} from '@mui/material';
import {
  Add, Edit, Visibility, LocalHospital, Person,
  Phone, Email, DateRange, Warning, CheckCircle
} from '@mui/icons-material';
import axios from 'axios';

const API_BASE = 'http://localhost:8001';

function PatientManagement() {
  const [patients, setPatients] = useState([]);
  const [selectedPatient, setSelectedPatient] = useState(null);
  const [openDialog, setOpenDialog] = useState(false);
  const [dialogType, setDialogType] = useState('add'); // 'add', 'edit', 'view'
  const [loading, setLoading] = useState(false);
  const [newPatient, setNewPatient] = useState({
    patient_id: '',
    first_name: '',
    last_name: '',
    date_of_birth: '',
    phone: '',
    email: '',
    medical_history: [],
    current_symptoms: '',
    insurance_id: '',
    emergency_contact: ''
  });

  useEffect(() => {
    loadSamplePatients();
  }, []);

  const loadSamplePatients = () => {
    // Sample patient data for demonstration
    const samplePatients = [
      {
        patient_id: 'P001',
        first_name: 'John',
        last_name: 'Doe',
        date_of_birth: '1978-05-15',
        phone: '(555) 123-4567',
        email: 'john.doe@email.com',
        medical_history: ['Hypertension', 'Diabetes Type 2'],
        current_symptoms: 'Chest pain, shortness of breath',
        insurance_id: 'INS-001234',
        emergency_contact: '(555) 987-6543',
        status: 'Active',
        last_visit: '2025-09-20'
      },
      {
        patient_id: 'P002',
        first_name: 'Jane',
        last_name: 'Smith',
        date_of_birth: '1985-08-22',
        phone: '(555) 234-5678',
        email: 'jane.smith@email.com',
        medical_history: ['Asthma', 'Allergies'],
        current_symptoms: 'Headache, fatigue',
        insurance_id: 'INS-005678',
        emergency_contact: '(555) 876-5432',
        status: 'Active',
        last_visit: '2025-09-25'
      },
      {
        patient_id: 'P003',
        first_name: 'Robert',
        last_name: 'Johnson',
        date_of_birth: '1972-12-03',
        phone: '(555) 345-6789',
        email: 'robert.j@email.com',
        medical_history: ['Heart Disease', 'High Cholesterol'],
        current_symptoms: 'Irregular heartbeat',
        insurance_id: 'INS-009012',
        emergency_contact: '(555) 765-4321',
        status: 'Critical',
        last_visit: '2025-09-27'
      }
    ];
    setPatients(samplePatients);
  };

  const handleAddPatient = async () => {
    setLoading(true);
    try {
      // Validate patient data using the healthcare system
      const validationResponse = await axios.post(`${API_BASE}/healthcare/process`, {
        patient_data: newPatient,
        workflow_type: 'validation',
        priority: 'standard'
      });

      if (validationResponse.data.success) {
        // Add patient to local state (in real app, this would be saved to database)
        const patientWithId = {
          ...newPatient,
          patient_id: `P${String(patients.length + 1).padStart(3, '0')}`,
          status: 'Active',
          last_visit: new Date().toISOString().split('T')[0]
        };
        setPatients([...patients, patientWithId]);
        setOpenDialog(false);
        resetForm();
        alert('Patient added successfully!');
      }
    } catch (error) {
      console.error('Failed to add patient:', error);
      alert('Failed to add patient. Please check the data and try again.');
    }
    setLoading(false);
  };

  const handleDiagnosePatient = async (patient) => {
    setLoading(true);
    try {
      const diagnosisResponse = await axios.post(`${API_BASE}/healthcare/process`, {
        patient_data: {
          patient_id: patient.patient_id,
          symptoms: patient.current_symptoms.split(', '),
          medical_history: patient.medical_history,
          age: calculateAge(patient.date_of_birth)
        },
        workflow_type: 'diagnosis',
        priority: patient.status === 'Critical' ? 'urgent' : 'standard'
      });

      if (diagnosisResponse.data.success) {
        alert(`Diagnosis completed for ${patient.first_name} ${patient.last_name}!\n\nResult: ${JSON.stringify(diagnosisResponse.data.result, null, 2)}`);
      }
    } catch (error) {
      console.error('Diagnosis failed:', error);
      alert('Diagnosis failed. Please try again.');
    }
    setLoading(false);
  };

  const calculateAge = (birthDate) => {
    const today = new Date();
    const birth = new Date(birthDate);
    let age = today.getFullYear() - birth.getFullYear();
    const monthDiff = today.getMonth() - birth.getMonth();
    if (monthDiff < 0 || (monthDiff === 0 && today.getDate() < birth.getDate())) {
      age--;
    }
    return age;
  };

  const resetForm = () => {
    setNewPatient({
      patient_id: '',
      first_name: '',
      last_name: '',
      date_of_birth: '',
      phone: '',
      email: '',
      medical_history: [],
      current_symptoms: '',
      insurance_id: '',
      emergency_contact: ''
    });
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'Critical': return 'error';
      case 'Active': return 'success';
      case 'Inactive': return 'default';
      default: return 'primary';
    }
  };

  return (
    <Box>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
        <Typography variant="h4">Patient Management</Typography>
        <Button
          variant="contained"
          startIcon={<Add />}
          onClick={() => {
            setDialogType('add');
            setOpenDialog(true);
            resetForm();
          }}
        >
          Add New Patient
        </Button>
      </Box>

      <Alert severity="info" sx={{ mb: 3 }}>
        <Typography variant="body2">
          This system integrates with AI diagnostic agents for real-time patient analysis.
          All patient data is validated using healthcare compliance standards.
        </Typography>
      </Alert>

      <Card>
        <CardContent>
          <TableContainer component={Paper}>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell>Patient ID</TableCell>
                  <TableCell>Name</TableCell>
                  <TableCell>Age</TableCell>
                  <TableCell>Contact</TableCell>
                  <TableCell>Status</TableCell>
                  <TableCell>Last Visit</TableCell>
                  <TableCell>Actions</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {patients.map((patient) => (
                  <TableRow key={patient.patient_id}>
                    <TableCell>{patient.patient_id}</TableCell>
                    <TableCell>
                      <Box display="flex" alignItems="center">
                        <Person sx={{ mr: 1 }} />
                        {patient.first_name} {patient.last_name}
                      </Box>
                    </TableCell>
                    <TableCell>{calculateAge(patient.date_of_birth)}</TableCell>
                    <TableCell>
                      <Box>
                        <Typography variant="body2" display="flex" alignItems="center">
                          <Phone sx={{ fontSize: 16, mr: 0.5 }} />
                          {patient.phone}
                        </Typography>
                        <Typography variant="body2" display="flex" alignItems="center">
                          <Email sx={{ fontSize: 16, mr: 0.5 }} />
                          {patient.email}
                        </Typography>
                      </Box>
                    </TableCell>
                    <TableCell>
                      <Chip
                        label={patient.status}
                        color={getStatusColor(patient.status)}
                        size="small"
                      />
                    </TableCell>
                    <TableCell>{patient.last_visit}</TableCell>
                    <TableCell>
                      <IconButton
                        color="primary"
                        onClick={() => {
                          setSelectedPatient(patient);
                          setDialogType('view');
                          setOpenDialog(true);
                        }}
                      >
                        <Visibility />
                      </IconButton>
                      <IconButton
                        color="secondary"
                        onClick={() => handleDiagnosePatient(patient)}
                        disabled={loading}
                      >
                        <LocalHospital />
                      </IconButton>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        </CardContent>
      </Card>

      {/* Add/Edit Patient Dialog */}
      <Dialog open={openDialog} onClose={() => setOpenDialog(false)} maxWidth="md" fullWidth>
        <DialogTitle>
          {dialogType === 'add' ? 'Add New Patient' :
           dialogType === 'edit' ? 'Edit Patient' : 'Patient Details'}
        </DialogTitle>
        <DialogContent>
          {dialogType === 'view' && selectedPatient ? (
            <Box>
              <Grid container spacing={2}>
                <Grid item xs={12} sm={6}>
                  <Typography variant="h6" gutterBottom>Personal Information</Typography>
                  <Typography><strong>Patient ID:</strong> {selectedPatient.patient_id}</Typography>
                  <Typography><strong>Name:</strong> {selectedPatient.first_name} {selectedPatient.last_name}</Typography>
                  <Typography><strong>Date of Birth:</strong> {selectedPatient.date_of_birth}</Typography>
                  <Typography><strong>Age:</strong> {calculateAge(selectedPatient.date_of_birth)}</Typography>
                </Grid>
                <Grid item xs={12} sm={6}>
                  <Typography variant="h6" gutterBottom>Contact Information</Typography>
                  <Typography><strong>Phone:</strong> {selectedPatient.phone}</Typography>
                  <Typography><strong>Email:</strong> {selectedPatient.email}</Typography>
                  <Typography><strong>Emergency Contact:</strong> {selectedPatient.emergency_contact}</Typography>
                </Grid>
                <Grid item xs={12}>
                  <Typography variant="h6" gutterBottom>Medical Information</Typography>
                  <Typography><strong>Medical History:</strong></Typography>
                  <Box display="flex" gap={1} mt={1}>
                    {selectedPatient.medical_history.map((condition, index) => (
                      <Chip key={index} label={condition} size="small" />
                    ))}
                  </Box>
                  <Typography sx={{ mt: 2 }}><strong>Current Symptoms:</strong> {selectedPatient.current_symptoms}</Typography>
                  <Typography><strong>Insurance ID:</strong> {selectedPatient.insurance_id}</Typography>
                </Grid>
              </Grid>
            </Box>
          ) : (
            <Grid container spacing={2}>
              <Grid item xs={12} sm={6}>
                <TextField
                  fullWidth
                  label="First Name"
                  value={newPatient.first_name}
                  onChange={(e) => setNewPatient({...newPatient, first_name: e.target.value})}
                  required
                />
              </Grid>
              <Grid item xs={12} sm={6}>
                <TextField
                  fullWidth
                  label="Last Name"
                  value={newPatient.last_name}
                  onChange={(e) => setNewPatient({...newPatient, last_name: e.target.value})}
                  required
                />
              </Grid>
              <Grid item xs={12} sm={6}>
                <TextField
                  fullWidth
                  label="Date of Birth"
                  type="date"
                  value={newPatient.date_of_birth}
                  onChange={(e) => setNewPatient({...newPatient, date_of_birth: e.target.value})}
                  InputLabelProps={{ shrink: true }}
                  required
                />
              </Grid>
              <Grid item xs={12} sm={6}>
                <TextField
                  fullWidth
                  label="Phone"
                  value={newPatient.phone}
                  onChange={(e) => setNewPatient({...newPatient, phone: e.target.value})}
                  required
                />
              </Grid>
              <Grid item xs={12} sm={6}>
                <TextField
                  fullWidth
                  label="Email"
                  type="email"
                  value={newPatient.email}
                  onChange={(e) => setNewPatient({...newPatient, email: e.target.value})}
                />
              </Grid>
              <Grid item xs={12} sm={6}>
                <TextField
                  fullWidth
                  label="Insurance ID"
                  value={newPatient.insurance_id}
                  onChange={(e) => setNewPatient({...newPatient, insurance_id: e.target.value})}
                />
              </Grid>
              <Grid item xs={12}>
                <TextField
                  fullWidth
                  label="Current Symptoms"
                  multiline
                  rows={3}
                  value={newPatient.current_symptoms}
                  onChange={(e) => setNewPatient({...newPatient, current_symptoms: e.target.value})}
                  placeholder="Describe current symptoms..."
                />
              </Grid>
              <Grid item xs={12}>
                <TextField
                  fullWidth
                  label="Emergency Contact"
                  value={newPatient.emergency_contact}
                  onChange={(e) => setNewPatient({...newPatient, emergency_contact: e.target.value})}
                />
              </Grid>
            </Grid>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setOpenDialog(false)}>
            {dialogType === 'view' ? 'Close' : 'Cancel'}
          </Button>
          {dialogType === 'add' && (
            <Button
              onClick={handleAddPatient}
              variant="contained"
              disabled={loading}
            >
              Add Patient
            </Button>
          )}
        </DialogActions>
      </Dialog>
    </Box>
  );
}

export default PatientManagement;
