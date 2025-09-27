import React, { useState } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { AppBar, Toolbar, Typography, Drawer, List, ListItem, ListItemIcon, ListItemText, Box, Container } from '@mui/material';
import { Dashboard, People, Assignment, Assessment, Mic, Visibility } from '@mui/icons-material';
import DashboardHome from './components/DashboardHome';
import PatientManagement from './components/PatientManagement';
import DiagnosticCenter from './components/DiagnosticCenter';
import ModelAccuracy from './components/ModelAccuracy';
import VoiceProcessing from './components/VoiceProcessing';
import VisionAnalysis from './components/VisionAnalysis';

const drawerWidth = 240;

function App() {
  const [selectedTab, setSelectedTab] = useState(0);

  const menuItems = [
    { text: 'Dashboard', icon: <Dashboard />, component: <DashboardHome /> },
    { text: 'Patients', icon: <People />, component: <PatientManagement /> },
    { text: 'Diagnostics', icon: <Assignment />, component: <DiagnosticCenter /> },
    { text: 'AI Accuracy', icon: <Assessment />, component: <ModelAccuracy /> },
    { text: 'Voice Processing', icon: <Mic />, component: <VoiceProcessing /> },
    { text: 'Vision Analysis', icon: <Visibility />, component: <VisionAnalysis /> }
  ];

  return (
    <Router>
      <Box sx={{ display: 'flex' }}>
        <AppBar position="fixed" sx={{ zIndex: (theme) => theme.zIndex.drawer + 1 }}>
          <Toolbar>
            <Typography variant="h6" noWrap component="div">
              Healthcare AI System Dashboard
            </Typography>
          </Toolbar>
        </AppBar>

        <Drawer
          variant="permanent"
          sx={{
            width: drawerWidth,
            flexShrink: 0,
            [`& .MuiDrawer-paper`]: { width: drawerWidth, boxSizing: 'border-box' },
          }}
        >
          <Toolbar />
          <Box sx={{ overflow: 'auto' }}>
            <List>
              {menuItems.map((item, index) => (
                <ListItem
                  button
                  key={item.text}
                  selected={selectedTab === index}
                  onClick={() => setSelectedTab(index)}
                >
                  <ListItemIcon>{item.icon}</ListItemIcon>
                  <ListItemText primary={item.text} />
                </ListItem>
              ))}
            </List>
          </Box>
        </Drawer>

        <Box component="main" sx={{ flexGrow: 1, p: 3 }}>
          <Toolbar />
          <Container maxWidth="xl">
            {menuItems[selectedTab].component}
          </Container>
        </Box>
      </Box>
    </Router>
  );
}

export default App;
