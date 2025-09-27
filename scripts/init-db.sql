-- Initialize Healthcare Database Schema
-- This script sets up the initial database structure and sample data

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create additional indexes for performance
CREATE INDEX IF NOT EXISTS idx_patients_patient_id ON patients(patient_id);
CREATE INDEX IF NOT EXISTS idx_medical_records_patient_id ON medical_records(patient_id);
CREATE INDEX IF NOT EXISTS idx_medical_records_type ON medical_records(record_type);
CREATE INDEX IF NOT EXISTS idx_diagnostic_sessions_patient_id ON diagnostic_sessions(patient_id);
CREATE INDEX IF NOT EXISTS idx_diagnostic_sessions_status ON diagnostic_sessions(status);

-- Insert sample patient data for testing
INSERT INTO patients (patient_id, first_name, last_name, date_of_birth, email, phone, insurance_id) VALUES
('P001', 'John', 'Doe', '1980-01-15', 'john.doe@email.com', '555-0123', 'INS001'),
('P002', 'Jane', 'Smith', '1975-06-22', 'jane.smith@email.com', '555-0124', 'INS002'),
('P003', 'Michael', 'Johnson', '1990-03-10', 'michael.j@email.com', '555-0125', 'INS003'),
('P004', 'Sarah', 'Williams', '1985-11-05', 'sarah.w@email.com', '555-0126', 'INS004'),
('P005', 'David', 'Brown', '1970-09-18', 'david.brown@email.com', '555-0127', 'INS005')
ON CONFLICT (patient_id) DO NOTHING;

-- Insert sample medical records
INSERT INTO medical_records (patient_id, record_type, record_data, provider_id) VALUES
('P001', 'diagnosis', '{"condition": "hypertension", "date": "2025-01-15", "severity": "mild", "status": "active"}', 'DR001'),
('P001', 'lab_result', '{"test": "lipid_panel", "cholesterol": 220, "hdl": 45, "ldl": 140, "date": "2025-01-10"}', 'LAB001'),
('P002', 'diagnosis', '{"condition": "diabetes_type2", "date": "2024-12-01", "hba1c": 7.2, "status": "active"}', 'DR002'),
('P002', 'medication', '{"name": "metformin", "dosage": "500mg", "frequency": "twice_daily", "start_date": "2024-12-01"}', 'DR002'),
('P003', 'vital_signs', '{"bp_systolic": 120, "bp_diastolic": 80, "heart_rate": 72, "temperature": 98.6, "date": "2025-01-20"}', 'NUR001')
ON CONFLICT DO NOTHING;

-- Create views for common queries
CREATE OR REPLACE VIEW patient_summary AS
SELECT
    p.patient_id,
    p.first_name,
    p.last_name,
    p.date_of_birth,
    EXTRACT(YEAR FROM AGE(p.date_of_birth)) as age,
    p.email,
    p.phone,
    COUNT(mr.id) as total_records,
    MAX(mr.created_at) as last_record_date
FROM patients p
LEFT JOIN medical_records mr ON p.patient_id = mr.patient_id
WHERE p.is_active = true
GROUP BY p.patient_id, p.first_name, p.last_name, p.date_of_birth, p.email, p.phone;

-- Create function for patient search
CREATE OR REPLACE FUNCTION search_patients(search_term TEXT)
RETURNS TABLE(
    patient_id VARCHAR,
    full_name TEXT,
    email VARCHAR,
    phone VARCHAR,
    age INTEGER
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        p.patient_id,
        CONCAT(p.first_name, ' ', p.last_name) as full_name,
        p.email,
        p.phone,
        EXTRACT(YEAR FROM AGE(p.date_of_birth))::INTEGER as age
    FROM patients p
    WHERE p.is_active = true
    AND (
        LOWER(p.first_name) LIKE LOWER('%' || search_term || '%')
        OR LOWER(p.last_name) LIKE LOWER('%' || search_term || '%')
        OR p.patient_id LIKE '%' || search_term || '%'
        OR p.email LIKE '%' || search_term || '%'
    )
    ORDER BY p.last_name, p.first_name;
END;
$$ LANGUAGE plpgsql;
