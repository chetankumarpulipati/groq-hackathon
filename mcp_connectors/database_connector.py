"""
Database MCP Connector for healthcare data storage and retrieval.
Provides secure, HIPAA-compliant database operations for patient records using PostgreSQL.
"""

import asyncio
import json
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
import asyncpg
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, DateTime, Text, Boolean, Float, Index
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.dialects.postgresql import UUID, JSONB
import redis
import uuid
from config.settings import config
from utils.logging import get_logger
from utils.error_handling import MCPConnectionError, handle_exception

logger = get_logger("database_mcp")

# Database models with PostgreSQL optimizations
Base = declarative_base()


class Patient(Base):
    """Patient record model with PostgreSQL UUID primary key."""
    __tablename__ = 'patients'

    id = Column(Integer, primary_key=True)
    patient_id = Column(String(50), unique=True, nullable=False, index=True)
    first_name = Column(String(100), nullable=False)
    last_name = Column(String(100), nullable=False)
    date_of_birth = Column(DateTime, nullable=False)
    email = Column(String(255))
    phone = Column(String(20))
    insurance_id = Column(String(100))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = Column(Boolean, default=True)

    # PostgreSQL specific indexes
    __table_args__ = (
        Index('idx_patients_name', 'first_name', 'last_name'),
        Index('idx_patients_dob', 'date_of_birth'),
        Index('idx_patients_active', 'is_active'),
    )


class MedicalRecord(Base):
    """Medical record model with JSONB storage for flexible data."""
    __tablename__ = 'medical_records'

    id = Column(Integer, primary_key=True)
    patient_id = Column(String(50), nullable=False, index=True)
    record_type = Column(String(50), nullable=False)  # diagnosis, lab_result, imaging, etc.
    record_data = Column(JSONB, nullable=False)  # PostgreSQL JSONB for better performance
    provider_id = Column(String(50))
    created_at = Column(DateTime, default=datetime.utcnow)
    is_confidential = Column(Boolean, default=True)

    # PostgreSQL specific indexes
    __table_args__ = (
        Index('idx_medical_records_type_patient', 'record_type', 'patient_id'),
        Index('idx_medical_records_created', 'created_at'),
        Index('idx_medical_records_jsonb', 'record_data', postgresql_using='gin'),
    )


class DiagnosticSession(Base):
    """Diagnostic session model with JSONB for session data."""
    __tablename__ = 'diagnostic_sessions'

    id = Column(Integer, primary_key=True)
    session_id = Column(String(100), unique=True, nullable=False, index=True)
    patient_id = Column(String(50), nullable=False, index=True)
    agent_id = Column(String(100), nullable=False)
    session_data = Column(JSONB, nullable=False)  # PostgreSQL JSONB
    diagnosis_result = Column(JSONB)  # PostgreSQL JSONB
    confidence_score = Column(Float)
    requires_review = Column(Boolean, default=True)
    status = Column(String(20), default='active')  # active, completed, archived
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime)

    # PostgreSQL specific indexes
    __table_args__ = (
        Index('idx_diagnostic_sessions_status_patient', 'status', 'patient_id'),
        Index('idx_diagnostic_sessions_created', 'created_at'),
        Index('idx_diagnostic_sessions_confidence', 'confidence_score'),
    )


class DatabaseMCPConnector:
    """
    MCP Connector for PostgreSQL database operations with healthcare data.
    Provides secure, audited access to patient records and medical data.
    """

    def __init__(self):
        self.engine = None
        self.SessionLocal = None
        self.redis_client = None
        self.metadata = MetaData()
        self.async_pool = None
        self._initialize_connections()

        logger.info("PostgreSQL DatabaseMCPConnector initialized")

    def _initialize_connections(self):
        """Initialize PostgreSQL database and optionally Redis cache connections."""
        try:
            # Initialize SQLAlchemy engine for PostgreSQL
            self.engine = create_engine(
                config.database.url,
                echo=False,  # Set to True for SQL debugging
                pool_pre_ping=True,
                pool_recycle=3600,
                pool_size=20,  # Increased pool size for PostgreSQL
                max_overflow=30,
                # PostgreSQL specific settings
                connect_args={
                    "sslmode": "prefer",
                    "application_name": "healthcare_system"
                }
            )

            # Create session factory
            self.SessionLocal = sessionmaker(
                autocommit=False,
                autoflush=False,
                bind=self.engine
            )

            # Create tables
            Base.metadata.create_all(bind=self.engine)

            # Initialize async connection pool for high-performance queries
            asyncio.create_task(self._initialize_async_pool())

            # Try to initialize Redis for caching (optional)
            try:
                redis_url = config.database.redis_url
                if redis_url and redis_url != "redis://localhost:6379/0":
                    self.redis_client = redis.from_url(redis_url)
                    self.redis_client.ping()  # Test connection
                    logger.info("✅ Redis cache connection established")
                else:
                    logger.info("🔄 Redis URL not configured or using default - skipping Redis")
                    self.redis_client = None
            except Exception as e:
                logger.warning(f"⚠️  Redis connection failed: {e}. System will continue without caching.")
                self.redis_client = None

            logger.info("✅ PostgreSQL database connections initialized successfully")

        except Exception as e:
            logger.error(f"❌ Database initialization failed: {e}")
            raise MCPConnectionError(f"PostgreSQL connection failed: {str(e)}")

    async def _initialize_async_pool(self):
        """Initialize asyncpg connection pool for high-performance operations."""
        try:
            # Extract connection details from SQLAlchemy URL
            db_url = config.database.url
            if db_url.startswith('postgresql://'):
                # Convert to asyncpg format
                asyncpg_url = db_url.replace('postgresql://', 'postgresql://')
                self.async_pool = await asyncpg.create_pool(
                    asyncpg_url,
                    min_size=5,
                    max_size=20,
                    command_timeout=60
                )
                logger.info("AsyncPG connection pool initialized")
        except Exception as e:
            logger.warning(f"AsyncPG pool initialization failed: {e}")
            self.async_pool = None

    @handle_exception
    async def store_patient_record(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Store or update patient record in database."""

        session = self.SessionLocal()
        try:
            patient_id = patient_data.get("patient_id")

            # Check if patient exists
            existing_patient = session.query(Patient).filter(
                Patient.patient_id == patient_id
            ).first()

            if existing_patient:
                # Update existing patient
                for key, value in patient_data.items():
                    if hasattr(existing_patient, key) and key != 'id':
                        setattr(existing_patient, key, value)
                existing_patient.updated_at = datetime.utcnow()

                session.commit()
                operation = "updated"

            else:
                # Create new patient
                new_patient = Patient(**patient_data)
                session.add(new_patient)
                session.commit()
                operation = "created"

            # Invalidate cache
            if self.redis_client:
                await self._invalidate_patient_cache(patient_id)

            logger.info(f"Patient record {operation} for ID: {patient_id}")

            return {
                "success": True,
                "operation": operation,
                "patient_id": patient_id,
                "timestamp": datetime.utcnow().isoformat()
            }

        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Database error storing patient record: {e}")
            raise MCPConnectionError(f"Failed to store patient record: {str(e)}")
        finally:
            session.close()

    @handle_exception
    async def retrieve_patient_record(self, patient_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve patient record from database."""

        # Try cache first
        if self.redis_client:
            cached_data = await self._get_from_cache(f"patient:{patient_id}")
            if cached_data:
                logger.debug(f"Patient record retrieved from cache: {patient_id}")
                return cached_data

        session = self.SessionLocal()
        try:
            patient = session.query(Patient).filter(
                Patient.patient_id == patient_id,
                Patient.is_active == True
            ).first()

            if not patient:
                return None

            patient_data = {
                "patient_id": patient.patient_id,
                "first_name": patient.first_name,
                "last_name": patient.last_name,
                "date_of_birth": patient.date_of_birth.isoformat() if patient.date_of_birth else None,
                "email": patient.email,
                "phone": patient.phone,
                "insurance_id": patient.insurance_id,
                "created_at": patient.created_at.isoformat(),
                "updated_at": patient.updated_at.isoformat()
            }

            # Cache the result
            if self.redis_client:
                await self._store_in_cache(f"patient:{patient_id}", patient_data, ttl=3600)

            logger.debug(f"Patient record retrieved from database: {patient_id}")
            return patient_data

        except SQLAlchemyError as e:
            logger.error(f"Database error retrieving patient record: {e}")
            raise MCPConnectionError(f"Failed to retrieve patient record: {str(e)}")
        finally:
            session.close()

    @handle_exception
    async def store_medical_record(self, medical_record_data: Dict[str, Any]) -> Dict[str, Any]:
        """Store medical record (diagnosis, lab results, imaging, etc.)."""

        session = self.SessionLocal()
        try:
            medical_record = MedicalRecord(
                patient_id=medical_record_data["patient_id"],
                record_type=medical_record_data["record_type"],
                record_data=json.dumps(medical_record_data["record_data"]),
                provider_id=medical_record_data.get("provider_id"),
                is_confidential=medical_record_data.get("is_confidential", True)
            )

            session.add(medical_record)
            session.commit()

            record_id = medical_record.id

            # Invalidate related caches
            if self.redis_client:
                await self._invalidate_medical_records_cache(medical_record_data["patient_id"])

            logger.info(f"Medical record stored: ID {record_id}, Type: {medical_record_data['record_type']}")

            return {
                "success": True,
                "record_id": record_id,
                "patient_id": medical_record_data["patient_id"],
                "record_type": medical_record_data["record_type"],
                "timestamp": datetime.utcnow().isoformat()
            }

        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Database error storing medical record: {e}")
            raise MCPConnectionError(f"Failed to store medical record: {str(e)}")
        finally:
            session.close()

    @handle_exception
    async def retrieve_medical_records(
        self,
        patient_id: str,
        record_type: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Retrieve medical records for a patient."""

        # Try cache first
        cache_key = f"medical_records:{patient_id}:{record_type or 'all'}:{limit}"
        if self.redis_client:
            cached_data = await self._get_from_cache(cache_key)
            if cached_data:
                logger.debug(f"Medical records retrieved from cache: {patient_id}")
                return cached_data

        session = self.SessionLocal()
        try:
            query = session.query(MedicalRecord).filter(
                MedicalRecord.patient_id == patient_id
            )

            if record_type:
                query = query.filter(MedicalRecord.record_type == record_type)

            records = query.order_by(MedicalRecord.created_at.desc()).limit(limit).all()

            medical_records = []
            for record in records:
                medical_records.append({
                    "record_id": record.id,
                    "patient_id": record.patient_id,
                    "record_type": record.record_type,
                    "record_data": json.loads(record.record_data),
                    "provider_id": record.provider_id,
                    "created_at": record.created_at.isoformat(),
                    "is_confidential": record.is_confidential
                })

            # Cache the results
            if self.redis_client:
                await self._store_in_cache(cache_key, medical_records, ttl=1800)

            logger.debug(f"Retrieved {len(medical_records)} medical records for patient: {patient_id}")
            return medical_records

        except SQLAlchemyError as e:
            logger.error(f"Database error retrieving medical records: {e}")
            raise MCPConnectionError(f"Failed to retrieve medical records: {str(e)}")
        finally:
            session.close()

    @handle_exception
    async def store_diagnostic_session(self, session_data: Dict[str, Any]) -> Dict[str, Any]:
        """Store diagnostic session and results."""

        db_session = self.SessionLocal()
        try:
            diagnostic_session = DiagnosticSession(
                session_id=session_data["session_id"],
                patient_id=session_data["patient_id"],
                agent_id=session_data["agent_id"],
                session_data=json.dumps(session_data["session_data"]),
                diagnosis_result=json.dumps(session_data.get("diagnosis_result", {})),
                confidence_score=session_data.get("confidence_score"),
                requires_review=session_data.get("requires_review", True),
                status=session_data.get("status", "active")
            )

            if session_data.get("completed_at"):
                diagnostic_session.completed_at = datetime.fromisoformat(session_data["completed_at"])

            db_session.add(diagnostic_session)
            db_session.commit()

            session_id = diagnostic_session.id

            logger.info(f"Diagnostic session stored: {session_data['session_id']}")

            return {
                "success": True,
                "session_id": session_data["session_id"],
                "database_id": session_id,
                "patient_id": session_data["patient_id"],
                "timestamp": datetime.utcnow().isoformat()
            }

        except SQLAlchemyError as e:
            db_session.rollback()
            logger.error(f"Database error storing diagnostic session: {e}")
            raise MCPConnectionError(f"Failed to store diagnostic session: {str(e)}")
        finally:
            db_session.close()

    @handle_exception
    async def retrieve_diagnostic_sessions(
        self,
        patient_id: str,
        status: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Retrieve diagnostic sessions for a patient."""

        session = self.SessionLocal()
        try:
            query = session.query(DiagnosticSession).filter(
                DiagnosticSession.patient_id == patient_id
            )

            if status:
                query = query.filter(DiagnosticSession.status == status)

            sessions = query.order_by(DiagnosticSession.created_at.desc()).limit(limit).all()

            diagnostic_sessions = []
            for diag_session in sessions:
                diagnostic_sessions.append({
                    "database_id": diag_session.id,
                    "session_id": diag_session.session_id,
                    "patient_id": diag_session.patient_id,
                    "agent_id": diag_session.agent_id,
                    "session_data": json.loads(diag_session.session_data),
                    "diagnosis_result": json.loads(diag_session.diagnosis_result) if diag_session.diagnosis_result else None,
                    "confidence_score": diag_session.confidence_score,
                    "requires_review": diag_session.requires_review,
                    "status": diag_session.status,
                    "created_at": diag_session.created_at.isoformat(),
                    "completed_at": diag_session.completed_at.isoformat() if diag_session.completed_at else None
                })

            logger.debug(f"Retrieved {len(diagnostic_sessions)} diagnostic sessions for patient: {patient_id}")
            return diagnostic_sessions

        except SQLAlchemyError as e:
            logger.error(f"Database error retrieving diagnostic sessions: {e}")
            raise MCPConnectionError(f"Failed to retrieve diagnostic sessions: {str(e)}")
        finally:
            session.close()

    async def _get_from_cache(self, key: str) -> Optional[Dict[str, Any]]:
        """Get data from Redis cache - returns None if Redis not available."""
        if not self.redis_client:
            return None

        try:
            cached_data = self.redis_client.get(key)
            if cached_data:
                return json.loads(cached_data)
        except Exception as e:
            logger.debug(f"Cache retrieval failed for key {key}: {e}")

        return None

    async def _store_in_cache(self, key: str, data: Any, ttl: int = 3600):
        """Store data in Redis cache - does nothing if Redis not available."""
        if not self.redis_client:
            return

        try:
            self.redis_client.setex(key, ttl, json.dumps(data, default=str))
        except Exception as e:
            logger.debug(f"Cache storage failed for key {key}: {e}")

    async def _invalidate_patient_cache(self, patient_id: str):
        """Invalidate patient-related cache entries - does nothing if Redis not available."""
        if not self.redis_client:
            return

        try:
            pattern = f"patient:{patient_id}*"
            keys = self.redis_client.keys(pattern)
            if keys:
                self.redis_client.delete(*keys)
        except Exception as e:
            logger.debug(f"Cache invalidation failed for patient {patient_id}: {e}")

    async def _invalidate_medical_records_cache(self, patient_id: str):
        """Invalidate medical records cache for a patient - does nothing if Redis not available."""
        if not self.redis_client:
            return

        try:
            pattern = f"medical_records:{patient_id}*"
            keys = self.redis_client.keys(pattern)
            if keys:
                self.redis_client.delete(*keys)
        except Exception as e:
            logger.debug(f"Medical records cache invalidation failed for patient {patient_id}: {e}")

    async def health_check(self) -> Dict[str, Any]:
        """Check database and cache connectivity."""

        health_status = {
            "database": {"status": "unknown", "details": ""},
            "cache": {"status": "unknown", "details": ""},
            "overall": "unknown"
        }

        # Test database connection
        try:
            session = self.SessionLocal()
            session.execute("SELECT 1")
            session.close()
            health_status["database"] = {"status": "healthy", "details": "PostgreSQL connection successful"}
        except Exception as e:
            health_status["database"] = {"status": "unhealthy", "details": str(e)}

        # Test cache connection
        if self.redis_client:
            try:
                self.redis_client.ping()
                health_status["cache"] = {"status": "healthy", "details": "Redis connection successful"}
            except Exception as e:
                health_status["cache"] = {"status": "unhealthy", "details": str(e)}
        else:
            health_status["cache"] = {"status": "disabled", "details": "Redis not configured - using in-memory fallback"}

        # Overall status - system is healthy even without Redis
        if health_status["database"]["status"] == "healthy":
            health_status["overall"] = "healthy"
        else:
            health_status["overall"] = "unhealthy"

        return health_status

    async def get_connection_info(self) -> Dict[str, Any]:
        """Get database connection information."""

        return {
            "database_url": config.database.url.split('@')[-1] if '@' in config.database.url else "Not configured",
            "redis_url": config.database.redis_url.split('@')[-1] if '@' in config.database.redis_url else "Not configured",
            "connection_pool_size": self.engine.pool.size() if self.engine else 0,
            "tables": ["patients", "medical_records", "diagnostic_sessions"],
            "cache_enabled": self.redis_client is not None
        }

    @handle_exception
    async def advanced_patient_search(self, search_params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Advanced patient search using PostgreSQL full-text search and JSONB queries."""

        if not self.async_pool:
            # Fallback to basic search
            return await self._basic_patient_search(search_params)

        try:
            async with self.async_pool.acquire() as connection:
                # Build dynamic query based on search parameters
                base_query = """
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
                """

                conditions = []
                params = []
                param_count = 1

                # Add search conditions
                if search_params.get('name'):
                    conditions.append(f"""
                        (LOWER(p.first_name) LIKE LOWER($${param_count}) 
                         OR LOWER(p.last_name) LIKE LOWER($${param_count}))
                    """)
                    params.append(f"%{search_params['name']}%")
                    param_count += 1

                if search_params.get('patient_id'):
                    conditions.append(f"p.patient_id LIKE $${param_count}")
                    params.append(f"%{search_params['patient_id']}%")
                    param_count += 1

                if search_params.get('email'):
                    conditions.append(f"LOWER(p.email) LIKE LOWER($${param_count})")
                    params.append(f"%{search_params['email']}%")
                    param_count += 1

                if search_params.get('age_range'):
                    min_age, max_age = search_params['age_range']
                    conditions.append(f"""
                        EXTRACT(YEAR FROM AGE(p.date_of_birth)) 
                        BETWEEN $${param_count} AND $${param_count + 1}
                    """)
                    params.extend([min_age, max_age])
                    param_count += 2

                # Add conditions to query
                if conditions:
                    base_query += " AND " + " AND ".join(conditions)

                base_query += """
                GROUP BY p.patient_id, p.first_name, p.last_name, p.date_of_birth, p.email, p.phone
                ORDER BY p.last_name, p.first_name
                LIMIT 100
                """

                rows = await connection.fetch(base_query, *params)

                results = []
                for row in rows:
                    results.append({
                        "patient_id": row['patient_id'],
                        "full_name": f"{row['first_name']} {row['last_name']}",
                        "age": row['age'],
                        "email": row['email'],
                        "phone": row['phone'],
                        "total_records": row['total_records'],
                        "last_record_date": row['last_record_date'].isoformat() if row['last_record_date'] else None
                    })

                logger.debug(f"Advanced search returned {len(results)} patients")
                return results

        except Exception as e:
            logger.error(f"Advanced patient search failed: {e}")
            raise MCPConnectionError(f"Patient search failed: {str(e)}")

    @handle_exception
    async def jsonb_medical_record_query(self, patient_id: str, query_params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Query medical records using PostgreSQL JSONB operators for complex searches."""

        if not self.async_pool:
            return await self.retrieve_medical_records(patient_id)

        try:
            async with self.async_pool.acquire() as connection:
                base_query = """
                SELECT 
                    id,
                    patient_id,
                    record_type,
                    record_data,
                    provider_id,
                    created_at
                FROM medical_records 
                WHERE patient_id = $1
                """

                params = [patient_id]
                param_count = 2
                conditions = []

                # JSONB queries
                if query_params.get('condition'):
                    conditions.append(f"record_data->>'condition' = ${param_count}")
                    params.append(query_params['condition'])
                    param_count += 1

                if query_params.get('medication'):
                    conditions.append(f"record_data->>'name' = ${param_count}")
                    params.append(query_params['medication'])
                    param_count += 1

                if query_params.get('test_name'):
                    conditions.append(f"record_data->>'test' = ${param_count}")
                    params.append(query_params['test_name'])
                    param_count += 1

                if query_params.get('date_range'):
                    start_date, end_date = query_params['date_range']
                    conditions.append(f"""
                        (record_data->>'date')::date 
                        BETWEEN ${param_count}::date AND ${param_count + 1}::date
                    """)
                    params.extend([start_date, end_date])
                    param_count += 2

                # Add record type filter
                if query_params.get('record_types'):
                    placeholders = ','.join([f'${i}' for i in range(param_count, param_count + len(query_params['record_types']))])
                    conditions.append(f"record_type IN ({placeholders})")
                    params.extend(query_params['record_types'])

                if conditions:
                    base_query += " AND " + " AND ".join(conditions)

                base_query += " ORDER BY created_at DESC LIMIT 100"

                rows = await connection.fetch(base_query, *params)

                results = []
                for row in rows:
                    results.append({
                        "record_id": row['id'],
                        "patient_id": row['patient_id'],
                        "record_type": row['record_type'],
                        "record_data": row['record_data'],
                        "provider_id": row['provider_id'],
                        "created_at": row['created_at'].isoformat()
                    })

                logger.debug(f"JSONB query returned {len(results)} records")
                return results

        except Exception as e:
            logger.error(f"JSONB medical record query failed: {e}")
            raise MCPConnectionError(f"Medical record query failed: {str(e)}")

    @handle_exception
    async def get_patient_analytics(self, patient_id: str) -> Dict[str, Any]:
        """Get comprehensive patient analytics using PostgreSQL aggregation functions."""

        if not self.async_pool:
            return {"error": "Analytics require PostgreSQL async pool"}

        try:
            async with self.async_pool.acquire() as connection:
                # Complex analytics query
                analytics_query = """
                WITH patient_stats AS (
                    SELECT 
                        p.patient_id,
                        p.first_name,
                        p.last_name,
                        EXTRACT(YEAR FROM AGE(p.date_of_birth)) as age,
                        COUNT(mr.id) as total_records,
                        COUNT(DISTINCT mr.record_type) as record_types,
                        MIN(mr.created_at) as first_record,
                        MAX(mr.created_at) as last_record,
                        COUNT(CASE WHEN mr.record_type = 'diagnosis' THEN 1 END) as diagnoses_count,
                        COUNT(CASE WHEN mr.record_type = 'lab_result' THEN 1 END) as lab_results_count,
                        COUNT(CASE WHEN mr.record_type = 'medication' THEN 1 END) as medications_count,
                        COUNT(CASE WHEN mr.record_type = 'vital_signs' THEN 1 END) as vital_signs_count
                    FROM patients p
                    LEFT JOIN medical_records mr ON p.patient_id = mr.patient_id
                    WHERE p.patient_id = $1 AND p.is_active = true
                    GROUP BY p.patient_id, p.first_name, p.last_name, p.date_of_birth
                ),
                recent_activity AS (
                    SELECT 
                        record_type,
                        COUNT(*) as count,
                        MAX(created_at) as last_activity
                    FROM medical_records 
                    WHERE patient_id = $1 
                    AND created_at >= NOW() - INTERVAL '30 days'
                    GROUP BY record_type
                ),
                diagnostic_sessions_stats AS (
                    SELECT 
                        COUNT(*) as total_sessions,
                        COUNT(CASE WHEN status = 'completed' THEN 1 END) as completed_sessions,
                        ROUND(AVG(confidence_score), 2) as avg_confidence,
                        MAX(created_at) as last_session
                    FROM diagnostic_sessions 
                    WHERE patient_id = $1
                )
                SELECT 
                    ps.*,
                    json_agg(
                        json_build_object(
                            'record_type', ra.record_type,
                            'recent_count', ra.count,
                            'last_activity', ra.last_activity
                        )
                    ) FILTER (WHERE ra.record_type IS NOT NULL) as recent_activity,
                    dss.total_sessions,
                    dss.completed_sessions,
                    dss.avg_confidence,
                    dss.last_session
                FROM patient_stats ps
                LEFT JOIN recent_activity ra ON true
                CROSS JOIN diagnostic_sessions_stats dss
                GROUP BY ps.patient_id, ps.first_name, ps.last_name, ps.age, 
                         ps.total_records, ps.record_types, ps.first_record, ps.last_record,
                         ps.diagnoses_count, ps.lab_results_count, ps.medications_count, ps.vital_signs_count,
                         dss.total_sessions, dss.completed_sessions, dss.avg_confidence, dss.last_session
                """

                row = await connection.fetchrow(analytics_query, patient_id)

                if not row:
                    return {"error": "Patient not found"}

                analytics = {
                    "patient_info": {
                        "patient_id": row['patient_id'],
                        "name": f"{row['first_name']} {row['last_name']}",
                        "age": row['age']
                    },
                    "record_summary": {
                        "total_records": row['total_records'],
                        "record_types": row['record_types'],
                        "first_record": row['first_record'].isoformat() if row['first_record'] else None,
                        "last_record": row['last_record'].isoformat() if row['last_record'] else None,
                        "diagnoses_count": row['diagnoses_count'],
                        "lab_results_count": row['lab_results_count'],
                        "medications_count": row['medications_count'],
                        "vital_signs_count": row['vital_signs_count']
                    },
                    "recent_activity": row['recent_activity'] or [],
                    "diagnostic_sessions": {
                        "total_sessions": row['total_sessions'],
                        "completed_sessions": row['completed_sessions'],
                        "avg_confidence": row['avg_confidence'],
                        "last_session": row['last_session'].isoformat() if row['last_session'] else None
                    },
                    "generated_at": datetime.utcnow().isoformat()
                }

                return analytics

        except Exception as e:
            logger.error(f"Patient analytics query failed: {e}")
            raise MCPConnectionError(f"Analytics query failed: {str(e)}")

    async def execute_custom_query(self, query: str, params: List[Any] = None) -> List[Dict[str, Any]]:
        """Execute custom PostgreSQL queries safely."""

        if not self.async_pool:
            raise MCPConnectionError("Custom queries require PostgreSQL async pool")

        try:
            async with self.async_pool.acquire() as connection:
                rows = await connection.fetch(query, *(params or []))

                results = []
                for row in rows:
                    results.append(dict(row))

                return results

        except Exception as e:
            logger.error(f"Custom query execution failed: {e}")
            raise MCPConnectionError(f"Custom query failed: {str(e)}")

    async def get_database_stats(self) -> Dict[str, Any]:
        """Get comprehensive database statistics."""

        if not self.async_pool:
            return {"error": "Database stats require PostgreSQL async pool"}

        try:
            async with self.async_pool.acquire() as connection:
                stats_query = """
                SELECT 
                    'patients' as table_name,
                    COUNT(*) as total_rows,
                    COUNT(CASE WHEN created_at >= NOW() - INTERVAL '24 hours' THEN 1 END) as recent_24h,
                    COUNT(CASE WHEN created_at >= NOW() - INTERVAL '7 days' THEN 1 END) as recent_7d,
                    MIN(created_at) as oldest_record,
                    MAX(created_at) as newest_record
                FROM patients
                WHERE is_active = true
                
                UNION ALL
                
                SELECT 
                    'medical_records' as table_name,
                    COUNT(*) as total_rows,
                    COUNT(CASE WHEN created_at >= NOW() - INTERVAL '24 hours' THEN 1 END) as recent_24h,
                    COUNT(CASE WHEN created_at >= NOW() - INTERVAL '7 days' THEN 1 END) as recent_7d,
                    MIN(created_at) as oldest_record,
                    MAX(created_at) as newest_record
                FROM medical_records
                
                UNION ALL
                
                SELECT 
                    'diagnostic_sessions' as table_name,
                    COUNT(*) as total_rows,
                    COUNT(CASE WHEN created_at >= NOW() - INTERVAL '24 hours' THEN 1 END) as recent_24h,
                    COUNT(CASE WHEN created_at >= NOW() - INTERVAL '7 days' THEN 1 END) as recent_7d,
                    MIN(created_at) as oldest_record,
                    MAX(created_at) as newest_record
                FROM diagnostic_sessions
                """

                rows = await connection.fetch(stats_query)

                stats = {}
                for row in rows:
                    stats[row['table_name']] = {
                        "total_rows": row['total_rows'],
                        "recent_24h": row['recent_24h'],
                        "recent_7d": row['recent_7d'],
                        "oldest_record": row['oldest_record'].isoformat() if row['oldest_record'] else None,
                        "newest_record": row['newest_record'].isoformat() if row['newest_record'] else None
                    }

                return {
                    "database_type": "PostgreSQL",
                    "table_stats": stats,
                    "generated_at": datetime.utcnow().isoformat()
                }

        except Exception as e:
            logger.error(f"Database stats query failed: {e}")
            return {"error": str(e)}
