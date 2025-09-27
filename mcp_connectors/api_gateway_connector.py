"""
API Gateway MCP Connector for external healthcare system integration.
Provides secure API interfaces for EHR systems, lab systems, and other healthcare services.
"""

import asyncio
import json
import aiohttp
from typing import Dict, Any, Optional, List, Union
from datetime import datetime, timedelta
import jwt
from cryptography.fernet import Fernet
from config.settings import config
from utils.logging import get_logger
from utils.error_handling import MCPConnectionError, handle_exception

logger = get_logger("api_gateway_mcp")


class APIGatewayMCPConnector:
    """
    MCP Connector for external API integration with healthcare systems.
    Handles authentication, data transformation, and secure communication.
    """

    def __init__(self):
        self.session = None
        self.api_endpoints = self._initialize_api_endpoints()
        self.authentication_tokens = {}
        self.encryption_key = self._get_encryption_key()
        self._initialize_session()

        logger.info("APIGatewayMCPConnector initialized")

    def _initialize_api_endpoints(self) -> Dict[str, Dict[str, Any]]:
        """Initialize external API endpoint configurations."""
        return {
            "ehr_system": {
                "base_url": "https://api.ehrsystem.com/v1",
                "auth_type": "oauth2",
                "auth_endpoint": "/auth/token",
                "endpoints": {
                    "patient_lookup": "/patients/{patient_id}",
                    "create_patient": "/patients",
                    "update_patient": "/patients/{patient_id}",
                    "get_medical_history": "/patients/{patient_id}/history",
                    "create_encounter": "/encounters",
                    "get_appointments": "/appointments"
                },
                "timeout": 30,
                "retry_attempts": 3
            },
            "lab_system": {
                "base_url": "https://api.labsystem.com/v2",
                "auth_type": "api_key",
                "auth_header": "X-API-Key",
                "endpoints": {
                    "submit_order": "/orders",
                    "get_results": "/results/{order_id}",
                    "get_patient_results": "/patients/{patient_id}/results",
                    "reference_ranges": "/reference-ranges"
                },
                "timeout": 45,
                "retry_attempts": 2
            },
            "imaging_system": {
                "base_url": "https://api.imagingsystem.com/v1",
                "auth_type": "jwt",
                "auth_endpoint": "/auth/login",
                "endpoints": {
                    "upload_image": "/images/upload",
                    "get_image": "/images/{image_id}",
                    "request_interpretation": "/interpretations",
                    "get_interpretation": "/interpretations/{interpretation_id}"
                },
                "timeout": 60,
                "retry_attempts": 3
            },
            "pharmacy_system": {
                "base_url": "https://api.pharmacy.com/v1",
                "auth_type": "basic",
                "endpoints": {
                    "check_availability": "/medications/{ndc}/availability",
                    "submit_prescription": "/prescriptions",
                    "get_prescription_status": "/prescriptions/{rx_id}/status",
                    "drug_interactions": "/interactions/check"
                },
                "timeout": 30,
                "retry_attempts": 2
            },
            "insurance_system": {
                "base_url": "https://api.insurance.com/v1",
                "auth_type": "oauth2",
                "auth_endpoint": "/oauth/token",
                "endpoints": {
                    "verify_eligibility": "/eligibility/{member_id}",
                    "get_coverage": "/coverage/{member_id}",
                    "submit_claim": "/claims",
                    "get_claim_status": "/claims/{claim_id}/status"
                },
                "timeout": 30,
                "retry_attempts": 3
            }
        }

    def _get_encryption_key(self) -> Fernet:
        """Get or generate encryption key for sensitive data."""
        encryption_key = config.api.secret_key.encode()
        # Use first 32 bytes and pad if necessary
        key = encryption_key[:32].ljust(32, b'0')
        # Convert to base64 for Fernet
        import base64
        fernet_key = base64.urlsafe_b64encode(key)
        return Fernet(fernet_key)

    async def _initialize_session(self):
        """Initialize HTTP session with appropriate configurations."""
        timeout = aiohttp.ClientTimeout(total=60)
        self.session = aiohttp.ClientSession(
            timeout=timeout,
            headers={
                "User-Agent": "HealthcareAI-System/1.0",
                "Content-Type": "application/json"
            }
        )
        logger.info("HTTP session initialized")

    @handle_exception
    async def authenticate_system(self, system_name: str, credentials: Dict[str, Any]) -> Dict[str, Any]:
        """Authenticate with external healthcare system."""

        if system_name not in self.api_endpoints:
            raise MCPConnectionError(f"Unknown system: {system_name}")

        system_config = self.api_endpoints[system_name]
        auth_type = system_config["auth_type"]

        try:
            if auth_type == "oauth2":
                token_data = await self._authenticate_oauth2(system_name, credentials)
            elif auth_type == "jwt":
                token_data = await self._authenticate_jwt(system_name, credentials)
            elif auth_type == "api_key":
                token_data = await self._authenticate_api_key(system_name, credentials)
            elif auth_type == "basic":
                token_data = await self._authenticate_basic(system_name, credentials)
            else:
                raise MCPConnectionError(f"Unsupported auth type: {auth_type}")

            # Store encrypted token
            encrypted_token = self.encryption_key.encrypt(json.dumps(token_data).encode())
            self.authentication_tokens[system_name] = {
                "encrypted_token": encrypted_token,
                "expires_at": datetime.utcnow() + timedelta(hours=1),
                "auth_type": auth_type
            }

            logger.info(f"Successfully authenticated with {system_name}")

            return {
                "success": True,
                "system": system_name,
                "auth_type": auth_type,
                "expires_at": self.authentication_tokens[system_name]["expires_at"].isoformat()
            }

        except Exception as e:
            logger.error(f"Authentication failed for {system_name}: {e}")
            raise MCPConnectionError(f"Authentication failed: {str(e)}")

    async def _authenticate_oauth2(self, system_name: str, credentials: Dict[str, Any]) -> Dict[str, Any]:
        """Authenticate using OAuth2 flow."""

        system_config = self.api_endpoints[system_name]
        auth_url = system_config["base_url"] + system_config["auth_endpoint"]

        auth_data = {
            "grant_type": "client_credentials",
            "client_id": credentials["client_id"],
            "client_secret": credentials["client_secret"],
            "scope": credentials.get("scope", "read write")
        }

        async with self.session.post(auth_url, data=auth_data) as response:
            if response.status == 200:
                token_data = await response.json()
                return {
                    "access_token": token_data["access_token"],
                    "token_type": token_data.get("token_type", "Bearer"),
                    "expires_in": token_data.get("expires_in", 3600)
                }
            else:
                raise MCPConnectionError(f"OAuth2 authentication failed: {response.status}")

    async def _authenticate_jwt(self, system_name: str, credentials: Dict[str, Any]) -> Dict[str, Any]:
        """Authenticate using JWT tokens."""

        system_config = self.api_endpoints[system_name]
        auth_url = system_config["base_url"] + system_config["auth_endpoint"]

        login_data = {
            "username": credentials["username"],
            "password": credentials["password"]
        }

        async with self.session.post(auth_url, json=login_data) as response:
            if response.status == 200:
                auth_response = await response.json()
                return {
                    "jwt_token": auth_response["token"],
                    "refresh_token": auth_response.get("refresh_token"),
                    "expires_in": auth_response.get("expires_in", 3600)
                }
            else:
                raise MCPConnectionError(f"JWT authentication failed: {response.status}")

    async def _authenticate_api_key(self, system_name: str, credentials: Dict[str, Any]) -> Dict[str, Any]:
        """Authenticate using API key."""

        return {
            "api_key": credentials["api_key"],
            "expires_in": 86400  # 24 hours
        }

    async def _authenticate_basic(self, system_name: str, credentials: Dict[str, Any]) -> Dict[str, Any]:
        """Authenticate using basic authentication."""

        import base64
        auth_string = f"{credentials['username']}:{credentials['password']}"
        encoded_auth = base64.b64encode(auth_string.encode()).decode()

        return {
            "basic_auth": encoded_auth,
            "expires_in": 3600
        }

    @handle_exception
    async def make_api_request(
        self,
        system_name: str,
        endpoint: str,
        method: str = "GET",
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        path_params: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Make authenticated API request to external system."""

        # Check authentication
        if not await self._is_authenticated(system_name):
            raise MCPConnectionError(f"Not authenticated with {system_name}")

        # Build request URL
        system_config = self.api_endpoints[system_name]
        base_url = system_config["base_url"]
        endpoint_path = system_config["endpoints"].get(endpoint)

        if not endpoint_path:
            raise MCPConnectionError(f"Unknown endpoint: {endpoint}")

        # Replace path parameters
        if path_params:
            for param, value in path_params.items():
                endpoint_path = endpoint_path.replace(f"{{{param}}}", str(value))

        url = base_url + endpoint_path

        # Prepare headers
        headers = await self._get_auth_headers(system_name)

        # Make request with retry logic
        retry_attempts = system_config.get("retry_attempts", 3)
        timeout = system_config.get("timeout", 30)

        for attempt in range(retry_attempts):
            try:
                timeout_config = aiohttp.ClientTimeout(total=timeout)

                async with self.session.request(
                    method,
                    url,
                    json=data if method in ["POST", "PUT", "PATCH"] else None,
                    params=params,
                    headers=headers,
                    timeout=timeout_config
                ) as response:

                    if response.status == 200:
                        response_data = await response.json()

                        logger.debug(f"API request successful: {system_name}/{endpoint}")

                        return {
                            "success": True,
                            "data": response_data,
                            "status_code": response.status,
                            "system": system_name,
                            "endpoint": endpoint,
                            "timestamp": datetime.utcnow().isoformat()
                        }

                    elif response.status == 401:
                        # Authentication expired, try to refresh
                        logger.warning(f"Authentication expired for {system_name}, attempting refresh")
                        # Remove expired token
                        if system_name in self.authentication_tokens:
                            del self.authentication_tokens[system_name]
                        raise MCPConnectionError("Authentication expired")

                    else:
                        error_text = await response.text()
                        logger.error(f"API request failed: {response.status} - {error_text}")

                        if attempt == retry_attempts - 1:
                            return {
                                "success": False,
                                "error": f"API request failed: {response.status}",
                                "error_details": error_text,
                                "status_code": response.status,
                                "system": system_name,
                                "endpoint": endpoint,
                                "timestamp": datetime.utcnow().isoformat()
                            }

                        # Wait before retry
                        await asyncio.sleep(2 ** attempt)

            except asyncio.TimeoutError:
                logger.warning(f"API request timeout for {system_name}/{endpoint} (attempt {attempt + 1})")
                if attempt == retry_attempts - 1:
                    raise MCPConnectionError(f"API request timeout after {retry_attempts} attempts")
                await asyncio.sleep(2 ** attempt)

            except Exception as e:
                logger.error(f"API request error: {e}")
                if attempt == retry_attempts - 1:
                    raise MCPConnectionError(f"API request failed: {str(e)}")
                await asyncio.sleep(2 ** attempt)

    async def _is_authenticated(self, system_name: str) -> bool:
        """Check if system is authenticated and token is valid."""

        if system_name not in self.authentication_tokens:
            return False

        token_info = self.authentication_tokens[system_name]

        # Check if token is expired
        if datetime.utcnow() >= token_info["expires_at"]:
            del self.authentication_tokens[system_name]
            return False

        return True

    async def _get_auth_headers(self, system_name: str) -> Dict[str, str]:
        """Get authentication headers for API requests."""

        if system_name not in self.authentication_tokens:
            raise MCPConnectionError(f"No authentication token for {system_name}")

        token_info = self.authentication_tokens[system_name]
        auth_type = token_info["auth_type"]

        # Decrypt token
        decrypted_token = json.loads(
            self.encryption_key.decrypt(token_info["encrypted_token"]).decode()
        )

        headers = {}

        if auth_type == "oauth2":
            headers["Authorization"] = f"{decrypted_token['token_type']} {decrypted_token['access_token']}"

        elif auth_type == "jwt":
            headers["Authorization"] = f"Bearer {decrypted_token['jwt_token']}"

        elif auth_type == "api_key":
            system_config = self.api_endpoints[system_name]
            auth_header = system_config.get("auth_header", "X-API-Key")
            headers[auth_header] = decrypted_token["api_key"]

        elif auth_type == "basic":
            headers["Authorization"] = f"Basic {decrypted_token['basic_auth']}"

        return headers

    # Specialized healthcare API methods

    async def lookup_patient_in_ehr(self, patient_id: str, system_name: str = "ehr_system") -> Dict[str, Any]:
        """Look up patient information in EHR system."""

        return await self.make_api_request(
            system_name=system_name,
            endpoint="patient_lookup",
            method="GET",
            path_params={"patient_id": patient_id}
        )

    async def create_patient_in_ehr(self, patient_data: Dict[str, Any], system_name: str = "ehr_system") -> Dict[str, Any]:
        """Create new patient record in EHR system."""

        return await self.make_api_request(
            system_name=system_name,
            endpoint="create_patient",
            method="POST",
            data=patient_data
        )

    async def submit_lab_order(self, order_data: Dict[str, Any], system_name: str = "lab_system") -> Dict[str, Any]:
        """Submit lab order to laboratory system."""

        return await self.make_api_request(
            system_name=system_name,
            endpoint="submit_order",
            method="POST",
            data=order_data
        )

    async def get_lab_results(self, order_id: str, system_name: str = "lab_system") -> Dict[str, Any]:
        """Retrieve lab results from laboratory system."""

        return await self.make_api_request(
            system_name=system_name,
            endpoint="get_results",
            method="GET",
            path_params={"order_id": order_id}
        )

    async def submit_prescription(self, prescription_data: Dict[str, Any], system_name: str = "pharmacy_system") -> Dict[str, Any]:
        """Submit prescription to pharmacy system."""

        return await self.make_api_request(
            system_name=system_name,
            endpoint="submit_prescription",
            method="POST",
            data=prescription_data
        )

    async def check_drug_interactions(self, medications: List[str], system_name: str = "pharmacy_system") -> Dict[str, Any]:
        """Check for drug interactions."""

        return await self.make_api_request(
            system_name=system_name,
            endpoint="drug_interactions",
            method="POST",
            data={"medications": medications}
        )

    async def verify_insurance_eligibility(self, member_id: str, system_name: str = "insurance_system") -> Dict[str, Any]:
        """Verify insurance eligibility."""

        return await self.make_api_request(
            system_name=system_name,
            endpoint="verify_eligibility",
            method="GET",
            path_params={"member_id": member_id}
        )

    async def upload_medical_image(self, image_data: Dict[str, Any], system_name: str = "imaging_system") -> Dict[str, Any]:
        """Upload medical image for processing."""

        return await self.make_api_request(
            system_name=system_name,
            endpoint="upload_image",
            method="POST",
            data=image_data
        )

    async def health_check(self) -> Dict[str, Any]:
        """Check connectivity to external systems."""

        health_status = {
            "systems": {},
            "overall": "unknown",
            "timestamp": datetime.utcnow().isoformat()
        }

        for system_name in self.api_endpoints.keys():
            try:
                # Simple connectivity test (could be enhanced with actual health endpoints)
                system_config = self.api_endpoints[system_name]
                base_url = system_config["base_url"]

                async with self.session.get(base_url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                    health_status["systems"][system_name] = {
                        "status": "reachable" if response.status < 500 else "unhealthy",
                        "response_code": response.status,
                        "authenticated": await self._is_authenticated(system_name)
                    }
            except Exception as e:
                health_status["systems"][system_name] = {
                    "status": "unreachable",
                    "error": str(e),
                    "authenticated": False
                }

        # Determine overall health
        system_statuses = [sys["status"] for sys in health_status["systems"].values()]
        if all(status == "reachable" for status in system_statuses):
            health_status["overall"] = "healthy"
        elif any(status == "reachable" for status in system_statuses):
            health_status["overall"] = "partial"
        else:
            health_status["overall"] = "unhealthy"

        return health_status

    async def get_connection_info(self) -> Dict[str, Any]:
        """Get API gateway connection information."""

        return {
            "connected_systems": list(self.api_endpoints.keys()),
            "authenticated_systems": [
                system for system in self.authentication_tokens.keys()
                if await self._is_authenticated(system)
            ],
            "session_active": self.session is not None and not self.session.closed,
            "encryption_enabled": True,
            "supported_auth_types": ["oauth2", "jwt", "api_key", "basic"]
        }

    async def close(self):
        """Close the HTTP session."""
        if self.session and not self.session.closed:
            await self.session.close()
            logger.info("API Gateway session closed")


# Global API gateway connector instance
api_gateway_connector = APIGatewayMCPConnector()
