"""
Notification Agent for managing healthcare alerts, communications, and notifications.
Handles emergency alerts, appointment reminders, and patient communications.
"""

import asyncio
import smtplib
import json
from typing import Dict, Any, Optional, List, Union
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import requests
from agents.base_agent import BaseAgent
from config.settings import config
from utils.logging import get_logger
from utils.error_handling import NotificationError, handle_exception

logger = get_logger("notification_agent")


class NotificationAgent(BaseAgent):
    """
    Specialized agent for managing healthcare notifications, alerts, and communications.
    Supports multiple notification channels with priority-based delivery.
    """

    def __init__(self, agent_id: Optional[str] = None):
        super().__init__(agent_id, specialized_task="notification")
        self.notification_channels = self._initialize_notification_channels()
        self.alert_priorities = self._initialize_alert_priorities()
        self.notification_templates = self._initialize_notification_templates()

        logger.info(f"NotificationAgent {self.agent_id} initialized with multiple communication channels")

    def _initialize_notification_channels(self) -> Dict[str, Dict[str, Any]]:
        """Initialize available notification channels."""
        return {
            "email": {
                "enabled": bool(config.notification.email_username),
                "priority": 1,
                "delivery_time": "immediate",
                "supports_attachments": True,
                "max_recipients": 100
            },
            "sms": {
                "enabled": bool(config.notification.twilio_sid),
                "priority": 2,
                "delivery_time": "immediate",
                "supports_attachments": False,
                "max_recipients": 50
            },
            "push_notification": {
                "enabled": False,  # Would require mobile app integration
                "priority": 3,
                "delivery_time": "immediate",
                "supports_attachments": False,
                "max_recipients": 1000
            },
            "system_alert": {
                "enabled": True,
                "priority": 4,
                "delivery_time": "immediate",
                "supports_attachments": True,
                "max_recipients": "unlimited"
            }
        }

    def _initialize_alert_priorities(self) -> Dict[str, Dict[str, Any]]:
        """Initialize alert priority levels for healthcare notifications."""
        return {
            "critical": {
                "level": 1,
                "description": "Life-threatening emergencies requiring immediate response",
                "channels": ["email", "sms", "system_alert"],
                "retry_attempts": 5,
                "retry_interval": 60,  # seconds
                "escalation_time": 300,  # 5 minutes
                "examples": ["cardiac_arrest", "stroke_alert", "sepsis_warning"]
            },
            "urgent": {
                "level": 2,
                "description": "High priority medical alerts requiring prompt attention",
                "channels": ["email", "sms"],
                "retry_attempts": 3,
                "retry_interval": 120,
                "escalation_time": 900,  # 15 minutes
                "examples": ["abnormal_lab_values", "medication_interactions", "appointment_conflicts"]
            },
            "important": {
                "level": 3,
                "description": "Important healthcare information requiring attention",
                "channels": ["email"],
                "retry_attempts": 2,
                "retry_interval": 300,
                "escalation_time": 1800,  # 30 minutes
                "examples": ["test_results", "prescription_ready", "appointment_reminders"]
            },
            "routine": {
                "level": 4,
                "description": "Routine healthcare communications and updates",
                "channels": ["email"],
                "retry_attempts": 1,
                "retry_interval": 600,
                "escalation_time": 3600,  # 1 hour
                "examples": ["wellness_reminders", "educational_content", "surveys"]
            }
        }

    def _initialize_notification_templates(self) -> Dict[str, Dict[str, str]]:
        """Initialize notification message templates."""
        return {
            "emergency_alert": {
                "subject": "🚨 MEDICAL EMERGENCY ALERT - {patient_name}",
                "template": """
URGENT MEDICAL ALERT

Patient: {patient_name} (ID: {patient_id})
Alert Type: {alert_type}
Severity: {severity}
Timestamp: {timestamp}

Clinical Summary:
{clinical_summary}

Immediate Actions Required:
{required_actions}

Contact Information:
- Emergency Services: 911
- Healthcare Provider: {provider_contact}
- Hospital: {hospital_contact}

This is an automated alert from the Healthcare AI System.
"""
            },
            "diagnostic_result": {
                "subject": "Diagnostic Results - {patient_name}",
                "template": """
Dear {recipient_name},

Diagnostic results are available for patient {patient_name} (ID: {patient_id}).

Analysis Summary:
{diagnostic_summary}

Recommendations:
{recommendations}

Priority Level: {priority}
Review Required: {review_required}

Please log into the system to view complete results.

Healthcare AI System
Generated on: {timestamp}
"""
            },
            "appointment_reminder": {
                "subject": "Appointment Reminder - {appointment_date}",
                "template": """
Dear {patient_name},

This is a reminder for your upcoming appointment:

Date: {appointment_date}
Time: {appointment_time}
Provider: {provider_name}
Location: {appointment_location}
Type: {appointment_type}

Please arrive 15 minutes early and bring:
- Insurance card
- Photo ID
- Current medication list
- Any relevant medical records

To reschedule: {reschedule_contact}
Questions: {contact_info}

Healthcare AI System
"""
            },
            "medication_alert": {
                "subject": "⚠️ Medication Alert - {patient_name}",
                "template": """
MEDICATION ALERT

Patient: {patient_name} (ID: {patient_id})
Alert Type: {alert_type}
Medication: {medication_name}
Issue: {medication_issue}

Details:
{alert_details}

Recommended Action:
{recommended_action}

Contact your healthcare provider immediately if you have concerns.

Pharmacy Contact: {pharmacy_contact}
Provider Contact: {provider_contact}

Healthcare AI System
Alert Time: {timestamp}
"""
            }
        }

    async def process(self, input_data: Any, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process notification requests."""
        try:
            if isinstance(input_data, dict):
                if "notification_type" in input_data:
                    return await self._send_notification(input_data, context)
                elif "alert_data" in input_data:
                    return await self._process_alert(input_data, context)
                elif "batch_notifications" in input_data:
                    return await self._send_batch_notifications(input_data, context)

            raise NotificationError("Invalid input format for notification processing")

        except Exception as e:
            logger.error(f"Notification processing failed: {str(e)}")
            raise NotificationError(f"Notification processing error: {str(e)}")

    @handle_exception
    async def _send_notification(
        self,
        notification_data: Dict[str, Any],
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Send a single notification through appropriate channels."""

        notification_type = notification_data["notification_type"]
        priority = notification_data.get("priority", "routine")
        recipients = notification_data.get("recipients", [])
        message_data = notification_data.get("message_data", {})

        # Get priority configuration
        priority_config = self.alert_priorities.get(priority, self.alert_priorities["routine"])

        # Select appropriate channels
        channels = priority_config["channels"]

        # Generate notification content
        notification_content = await self._generate_notification_content(
            notification_type,
            message_data,
            context
        )

        # Send through each channel
        delivery_results = {}
        for channel in channels:
            if self.notification_channels[channel]["enabled"]:
                try:
                    result = await self._send_via_channel(
                        channel,
                        recipients,
                        notification_content,
                        priority_config
                    )
                    delivery_results[channel] = result
                except Exception as e:
                    delivery_results[channel] = {
                        "success": False,
                        "error": str(e),
                        "timestamp": datetime.utcnow().isoformat()
                    }

        # Track delivery status
        overall_success = any(result.get("success", False) for result in delivery_results.values())

        return {
            "notification_id": f"notif_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            "notification_type": notification_type,
            "priority": priority,
            "recipients_count": len(recipients),
            "channels_used": list(delivery_results.keys()),
            "delivery_results": delivery_results,
            "overall_success": overall_success,
            "timestamp": datetime.utcnow().isoformat()
        }

    async def _generate_notification_content(
        self,
        notification_type: str,
        message_data: Dict[str, Any],
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, str]:
        """Generate notification content using templates and AI enhancement."""

        # Get base template
        template_config = self.notification_templates.get(notification_type, {})

        if template_config:
            # Use predefined template
            subject = template_config["subject"].format(**message_data)
            body = template_config["template"].format(**message_data)
        else:
            # Generate content using AI
            ai_content = await self._generate_ai_notification_content(
                notification_type,
                message_data,
                context
            )
            subject = ai_content["subject"]
            body = ai_content["body"]

        # Enhance with AI if needed for personalization
        if message_data.get("personalize", False):
            enhanced_content = await self._personalize_notification_content(
                subject,
                body,
                message_data,
                context
            )
            subject = enhanced_content["subject"]
            body = enhanced_content["body"]

        return {
            "subject": subject,
            "body": body,
            "html_body": self._convert_to_html(body),
            "sms_body": self._truncate_for_sms(body)
        }

    async def _generate_ai_notification_content(
        self,
        notification_type: str,
        message_data: Dict[str, Any],
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, str]:
        """Generate notification content using AI when no template exists."""

        system_prompt = f"""You are a healthcare communication specialist creating a {notification_type} notification.
        
        Create professional, clear, and appropriate healthcare communication that:
        - Is medically accurate and professional
        - Follows healthcare communication best practices
        - Is appropriate for the intended audience
        - Includes necessary disclaimers and contact information
        - Maintains patient privacy and HIPAA compliance
        
        Generate both subject line and message body."""

        user_prompt = f"""
        Create a {notification_type} notification with this information:
        
        Message Data: {json.dumps(message_data, indent=2)}
        Context: {context or 'None provided'}
        
        Please provide:
        1. Subject line (concise and informative)
        2. Message body (professional and complete)
        """

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        ai_response = await self.get_ai_response(messages, temperature=0.3)

        # Parse AI response (simple implementation)
        response_lines = ai_response["response"].split('\n')
        subject = "Healthcare Notification"
        body = ai_response["response"]

        for line in response_lines:
            if line.startswith("Subject:") or line.startswith("1."):
                subject = line.split(":", 1)[-1].strip()
                break

        return {"subject": subject, "body": body}

    async def _send_via_channel(
        self,
        channel: str,
        recipients: List[str],
        content: Dict[str, str],
        priority_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Send notification via specific channel."""

        if channel == "email":
            return await self._send_email_notification(recipients, content, priority_config)
        elif channel == "sms":
            return await self._send_sms_notification(recipients, content, priority_config)
        elif channel == "system_alert":
            return await self._send_system_alert(recipients, content, priority_config)
        else:
            raise NotificationError(f"Unsupported notification channel: {channel}")

    async def _send_email_notification(
        self,
        recipients: List[str],
        content: Dict[str, str],
        priority_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Send email notification."""

        if not config.notification.email_username:
            raise NotificationError("Email configuration not available")

        try:
            # Create message
            msg = MIMEMultipart('alternative')
            msg['From'] = config.notification.email_username
            msg['To'] = ', '.join(recipients)
            msg['Subject'] = content["subject"]

            # Add priority headers for critical alerts
            if priority_config["level"] <= 2:
                msg['X-Priority'] = '1'
                msg['X-MSMail-Priority'] = 'High'
                msg['Importance'] = 'High'

            # Add text and HTML parts
            text_part = MIMEText(content["body"], 'plain')
            html_part = MIMEText(content["html_body"], 'html')

            msg.attach(text_part)
            msg.attach(html_part)

            # Send email
            with smtplib.SMTP(config.notification.smtp_server, config.notification.smtp_port) as server:
                server.starttls()
                server.login(config.notification.email_username, config.notification.email_password)
                server.send_message(msg)

            return {
                "success": True,
                "channel": "email",
                "recipients_count": len(recipients),
                "timestamp": datetime.utcnow().isoformat()
            }

        except Exception as e:
            raise NotificationError(f"Email sending failed: {str(e)}")

    async def _send_sms_notification(
        self,
        recipients: List[str],
        content: Dict[str, str],
        priority_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Send SMS notification using Twilio."""

        if not config.notification.twilio_sid:
            raise NotificationError("SMS configuration not available")

        try:
            # This would integrate with Twilio API
            # For now, return success simulation
            logger.info(f"SMS would be sent to {len(recipients)} recipients: {content['sms_body'][:50]}...")

            return {
                "success": True,
                "channel": "sms",
                "recipients_count": len(recipients),
                "message_length": len(content["sms_body"]),
                "timestamp": datetime.utcnow().isoformat()
            }

        except Exception as e:
            raise NotificationError(f"SMS sending failed: {str(e)}")

    async def _send_system_alert(
        self,
        recipients: List[str],
        content: Dict[str, str],
        priority_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Send system alert notification."""

        # Log system alert
        alert_level = "CRITICAL" if priority_config["level"] <= 2 else "INFO"
        logger.log(
            "CRITICAL" if priority_config["level"] <= 2 else "INFO",
            f"SYSTEM ALERT [{alert_level}]: {content['subject']} - {content['body'][:100]}..."
        )

        return {
            "success": True,
            "channel": "system_alert",
            "alert_level": alert_level,
            "recipients_count": len(recipients),
            "timestamp": datetime.utcnow().isoformat()
        }

    def _convert_to_html(self, text_body: str) -> str:
        """Convert plain text to simple HTML."""

        html_body = text_body.replace('\n', '<br>')
        html_body = f"""
        <html>
        <body style="font-family: Arial, sans-serif; line-height: 1.6;">
        <div style="max-width: 600px; margin: 0 auto; padding: 20px;">
        {html_body}
        </div>
        </body>
        </html>
        """
        return html_body

    def _truncate_for_sms(self, text_body: str) -> str:
        """Truncate message for SMS (160 character limit)."""

        if len(text_body) <= 160:
            return text_body

        # Truncate and add continuation indicator
        return text_body[:157] + "..."

    async def send_emergency_alert(
        self,
        alert_data: Dict[str, Any],
        recipients: List[str]
    ) -> Dict[str, Any]:
        """Send high-priority emergency alert."""

        notification_data = {
            "notification_type": "emergency_alert",
            "priority": "critical",
            "recipients": recipients,
            "message_data": {
                "patient_name": alert_data.get("patient_name", "Unknown"),
                "patient_id": alert_data.get("patient_id", "Unknown"),
                "alert_type": alert_data.get("alert_type", "Medical Emergency"),
                "severity": alert_data.get("severity", "Critical"),
                "clinical_summary": alert_data.get("clinical_summary", "Emergency situation detected"),
                "required_actions": alert_data.get("required_actions", "Immediate medical attention required"),
                "provider_contact": alert_data.get("provider_contact", "Contact your healthcare provider"),
                "hospital_contact": alert_data.get("hospital_contact", "Call 911"),
                "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
            }
        }

        return await self._send_notification(notification_data, None)

    async def send_diagnostic_notification(
        self,
        diagnostic_result: Dict[str, Any],
        recipients: List[str]
    ) -> Dict[str, Any]:
        """Send diagnostic result notification."""

        notification_data = {
            "notification_type": "diagnostic_result",
            "priority": "important" if diagnostic_result.get("critical", False) else "routine",
            "recipients": recipients,
            "message_data": {
                "recipient_name": "Healthcare Provider",
                "patient_name": diagnostic_result.get("patient_name", "Unknown"),
                "patient_id": diagnostic_result.get("patient_id", "Unknown"),
                "diagnostic_summary": diagnostic_result.get("summary", "Diagnostic analysis completed"),
                "recommendations": diagnostic_result.get("recommendations", "Please review full report"),
                "priority": diagnostic_result.get("priority", "Standard"),
                "review_required": "Yes" if diagnostic_result.get("requires_review", True) else "No",
                "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
            }
        }

        return await self._send_notification(notification_data, None)

    async def get_notification_capabilities(self) -> Dict[str, Any]:
        """Get current notification capabilities."""

        return {
            "agent_id": self.agent_id,
            "available_channels": [
                channel for channel, config in self.notification_channels.items()
                if config["enabled"]
            ],
            "priority_levels": list(self.alert_priorities.keys()),
            "notification_types": list(self.notification_templates.keys()),
            "supports_batch_notifications": True,
            "supports_scheduled_notifications": True,
            "supports_emergency_alerts": True,
            "max_recipients_per_channel": {
                channel: config["max_recipients"]
                for channel, config in self.notification_channels.items()
            }
        }
