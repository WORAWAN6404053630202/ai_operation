"""
Alerting system for Thai Regulatory AI.

Sends alerts for:
- Cost spikes (exceeds budget)
- High latency (slow responses)
- Error rates (failures)
- Rate limit violations
- System health issues

Alert channels:
- Slack webhook
- Email (SMTP)
- Console logging

Usage:
    from code.utils.alerting import AlertManager, Alert, AlertLevel
    
    # Initialize
    alerts = AlertManager(
        slack_webhook="https://hooks.slack.com/services/...",
        email_config={
            "smtp_host": "smtp.gmail.com",
            "smtp_port": 587,
            "from_email": "alerts@example.com",
            "to_emails": ["team@example.com"]
        }
    )
    
    # Send alert
    alerts.send_alert(
        Alert(
            level=AlertLevel.CRITICAL,
            title="High Cost Detected",
            message="Cost exceeded $100/day",
            metadata={"cost": 125.50}
        )
    )
"""

import os
import json
import logging
import smtplib
import requests
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from enum import Enum
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta


logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class Alert:
    """Alert data structure."""
    level: AlertLevel
    title: str
    message: str
    metadata: Optional[Dict[str, Any]] = None
    timestamp: Optional[datetime] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data["level"] = self.level.value
        data["timestamp"] = self.timestamp.isoformat()
        return data
    
    def get_emoji(self) -> str:
        """Get emoji for alert level."""
        emojis = {
            AlertLevel.INFO: "ℹ️",
            AlertLevel.WARNING: "⚠️",
            AlertLevel.ERROR: "❌",
            AlertLevel.CRITICAL: "🚨"
        }
        return emojis.get(self.level, "📢")


class AlertManager:
    """
    Manage alerts and notifications.
    
    Supports multiple channels: Slack, Email, Console
    """
    
    def __init__(
        self,
        slack_webhook: Optional[str] = None,
        email_config: Optional[Dict[str, Any]] = None,
        alert_cooldown: int = 300  # 5 minutes cooldown between same alerts
    ):
        """
        Initialize alert manager.
        
        Args:
            slack_webhook: Slack webhook URL
            email_config: Email configuration dict
            alert_cooldown: Cooldown period in seconds
        """
        self.slack_webhook = slack_webhook or os.getenv("SLACK_WEBHOOK_URL")
        self.email_config = email_config or {}
        self.alert_cooldown = alert_cooldown
        
        # Track sent alerts (for cooldown)
        self._alert_history: Dict[str, datetime] = {}
    
    def send_alert(self, alert: Alert, channels: Optional[List[str]] = None):
        """
        Send alert to specified channels.
        
        Args:
            alert: Alert object
            channels: List of channels ("slack", "email", "console")
                     If None, sends to all configured channels
        """
        # Check cooldown
        alert_key = f"{alert.level.value}:{alert.title}"
        if self._is_in_cooldown(alert_key):
            logger.debug(f"Alert in cooldown: {alert_key}")
            return
        
        # Default to all channels if not specified
        if channels is None:
            channels = []
            if self.slack_webhook:
                channels.append("slack")
            if self.email_config:
                channels.append("email")
            channels.append("console")
        
        # Send to each channel
        for channel in channels:
            try:
                if channel == "slack":
                    self._send_to_slack(alert)
                elif channel == "email":
                    self._send_to_email(alert)
                elif channel == "console":
                    self._send_to_console(alert)
            except Exception as e:
                logger.error(f"Failed to send alert to {channel}: {e}")
        
        # Update history
        self._alert_history[alert_key] = datetime.now()
    
    def _is_in_cooldown(self, alert_key: str) -> bool:
        """Check if alert is in cooldown period."""
        if alert_key not in self._alert_history:
            return False
        
        last_sent = self._alert_history[alert_key]
        cooldown_end = last_sent + timedelta(seconds=self.alert_cooldown)
        return datetime.now() < cooldown_end
    
    def _send_to_slack(self, alert: Alert):
        """Send alert to Slack."""
        if not self.slack_webhook:
            logger.warning("Slack webhook not configured")
            return
        
        # Color coding
        colors = {
            AlertLevel.INFO: "#36a64f",      # Green
            AlertLevel.WARNING: "#ff9900",   # Orange
            AlertLevel.ERROR: "#ff0000",     # Red
            AlertLevel.CRITICAL: "#8b0000"   # Dark red
        }
        
        # Build Slack message
        payload = {
            "text": f"{alert.get_emoji()} *{alert.title}*",
            "attachments": [{
                "color": colors.get(alert.level, "#808080"),
                "fields": [
                    {
                        "title": "Level",
                        "value": alert.level.value.upper(),
                        "short": True
                    },
                    {
                        "title": "Time",
                        "value": alert.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                        "short": True
                    },
                    {
                        "title": "Message",
                        "value": alert.message,
                        "short": False
                    }
                ]
            }]
        }
        
        # Add metadata if present
        if alert.metadata:
            metadata_str = "\n".join([
                f"• *{k}*: {v}"
                for k, v in alert.metadata.items()
            ])
            payload["attachments"][0]["fields"].append({
                "title": "Details",
                "value": metadata_str,
                "short": False
            })
        
        # Send to Slack
        response = requests.post(
            self.slack_webhook,
            json=payload,
            timeout=10
        )
        
        if response.status_code != 200:
            raise Exception(f"Slack returned {response.status_code}: {response.text}")
        
        logger.info(f"Alert sent to Slack: {alert.title}")
    
    def _send_to_email(self, alert: Alert):
        """Send alert via email."""
        if not self.email_config:
            logger.warning("Email not configured")
            return
        
        smtp_host = self.email_config.get("smtp_host")
        smtp_port = self.email_config.get("smtp_port", 587)
        username = self.email_config.get("username")
        password = self.email_config.get("password")
        from_email = self.email_config.get("from_email")
        to_emails = self.email_config.get("to_emails", [])
        
        if not all([smtp_host, username, password, from_email, to_emails]):
            logger.warning("Incomplete email configuration")
            return
        
        # Build email
        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"[{alert.level.value.upper()}] {alert.title}"
        msg["From"] = from_email
        msg["To"] = ", ".join(to_emails)
        
        # Plain text version
        text_body = f"""
{alert.get_emoji()} {alert.title}

Level: {alert.level.value.upper()}
Time: {alert.timestamp.strftime("%Y-%m-%d %H:%M:%S")}

Message:
{alert.message}
"""
        
        if alert.metadata:
            text_body += "\n\nDetails:\n"
            for key, value in alert.metadata.items():
                text_body += f"  {key}: {value}\n"
        
        # HTML version
        html_body = f"""
<html>
<body>
    <h2>{alert.get_emoji()} {alert.title}</h2>
    <p><strong>Level:</strong> {alert.level.value.upper()}</p>
    <p><strong>Time:</strong> {alert.timestamp.strftime("%Y-%m-%d %H:%M:%S")}</p>
    <h3>Message:</h3>
    <p>{alert.message}</p>
"""
        
        if alert.metadata:
            html_body += "<h3>Details:</h3><ul>"
            for key, value in alert.metadata.items():
                html_body += f"<li><strong>{key}:</strong> {value}</li>"
            html_body += "</ul>"
        
        html_body += "</body></html>"
        
        # Attach parts
        msg.attach(MIMEText(text_body, "plain"))
        msg.attach(MIMEText(html_body, "html"))
        
        # Send email
        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.starttls()
            server.login(username, password)
            server.send_message(msg)
        
        logger.info(f"Alert sent via email: {alert.title}")
    
    def _send_to_console(self, alert: Alert):
        """Log alert to console."""
        log_methods = {
            AlertLevel.INFO: logger.info,
            AlertLevel.WARNING: logger.warning,
            AlertLevel.ERROR: logger.error,
            AlertLevel.CRITICAL: logger.critical
        }
        
        log_method = log_methods.get(alert.level, logger.info)
        
        message = f"{alert.get_emoji()} [{alert.level.value.upper()}] {alert.title}: {alert.message}"
        
        if alert.metadata:
            message += f" | Details: {json.dumps(alert.metadata)}"
        
        log_method(message)


# Pre-configured alert templates
class AlertTemplates:
    """Common alert templates."""
    
    @staticmethod
    def high_cost(daily_cost: float, threshold: float = 100.0) -> Alert:
        """Cost exceeded threshold."""
        return Alert(
            level=AlertLevel.CRITICAL if daily_cost > threshold * 2 else AlertLevel.WARNING,
            title="High Cost Detected",
            message=f"Daily cost ${daily_cost:.2f} exceeds threshold ${threshold:.2f}",
            metadata={
                "daily_cost": daily_cost,
                "threshold": threshold,
                "overage": daily_cost - threshold
            }
        )
    
    @staticmethod
    def high_latency(avg_latency_ms: float, threshold_ms: float = 2000.0) -> Alert:
        """Response time exceeded threshold."""
        return Alert(
            level=AlertLevel.WARNING,
            title="High Latency Detected",
            message=f"Average response time {avg_latency_ms:.0f}ms exceeds threshold {threshold_ms:.0f}ms",
            metadata={
                "avg_latency_ms": avg_latency_ms,
                "threshold_ms": threshold_ms
            }
        )
    
    @staticmethod
    def high_error_rate(error_rate: float, threshold: float = 0.05) -> Alert:
        """Error rate exceeded threshold."""
        return Alert(
            level=AlertLevel.ERROR if error_rate > threshold * 2 else AlertLevel.WARNING,
            title="High Error Rate",
            message=f"Error rate {error_rate*100:.1f}% exceeds threshold {threshold*100:.1f}%",
            metadata={
                "error_rate": error_rate,
                "threshold": threshold
            }
        )
    
    @staticmethod
    def rate_limit_abuse(session_id: str, blocked_count: int) -> Alert:
        """Rate limit violations."""
        return Alert(
            level=AlertLevel.WARNING,
            title="Rate Limit Abuse Detected",
            message=f"Session {session_id} blocked {blocked_count} times",
            metadata={
                "session_id": session_id,
                "blocked_count": blocked_count
            }
        )
    
    @staticmethod
    def system_unhealthy(reason: str) -> Alert:
        """System health check failed."""
        return Alert(
            level=AlertLevel.CRITICAL,
            title="System Health Check Failed",
            message=reason,
            metadata={"reason": reason}
        )


if __name__ == "__main__":
    # Example usage
    manager = AlertManager()
    
    # Test console alert
    manager.send_alert(
        AlertTemplates.high_cost(125.50, 100.0),
        channels=["console"]
    )
    
    # Test high latency alert
    manager.send_alert(
        AlertTemplates.high_latency(3500, 2000),
        channels=["console"]
    )
