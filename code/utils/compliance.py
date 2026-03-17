"""
Compliance and audit logging for Thai Regulatory AI.

Features:
- Comprehensive audit logging
- GDPR compliance tools
- Data export (right to access)
- Data deletion (right to erasure)
- Conversation archiving
- Compliance reports

Usage:
    from code.utils.compliance import AuditLogger, GDPRManager, ComplianceReport
    
    # Audit logging
    audit = AuditLogger()
    audit.log_action(
        user_id="user123",
        action="data_access",
        resource_type="conversation",
        resource_id="session_abc"
    )
    
    # GDPR operations
    gdpr = GDPRManager()
    data = gdpr.export_user_data("user123")
    gdpr.delete_user_data("user123")
    
    # Compliance report
    report = ComplianceReport()
    summary = report.generate_report(start_date, end_date)
"""

import os
import json
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from pathlib import Path
import hashlib

from sqlalchemy.orm import Session

from code.utils.models import AuditLog, User, Session as SessionModel, Message
from code.utils.database import get_db_context


logger = logging.getLogger(__name__)


class AuditLogger:
    """
    Comprehensive audit logging system.
    
    Logs all sensitive operations for compliance and security.
    """
    
    def __init__(self, db: Optional[Session] = None):
        """
        Initialize audit logger.
        
        Args:
            db: Database session
        """
        self.db = db
    
    def log_action(
        self,
        user_id: Optional[str],
        action: str,
        resource_type: str,
        resource_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ):
        """
        Log an action.
        
        Args:
            user_id: User performing action
            action: Action type
            resource_type: Type of resource
            resource_id: Resource identifier
            details: Additional details
            ip_address: Client IP address
            user_agent: Client user agent
        """
        if self.db:
            log_entry = AuditLog(
                user_id=user_id,
                action=action,
                resource_type=resource_type,
                resource_id=resource_id,
                details=details or {},
                ip_address=ip_address,
                user_agent=user_agent
            )
            
            self.db.add(log_entry)
            self.db.commit()
        
        # Also log to file
        self._log_to_file({
            "timestamp": datetime.now().isoformat(),
            "user_id": user_id,
            "action": action,
            "resource_type": resource_type,
            "resource_id": resource_id,
            "details": details,
            "ip_address": ip_address
        })
    
    def _log_to_file(self, log_data: Dict[str, Any]):
        """Write audit log to file."""
        log_dir = Path("logs/audit")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Daily log files
        log_file = log_dir / f"audit_{datetime.now().strftime('%Y%m%d')}.jsonl"
        
        with open(log_file, "a") as f:
            f.write(json.dumps(log_data) + "\n")
    
    def get_user_actions(
        self,
        user_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        action_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get user action history.
        
        Args:
            user_id: User ID
            start_date: Start date filter
            end_date: End date filter
            action_type: Filter by action type
            
        Returns:
            List of actions
        """
        if not self.db:
            return []
        
        query = self.db.query(AuditLog).filter(AuditLog.user_id == user_id)
        
        if start_date:
            query = query.filter(AuditLog.timestamp >= start_date)
        if end_date:
            query = query.filter(AuditLog.timestamp <= end_date)
        if action_type:
            query = query.filter(AuditLog.action == action_type)
        
        logs = query.order_by(AuditLog.timestamp.desc()).all()
        
        return [
            {
                "timestamp": log.timestamp.isoformat(),
                "action": log.action,
                "resource_type": log.resource_type,
                "resource_id": log.resource_id,
                "details": log.details
            }
            for log in logs
        ]


class GDPRManager:
    """
    GDPR compliance manager.
    
    Implements right to access, erasure, and portability.
    """
    
    def __init__(self, db: Optional[Session] = None):
        """
        Initialize GDPR manager.
        
        Args:
            db: Database session
        """
        self.db = db
        self.audit = AuditLogger(db)
    
    def export_user_data(
        self,
        user_id: str,
        include_conversations: bool = True,
        include_audit_logs: bool = True
    ) -> Dict[str, Any]:
        """
        Export all user data (right to access).
        
        Args:
            user_id: User ID
            include_conversations: Include conversation history
            include_audit_logs: Include audit logs
            
        Returns:
            Complete user data package
        """
        logger.info(f"Exporting data for user: {user_id}")
        
        if not self.db:
            raise RuntimeError("Database required")
        
        # Get user
        user = self.db.query(User).filter(User.user_id == user_id).first()
        if not user:
            raise ValueError(f"User not found: {user_id}")
        
        data = {
            "user_id": user_id,
            "export_date": datetime.now().isoformat(),
            "user_profile": {
                "email": user.email,
                "role": user.role,
                "tier": user.tier,
                "created_at": user.created_at.isoformat(),
                "metadata": user.metadata
            }
        }
        
        # Export conversations
        if include_conversations:
            sessions = self.db.query(SessionModel).filter(
                SessionModel.user_id == user_id
            ).all()
            
            conversations = []
            for session in sessions:
                messages = self.db.query(Message).filter(
                    Message.session_id == session.session_id
                ).order_by(Message.created_at).all()
                
                conversations.append({
                    "session_id": session.session_id,
                    "persona": session.persona,
                    "created_at": session.created_at.isoformat(),
                    "messages": [
                        {
                            "role": msg.role,
                            "content": msg.content,
                            "timestamp": msg.created_at.isoformat()
                        }
                        for msg in messages
                    ]
                })
            
            data["conversations"] = conversations
        
        # Export audit logs
        if include_audit_logs:
            audit_logs = self.audit.get_user_actions(user_id)
            data["audit_logs"] = audit_logs
        
        # Log export action
        self.audit.log_action(
            user_id=user_id,
            action="data_export",
            resource_type="user_data",
            resource_id=user_id,
            details={"exported_at": datetime.now().isoformat()}
        )
        
        return data
    
    def save_export_to_file(
        self,
        user_id: str,
        data: Dict[str, Any],
        output_dir: str = "exports"
    ) -> str:
        """
        Save export to file.
        
        Args:
            user_id: User ID
            data: Export data
            output_dir: Output directory
            
        Returns:
            File path
        """
        # Create directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"user_data_export_{user_id}_{timestamp}.json"
        filepath = Path(output_dir) / filename
        
        # Write file
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved export to: {filepath}")
        return str(filepath)
    
    def delete_user_data(
        self,
        user_id: str,
        delete_audit_logs: bool = False,
        soft_delete: bool = True
    ) -> Dict[str, int]:
        """
        Delete user data (right to erasure).
        
        Args:
            user_id: User ID
            delete_audit_logs: Also delete audit logs
            soft_delete: Mark as deleted instead of hard delete
            
        Returns:
            Deletion summary
        """
        logger.warning(f"Deleting data for user: {user_id}")
        
        if not self.db:
            raise RuntimeError("Database required")
        
        summary = {
            "sessions_deleted": 0,
            "messages_deleted": 0,
            "user_deleted": 0
        }
        
        # Get user
        user = self.db.query(User).filter(User.user_id == user_id).first()
        if not user:
            raise ValueError(f"User not found: {user_id}")
        
        # Log deletion request
        self.audit.log_action(
            user_id=user_id,
            action="data_deletion_request",
            resource_type="user_data",
            resource_id=user_id,
            details={"soft_delete": soft_delete}
        )
        
        if soft_delete:
            # Anonymize user data
            user.email = f"deleted_{hashlib.md5(user_id.encode()).hexdigest()[:8]}@deleted.local"
            user.is_active = False
            user.metadata = {"deleted_at": datetime.now().isoformat()}
            summary["user_deleted"] = 1
            
            # Anonymize messages
            sessions = self.db.query(SessionModel).filter(
                SessionModel.user_id == user_id
            ).all()
            
            for session in sessions:
                messages = self.db.query(Message).filter(
                    Message.session_id == session.session_id
                ).all()
                
                for message in messages:
                    message.content = "[DELETED]"
                    summary["messages_deleted"] += 1
                
                session.is_active = False
                summary["sessions_deleted"] += 1
            
        else:
            # Hard delete
            # Delete messages
            for session in user.sessions:
                deleted_messages = self.db.query(Message).filter(
                    Message.session_id == session.session_id
                ).delete()
                summary["messages_deleted"] += deleted_messages
            
            # Delete sessions
            deleted_sessions = self.db.query(SessionModel).filter(
                SessionModel.user_id == user_id
            ).delete()
            summary["sessions_deleted"] = deleted_sessions
            
            # Delete user
            self.db.delete(user)
            summary["user_deleted"] = 1
            
            # Delete audit logs if requested
            if delete_audit_logs:
                self.db.query(AuditLog).filter(
                    AuditLog.user_id == user_id
                ).delete()
        
        self.db.commit()
        
        logger.info(f"Deletion complete: {summary}")
        return summary
    
    def anonymize_conversation(self, session_id: str):
        """
        Anonymize a conversation.
        
        Args:
            session_id: Session ID
        """
        if not self.db:
            return
        
        session = self.db.query(SessionModel).filter(
            SessionModel.session_id == session_id
        ).first()
        
        if session:
            messages = self.db.query(Message).filter(
                Message.session_id == session_id
            ).all()
            
            for message in messages:
                # Mask PII in content
                from code.utils.security import SecurityManager
                security = SecurityManager()
                message.content = security.mask_pii(message.content)
            
            session.metadata = {
                **session.metadata,
                "anonymized_at": datetime.now().isoformat()
            }
            
            self.db.commit()
            logger.info(f"Anonymized conversation: {session_id}")


class ComplianceReport:
    """Generate compliance reports."""
    
    def __init__(self, db: Optional[Session] = None):
        """
        Initialize compliance reporter.
        
        Args:
            db: Database session
        """
        self.db = db
    
    def generate_report(
        self,
        start_date: datetime,
        end_date: datetime,
        report_type: str = "summary"
    ) -> Dict[str, Any]:
        """
        Generate compliance report.
        
        Args:
            start_date: Report start date
            end_date: Report end date
            report_type: Type of report
            
        Returns:
            Report data
        """
        if not self.db:
            raise RuntimeError("Database required")
        
        report = {
            "report_type": report_type,
            "period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat()
            },
            "generated_at": datetime.now().isoformat()
        }
        
        # User statistics
        total_users = self.db.query(User).count()
        active_users = self.db.query(User).filter(User.is_active == True).count()
        
        report["users"] = {
            "total": total_users,
            "active": active_users,
            "inactive": total_users - active_users
        }
        
        # Data requests
        export_requests = self.db.query(AuditLog).filter(
            AuditLog.action == "data_export",
            AuditLog.timestamp >= start_date,
            AuditLog.timestamp <= end_date
        ).count()
        
        deletion_requests = self.db.query(AuditLog).filter(
            AuditLog.action == "data_deletion_request",
            AuditLog.timestamp >= start_date,
            AuditLog.timestamp <= end_date
        ).count()
        
        report["gdpr_requests"] = {
            "data_exports": export_requests,
            "deletion_requests": deletion_requests
        }
        
        # Audit log summary
        audit_count = self.db.query(AuditLog).filter(
            AuditLog.timestamp >= start_date,
            AuditLog.timestamp <= end_date
        ).count()
        
        report["audit_logs"] = {
            "total_events": audit_count
        }
        
        # Conversations
        total_sessions = self.db.query(SessionModel).filter(
            SessionModel.created_at >= start_date,
            SessionModel.created_at <= end_date
        ).count()
        
        total_messages = self.db.query(Message).filter(
            Message.created_at >= start_date,
            Message.created_at <= end_date
        ).count()
        
        report["conversations"] = {
            "total_sessions": total_sessions,
            "total_messages": total_messages
        }
        
        return report
    
    def export_audit_logs(
        self,
        start_date: datetime,
        end_date: datetime,
        output_file: str
    ):
        """
        Export audit logs to file.
        
        Args:
            start_date: Start date
            end_date: End date
            output_file: Output file path
        """
        if not self.db:
            raise RuntimeError("Database required")
        
        logs = self.db.query(AuditLog).filter(
            AuditLog.timestamp >= start_date,
            AuditLog.timestamp <= end_date
        ).order_by(AuditLog.timestamp).all()
        
        data = [
            {
                "timestamp": log.timestamp.isoformat(),
                "user_id": log.user_id,
                "action": log.action,
                "resource_type": log.resource_type,
                "resource_id": log.resource_id,
                "details": log.details,
                "ip_address": log.ip_address
            }
            for log in logs
        ]
        
        with open(output_file, "w") as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Exported {len(data)} audit logs to {output_file}")


if __name__ == "__main__":
    # Example usage
    from code.utils.database import init_db, get_db_context
    
    init_db()
    
    with get_db_context() as db:
        # Audit logging
        audit = AuditLogger(db)
        audit.log_action(
            user_id="user123",
            action="login",
            resource_type="session",
            ip_address="192.168.1.1"
        )
        
        # GDPR operations
        gdpr = GDPRManager(db)
        
        # Export user data
        # data = gdpr.export_user_data("user123")
        # filepath = gdpr.save_export_to_file("user123", data)
        # print(f"Exported to: {filepath}")
        
        # Compliance report
        reporter = ComplianceReport(db)
        report = reporter.generate_report(
            start_date=datetime.now() - timedelta(days=30),
            end_date=datetime.now()
        )
        print(f"Compliance report: {json.dumps(report, indent=2)}")
