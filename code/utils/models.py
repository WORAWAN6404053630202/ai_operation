"""
Database models for Thai Regulatory AI.

Using SQLAlchemy ORM for database abstraction.
Supports PostgreSQL and SQLite.

Models:
- User: User accounts and profiles
- APIKey: API key management
- Session: Conversation sessions
- Message: Chat messages
- Metric: Performance metrics
- Alert: Alert history

Usage:
    from code.utils.database import init_db, get_db
    from code.utils.models import User, Session
    
    # Initialize database
    init_db("postgresql://user:pass@localhost/dbname")
    
    # Use database
    db = next(get_db())
    user = User(user_id="user123", email="user@example.com")
    db.add(user)
    db.commit()
"""

from datetime import datetime
from typing import Optional
from sqlalchemy import (
    Column, String, Integer, Float, Boolean, DateTime, Text, JSON, ForeignKey, Index
)
from sqlalchemy.orm import relationship, declarative_base
from sqlalchemy.sql import func


Base = declarative_base()


class User(Base):
    """User account model."""
    
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(100), unique=True, nullable=False, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    role = Column(String(50), default="user", nullable=False)  # user, admin, superadmin
    tier = Column(String(50), default="free", nullable=False)  # free, basic, premium, enterprise
    is_active = Column(Boolean, default=True, nullable=False)
    metadata = Column(JSON, default={})
    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now(), nullable=False)
    
    # Relationships
    api_keys = relationship("APIKey", back_populates="user", cascade="all, delete-orphan")
    sessions = relationship("Session", back_populates="user", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<User(user_id='{self.user_id}', email='{self.email}', tier='{self.tier}')>"


class APIKey(Base):
    """API key model."""
    
    __tablename__ = "api_keys"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    key_id = Column(String(100), unique=True, nullable=False, index=True)
    user_id = Column(String(100), ForeignKey("users.user_id"), nullable=False, index=True)
    name = Column(String(255), nullable=False)
    key_hash = Column(String(64), unique=True, nullable=False)  # SHA-256 hash
    tier = Column(String(50), default="free", nullable=False)
    rate_limit = Column(Integer, default=10, nullable=False)  # requests per minute
    is_active = Column(Boolean, default=True, nullable=False)
    expires_at = Column(DateTime, nullable=True)
    last_used_at = Column(DateTime, nullable=True)
    metadata = Column(JSON, default={})
    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now(), nullable=False)
    
    # Relationships
    user = relationship("User", back_populates="api_keys")
    
    __table_args__ = (
        Index("idx_api_keys_user_active", "user_id", "is_active"),
    )
    
    def __repr__(self):
        return f"<APIKey(key_id='{self.key_id}', user_id='{self.user_id}', tier='{self.tier}')>"


class Session(Base):
    """Conversation session model."""
    
    __tablename__ = "sessions"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(100), unique=True, nullable=False, index=True)
    user_id = Column(String(100), ForeignKey("users.user_id"), nullable=True, index=True)
    persona = Column(String(50), default="practical", nullable=False)  # practical, academic
    state = Column(JSON, default={})
    is_active = Column(Boolean, default=True, nullable=False)
    metadata = Column(JSON, default={})
    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now(), nullable=False)
    ended_at = Column(DateTime, nullable=True)
    
    # Relationships
    user = relationship("User", back_populates="sessions")
    messages = relationship("Message", back_populates="session", cascade="all, delete-orphan")
    
    __table_args__ = (
        Index("idx_sessions_user_active", "user_id", "is_active"),
    )
    
    def __repr__(self):
        return f"<Session(session_id='{self.session_id}', persona='{self.persona}')>"


class Message(Base):
    """Chat message model."""
    
    __tablename__ = "messages"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(100), ForeignKey("sessions.session_id"), nullable=False, index=True)
    role = Column(String(50), nullable=False)  # user, assistant, system
    content = Column(Text, nullable=False)
    tokens = Column(Integer, default=0)
    cost = Column(Float, default=0.0)
    latency_ms = Column(Float, default=0.0)
    metadata = Column(JSON, default={})
    created_at = Column(DateTime, default=func.now(), nullable=False)
    
    # Relationships
    session = relationship("Session", back_populates="messages")
    
    __table_args__ = (
        Index("idx_messages_session_created", "session_id", "created_at"),
    )
    
    def __repr__(self):
        return f"<Message(session_id='{self.session_id}', role='{self.role}')>"


class Metric(Base):
    """Performance metric model."""
    
    __tablename__ = "metrics"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    metric_type = Column(String(50), nullable=False, index=True)  # cache, llm, latency, cost
    metric_name = Column(String(100), nullable=False)
    value = Column(Float, nullable=False)
    unit = Column(String(50), default="")
    session_id = Column(String(100), nullable=True, index=True)
    user_id = Column(String(100), nullable=True, index=True)
    metadata = Column(JSON, default={})
    timestamp = Column(DateTime, default=func.now(), nullable=False, index=True)
    
    __table_args__ = (
        Index("idx_metrics_type_timestamp", "metric_type", "timestamp"),
        Index("idx_metrics_user_timestamp", "user_id", "timestamp"),
    )
    
    def __repr__(self):
        return f"<Metric(type='{self.metric_type}', name='{self.metric_name}', value={self.value})>"


class Alert(Base):
    """Alert history model."""
    
    __tablename__ = "alerts"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    level = Column(String(50), nullable=False, index=True)  # info, warning, error, critical
    title = Column(String(255), nullable=False)
    message = Column(Text, nullable=False)
    channels = Column(JSON, default=[])  # List of channels sent to
    metadata = Column(JSON, default={})
    acknowledged = Column(Boolean, default=False)
    acknowledged_at = Column(DateTime, nullable=True)
    acknowledged_by = Column(String(100), nullable=True)
    created_at = Column(DateTime, default=func.now(), nullable=False, index=True)
    
    __table_args__ = (
        Index("idx_alerts_level_created", "level", "created_at"),
    )
    
    def __repr__(self):
        return f"<Alert(level='{self.level}', title='{self.title}')>"


class AuditLog(Base):
    """Audit log for compliance."""
    
    __tablename__ = "audit_logs"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(100), nullable=True, index=True)
    action = Column(String(100), nullable=False, index=True)
    resource_type = Column(String(50), nullable=False)
    resource_id = Column(String(100), nullable=True)
    details = Column(JSON, default={})
    ip_address = Column(String(50), nullable=True)
    user_agent = Column(String(500), nullable=True)
    timestamp = Column(DateTime, default=func.now(), nullable=False, index=True)
    
    __table_args__ = (
        Index("idx_audit_user_timestamp", "user_id", "timestamp"),
        Index("idx_audit_action_timestamp", "action", "timestamp"),
    )
    
    def __repr__(self):
        return f"<AuditLog(user='{self.user_id}', action='{self.action}')>"


class Cache(Base):
    """Cache storage model (for persistent cache)."""
    
    __tablename__ = "cache"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    cache_key = Column(String(255), unique=True, nullable=False, index=True)
    cache_value = Column(Text, nullable=False)
    ttl_seconds = Column(Integer, default=3600)
    hit_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=func.now(), nullable=False)
    expires_at = Column(DateTime, nullable=False, index=True)
    
    __table_args__ = (
        Index("idx_cache_expires", "expires_at"),
    )
    
    def __repr__(self):
        return f"<Cache(key='{self.cache_key}', hits={self.hit_count})>"
