"""
Multi-tenancy support for Thai Regulatory AI.

Features:
- Tenant isolation
- Per-tenant configurations
- Resource quotas
- Tenant-specific vector stores
- Tenant-level analytics

Usage:
    from code.utils.multitenancy import TenantManager, get_current_tenant
    
    # Initialize
    tenant_mgr = TenantManager()
    
    # Create tenant
    tenant = tenant_mgr.create_tenant(
        tenant_id="company_abc",
        name="Company ABC",
        tier="enterprise"
    )
    
    # Get tenant context
    with tenant_mgr.tenant_context(tenant_id):
        # All operations in this context are isolated to tenant
        vector_store = get_tenant_vector_store()
"""

import os
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field, asdict
from datetime import datetime
from contextlib import contextmanager
from threading import local

from sqlalchemy import Column, String, Integer, Float, Boolean, DateTime, JSON
from sqlalchemy.orm import Session

from code.utils.models import Base


logger = logging.getLogger(__name__)


# Thread-local storage for current tenant
_tenant_context = local()


class Tenant(Base):
    """Tenant model for multi-tenancy."""
    
    __tablename__ = "tenants"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    tenant_id = Column(String(100), unique=True, nullable=False, index=True)
    name = Column(String(255), nullable=False)
    tier = Column(String(50), default="basic", nullable=False)
    
    # Configuration
    config = Column(JSON, default={})
    
    # Resource quotas
    max_users = Column(Integer, default=10)
    max_sessions = Column(Integer, default=100)
    max_api_calls_per_day = Column(Integer, default=1000)
    max_tokens_per_day = Column(Integer, default=100000)
    max_storage_mb = Column(Integer, default=1000)
    
    # Status
    is_active = Column(Boolean, default=True, nullable=False)
    suspended = Column(Boolean, default=False)
    suspension_reason = Column(String(500), nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.now, nullable=False)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    expires_at = Column(DateTime, nullable=True)
    
    # Metadata
    metadata = Column(JSON, default={})
    
    def __repr__(self):
        return f"<Tenant(tenant_id='{self.tenant_id}', name='{self.name}', tier='{self.tier}')>"


class TenantUsage(Base):
    """Track tenant resource usage."""
    
    __tablename__ = "tenant_usage"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    tenant_id = Column(String(100), nullable=False, index=True)
    
    # Usage metrics
    api_calls = Column(Integer, default=0)
    tokens_used = Column(Integer, default=0)
    storage_mb = Column(Float, default=0.0)
    active_sessions = Column(Integer, default=0)
    active_users = Column(Integer, default=0)
    
    # Cost
    total_cost = Column(Float, default=0.0)
    
    # Period
    period_start = Column(DateTime, nullable=False)
    period_end = Column(DateTime, nullable=False)
    
    # Metadata
    metadata = Column(JSON, default={})
    
    def __repr__(self):
        return f"<TenantUsage(tenant_id='{self.tenant_id}', api_calls={self.api_calls})>"


@dataclass
class TenantConfig:
    """Tenant configuration."""
    
    # LLM settings
    default_model: str = "gpt-4"
    default_temperature: float = 0.7
    max_tokens: int = 2000
    
    # RAG settings
    similarity_threshold: float = 0.7
    max_retrieved_docs: int = 5
    enable_hybrid_search: bool = True
    
    # Features
    enable_caching: bool = True
    enable_analytics: bool = True
    enable_streaming: bool = False
    
    # Security
    require_authentication: bool = True
    allowed_domains: List[str] = field(default_factory=list)
    ip_whitelist: List[str] = field(default_factory=list)
    
    # Custom
    custom_settings: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class TenantManager:
    """
    Manage multi-tenant operations.
    
    Provides tenant isolation and resource management.
    """
    
    def __init__(self, db: Optional[Session] = None):
        """
        Initialize tenant manager.
        
        Args:
            db: Database session
        """
        self.db = db
        
        # Tier quotas
        self.tier_quotas = {
            "free": {
                "max_users": 5,
                "max_sessions": 50,
                "max_api_calls_per_day": 100,
                "max_tokens_per_day": 10000,
                "max_storage_mb": 100
            },
            "basic": {
                "max_users": 25,
                "max_sessions": 500,
                "max_api_calls_per_day": 5000,
                "max_tokens_per_day": 500000,
                "max_storage_mb": 1000
            },
            "premium": {
                "max_users": 100,
                "max_sessions": 5000,
                "max_api_calls_per_day": 50000,
                "max_tokens_per_day": 5000000,
                "max_storage_mb": 10000
            },
            "enterprise": {
                "max_users": -1,  # Unlimited
                "max_sessions": -1,
                "max_api_calls_per_day": -1,
                "max_tokens_per_day": -1,
                "max_storage_mb": -1
            }
        }
    
    def create_tenant(
        self,
        tenant_id: str,
        name: str,
        tier: str = "basic",
        config: Optional[TenantConfig] = None
    ) -> Tenant:
        """
        Create new tenant.
        
        Args:
            tenant_id: Unique tenant ID
            name: Tenant name
            tier: Subscription tier
            config: Tenant configuration
            
        Returns:
            Tenant object
        """
        if not self.db:
            raise RuntimeError("Database session required")
        
        # Get tier quotas
        quotas = self.tier_quotas.get(tier, self.tier_quotas["basic"])
        
        # Create tenant
        tenant = Tenant(
            tenant_id=tenant_id,
            name=name,
            tier=tier,
            config=(config.to_dict() if config else {}),
            **quotas
        )
        
        self.db.add(tenant)
        self.db.commit()
        
        logger.info(f"Created tenant: {tenant_id} ({tier})")
        
        # Create vector store collection for tenant
        self._create_tenant_vector_store(tenant_id)
        
        return tenant
    
    def get_tenant(self, tenant_id: str) -> Optional[Tenant]:
        """Get tenant by ID."""
        if not self.db:
            return None
        
        return self.db.query(Tenant).filter(
            Tenant.tenant_id == tenant_id
        ).first()
    
    def update_tenant(
        self,
        tenant_id: str,
        **updates
    ) -> Optional[Tenant]:
        """
        Update tenant.
        
        Args:
            tenant_id: Tenant ID
            **updates: Fields to update
            
        Returns:
            Updated tenant
        """
        if not self.db:
            return None
        
        tenant = self.get_tenant(tenant_id)
        if not tenant:
            return None
        
        for key, value in updates.items():
            if hasattr(tenant, key):
                setattr(tenant, key, value)
        
        self.db.commit()
        logger.info(f"Updated tenant: {tenant_id}")
        
        return tenant
    
    def suspend_tenant(self, tenant_id: str, reason: str):
        """Suspend tenant access."""
        self.update_tenant(
            tenant_id,
            suspended=True,
            suspension_reason=reason
        )
        logger.warning(f"Suspended tenant {tenant_id}: {reason}")
    
    def check_quota(
        self,
        tenant_id: str,
        resource: str,
        amount: int = 1
    ) -> bool:
        """
        Check if tenant has quota for resource.
        
        Args:
            tenant_id: Tenant ID
            resource: Resource type (api_calls, tokens, storage)
            amount: Amount to check
            
        Returns:
            True if quota available
        """
        tenant = self.get_tenant(tenant_id)
        if not tenant or tenant.suspended:
            return False
        
        # Get usage
        usage = self._get_current_usage(tenant_id)
        
        # Check quota
        quota_attr = f"max_{resource}_per_day"
        usage_attr = resource
        
        if not hasattr(tenant, quota_attr):
            return True
        
        quota = getattr(tenant, quota_attr)
        current_usage = getattr(usage, usage_attr, 0)
        
        # -1 means unlimited
        if quota == -1:
            return True
        
        return (current_usage + amount) <= quota
    
    def track_usage(
        self,
        tenant_id: str,
        api_calls: int = 0,
        tokens: int = 0,
        storage_mb: float = 0.0,
        cost: float = 0.0
    ):
        """
        Track tenant resource usage.
        
        Args:
            tenant_id: Tenant ID
            api_calls: Number of API calls
            tokens: Number of tokens
            storage_mb: Storage in MB
            cost: Cost in USD
        """
        if not self.db:
            return
        
        usage = self._get_current_usage(tenant_id)
        
        usage.api_calls += api_calls
        usage.tokens_used += tokens
        usage.storage_mb += storage_mb
        usage.total_cost += cost
        
        self.db.commit()
    
    def _get_current_usage(self, tenant_id: str) -> TenantUsage:
        """Get current usage period for tenant."""
        if not self.db:
            raise RuntimeError("Database session required")
        
        # Get current period (daily)
        now = datetime.now()
        period_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        period_end = period_start.replace(hour=23, minute=59, second=59)
        
        # Find or create usage record
        usage = self.db.query(TenantUsage).filter(
            TenantUsage.tenant_id == tenant_id,
            TenantUsage.period_start == period_start
        ).first()
        
        if not usage:
            usage = TenantUsage(
                tenant_id=tenant_id,
                period_start=period_start,
                period_end=period_end
            )
            self.db.add(usage)
            self.db.commit()
        
        return usage
    
    def _create_tenant_vector_store(self, tenant_id: str):
        """Create isolated vector store for tenant."""
        try:
            from code.service.local_vector_store import LocalVectorStore
            
            collection_name = f"tenant_{tenant_id}"
            vector_store = LocalVectorStore(collection_name=collection_name)
            
            logger.info(f"Created vector store for tenant: {collection_name}")
        except Exception as e:
            logger.error(f"Failed to create tenant vector store: {e}")
    
    @contextmanager
    def tenant_context(self, tenant_id: str):
        """
        Set tenant context for operations.
        
        Args:
            tenant_id: Tenant ID
            
        Usage:
            with tenant_mgr.tenant_context("company_abc"):
                # All operations here are isolated to tenant
                process_request()
        """
        # Save previous context
        previous = getattr(_tenant_context, "tenant_id", None)
        
        try:
            # Set current tenant
            _tenant_context.tenant_id = tenant_id
            yield tenant_id
        finally:
            # Restore previous context
            _tenant_context.tenant_id = previous


def get_current_tenant() -> Optional[str]:
    """
    Get current tenant ID from context.
    
    Returns:
        Tenant ID or None
    """
    return getattr(_tenant_context, "tenant_id", None)


def get_tenant_vector_store(tenant_id: Optional[str] = None):
    """
    Get vector store for tenant.
    
    Args:
        tenant_id: Tenant ID (uses current context if not provided)
        
    Returns:
        Vector store instance
    """
    from code.service.local_vector_store import LocalVectorStore
    
    tenant_id = tenant_id or get_current_tenant()
    
    if not tenant_id:
        raise ValueError("No tenant context available")
    
    collection_name = f"tenant_{tenant_id}"
    return LocalVectorStore(collection_name=collection_name)


def require_tenant(func):
    """Decorator to require tenant context."""
    from functools import wraps
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        tenant_id = get_current_tenant()
        if not tenant_id:
            raise RuntimeError("Tenant context required")
        return func(*args, **kwargs)
    
    return wrapper


if __name__ == "__main__":
    # Example usage
    from code.utils.database import init_db, get_db_context
    
    # Initialize database
    init_db()
    
    with get_db_context() as db:
        # Create tenant manager
        tenant_mgr = TenantManager(db)
        
        # Create tenant
        tenant = tenant_mgr.create_tenant(
            tenant_id="company_abc",
            name="Company ABC",
            tier="premium"
        )
        print(f"Created tenant: {tenant.tenant_id}")
        
        # Use tenant context
        with tenant_mgr.tenant_context("company_abc"):
            current = get_current_tenant()
            print(f"Current tenant: {current}")
            
            # Check quota
            has_quota = tenant_mgr.check_quota("company_abc", "api_calls", 1)
            print(f"Has quota: {has_quota}")
            
            # Track usage
            tenant_mgr.track_usage(
                "company_abc",
                api_calls=1,
                tokens=100,
                cost=0.01
            )
            print("Tracked usage")
