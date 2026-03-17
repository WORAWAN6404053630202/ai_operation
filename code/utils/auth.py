"""
Authentication and Authorization system for Thai Regulatory AI.

Features:
- API key authentication
- JWT token generation/validation
- Rate limiting per API key
- User management
- Role-based access control (RBAC)

Usage:
    from code.utils.auth import AuthManager, create_api_key, verify_api_key
    from fastapi import Depends, HTTPException
    
    # Initialize
    auth = AuthManager(secret_key="your-secret-key")
    
    # Create API key
    api_key = auth.create_api_key(
        user_id="user123",
        name="Production Key",
        tier="premium"
    )
    
    # Verify API key (in FastAPI endpoint)
    @app.get("/protected")
    async def protected_endpoint(
        api_key_data: dict = Depends(auth.verify_api_key)
    ):
        return {"user_id": api_key_data["user_id"]}
"""

import os
import secrets
import hashlib
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, asdict, field
import jwt
from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader


logger = logging.getLogger(__name__)


# API key header
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


@dataclass
class APIKey:
    """API key data structure."""
    key_id: str
    user_id: str
    name: str
    key_hash: str
    tier: str = "free"  # free, basic, premium, enterprise
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    is_active: bool = True
    rate_limit: int = 10  # requests per minute
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data["created_at"] = self.created_at.isoformat()
        if self.expires_at:
            data["expires_at"] = self.expires_at.isoformat()
        return data
    
    def is_expired(self) -> bool:
        """Check if API key is expired."""
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at


@dataclass
class User:
    """User data structure."""
    user_id: str
    email: str
    role: str = "user"  # user, admin, superadmin
    tier: str = "free"
    created_at: datetime = field(default_factory=datetime.now)
    is_active: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data["created_at"] = self.created_at.isoformat()
        return data


class AuthManager:
    """
    Manage authentication and authorization.
    
    Supports:
    - API key auth
    - JWT tokens
    - User management
    - Role-based access
    """
    
    def __init__(
        self,
        secret_key: Optional[str] = None,
        jwt_algorithm: str = "HS256",
        jwt_expiration_hours: int = 24
    ):
        """
        Initialize auth manager.
        
        Args:
            secret_key: Secret key for JWT
            jwt_algorithm: JWT algorithm
            jwt_expiration_hours: JWT expiration time
        """
        self.secret_key = secret_key or os.getenv("JWT_SECRET_KEY", "dev-secret-key-change-in-production")
        self.jwt_algorithm = jwt_algorithm
        self.jwt_expiration_hours = jwt_expiration_hours
        
        # In-memory storage (replace with database in production)
        self._api_keys: Dict[str, APIKey] = {}
        self._users: Dict[str, User] = {}
        
        # Tier configurations
        self.tier_limits = {
            "free": {"rate_limit": 10, "max_tokens": 1000},
            "basic": {"rate_limit": 30, "max_tokens": 5000},
            "premium": {"rate_limit": 100, "max_tokens": 20000},
            "enterprise": {"rate_limit": 1000, "max_tokens": 100000}
        }
    
    def create_user(
        self,
        user_id: str,
        email: str,
        role: str = "user",
        tier: str = "free"
    ) -> User:
        """
        Create new user.
        
        Args:
            user_id: Unique user ID
            email: User email
            role: User role
            tier: Subscription tier
            
        Returns:
            User object
        """
        if user_id in self._users:
            raise ValueError(f"User {user_id} already exists")
        
        user = User(
            user_id=user_id,
            email=email,
            role=role,
            tier=tier
        )
        
        self._users[user_id] = user
        logger.info(f"Created user: {user_id}")
        return user
    
    def get_user(self, user_id: str) -> Optional[User]:
        """Get user by ID."""
        return self._users.get(user_id)
    
    def create_api_key(
        self,
        user_id: str,
        name: str,
        tier: Optional[str] = None,
        expires_in_days: Optional[int] = None
    ) -> tuple[str, APIKey]:
        """
        Create new API key.
        
        Args:
            user_id: User ID
            name: Key name/description
            tier: Override user tier
            expires_in_days: Expiration time
            
        Returns:
            Tuple of (raw_key, api_key_object)
        """
        # Verify user exists
        user = self.get_user(user_id)
        if not user:
            raise ValueError(f"User {user_id} not found")
        
        # Generate random key
        raw_key = f"sk_{secrets.token_urlsafe(32)}"
        key_hash = self._hash_key(raw_key)
        key_id = secrets.token_urlsafe(16)
        
        # Determine tier and rate limit
        key_tier = tier or user.tier
        rate_limit = self.tier_limits.get(key_tier, {}).get("rate_limit", 10)
        
        # Calculate expiration
        expires_at = None
        if expires_in_days:
            expires_at = datetime.now() + timedelta(days=expires_in_days)
        
        # Create API key
        api_key = APIKey(
            key_id=key_id,
            user_id=user_id,
            name=name,
            key_hash=key_hash,
            tier=key_tier,
            expires_at=expires_at,
            rate_limit=rate_limit
        )
        
        self._api_keys[key_hash] = api_key
        logger.info(f"Created API key for user {user_id}: {key_id}")
        
        return raw_key, api_key
    
    def verify_api_key(self, api_key: str = Security(api_key_header)) -> Dict[str, Any]:
        """
        Verify API key (FastAPI dependency).
        
        Args:
            api_key: API key from header
            
        Returns:
            API key data
            
        Raises:
            HTTPException: If invalid
        """
        if not api_key:
            raise HTTPException(
                status_code=401,
                detail="API key required"
            )
        
        # Hash and lookup
        key_hash = self._hash_key(api_key)
        api_key_obj = self._api_keys.get(key_hash)
        
        if not api_key_obj:
            raise HTTPException(
                status_code=401,
                detail="Invalid API key"
            )
        
        # Check active status
        if not api_key_obj.is_active:
            raise HTTPException(
                status_code=401,
                detail="API key is inactive"
            )
        
        # Check expiration
        if api_key_obj.is_expired():
            raise HTTPException(
                status_code=401,
                detail="API key has expired"
            )
        
        # Return key data
        return {
            "key_id": api_key_obj.key_id,
            "user_id": api_key_obj.user_id,
            "tier": api_key_obj.tier,
            "rate_limit": api_key_obj.rate_limit
        }
    
    def revoke_api_key(self, key_id: str):
        """Revoke (deactivate) API key."""
        for api_key in self._api_keys.values():
            if api_key.key_id == key_id:
                api_key.is_active = False
                logger.info(f"Revoked API key: {key_id}")
                return
        
        raise ValueError(f"API key {key_id} not found")
    
    def create_jwt(self, user_id: str, additional_claims: Optional[Dict] = None) -> str:
        """
        Create JWT token.
        
        Args:
            user_id: User ID
            additional_claims: Extra JWT claims
            
        Returns:
            JWT token string
        """
        user = self.get_user(user_id)
        if not user:
            raise ValueError(f"User {user_id} not found")
        
        # Build claims
        now = datetime.utcnow()
        claims = {
            "sub": user_id,
            "email": user.email,
            "role": user.role,
            "tier": user.tier,
            "iat": now,
            "exp": now + timedelta(hours=self.jwt_expiration_hours)
        }
        
        if additional_claims:
            claims.update(additional_claims)
        
        # Encode JWT
        token = jwt.encode(claims, self.secret_key, algorithm=self.jwt_algorithm)
        return token
    
    def verify_jwt(self, token: str) -> Dict[str, Any]:
        """
        Verify JWT token.
        
        Args:
            token: JWT token string
            
        Returns:
            Decoded claims
            
        Raises:
            HTTPException: If invalid
        """
        try:
            claims = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.jwt_algorithm]
            )
            return claims
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=401,
                detail="Token has expired"
            )
        except jwt.InvalidTokenError:
            raise HTTPException(
                status_code=401,
                detail="Invalid token"
            )
    
    def require_role(self, required_role: str):
        """
        Dependency to check user role.
        
        Args:
            required_role: Required role (user, admin, superadmin)
            
        Returns:
            Dependency function
        """
        def role_checker(api_key_data: Dict = Security(self.verify_api_key)):
            user = self.get_user(api_key_data["user_id"])
            if not user:
                raise HTTPException(status_code=404, detail="User not found")
            
            # Role hierarchy: superadmin > admin > user
            role_hierarchy = {"superadmin": 3, "admin": 2, "user": 1}
            user_level = role_hierarchy.get(user.role, 0)
            required_level = role_hierarchy.get(required_role, 0)
            
            if user_level < required_level:
                raise HTTPException(
                    status_code=403,
                    detail=f"Requires {required_role} role"
                )
            
            return api_key_data
        
        return role_checker
    
    def require_tier(self, required_tier: str):
        """
        Dependency to check subscription tier.
        
        Args:
            required_tier: Required tier
            
        Returns:
            Dependency function
        """
        def tier_checker(api_key_data: Dict = Security(self.verify_api_key)):
            # Tier hierarchy: enterprise > premium > basic > free
            tier_hierarchy = {"enterprise": 4, "premium": 3, "basic": 2, "free": 1}
            user_tier_level = tier_hierarchy.get(api_key_data["tier"], 0)
            required_level = tier_hierarchy.get(required_tier, 0)
            
            if user_tier_level < required_level:
                raise HTTPException(
                    status_code=403,
                    detail=f"Requires {required_tier} tier or higher"
                )
            
            return api_key_data
        
        return tier_checker
    
    @staticmethod
    def _hash_key(key: str) -> str:
        """Hash API key using SHA-256."""
        return hashlib.sha256(key.encode()).hexdigest()
    
    def get_tier_info(self, tier: str) -> Dict[str, Any]:
        """Get tier configuration."""
        return self.tier_limits.get(tier, self.tier_limits["free"])


if __name__ == "__main__":
    # Example usage
    auth = AuthManager()
    
    # Create user
    user = auth.create_user(
        user_id="user123",
        email="user@example.com",
        tier="premium"
    )
    print(f"Created user: {user.user_id}")
    
    # Create API key
    raw_key, api_key = auth.create_api_key(
        user_id="user123",
        name="Production Key",
        expires_in_days=365
    )
    print(f"API Key: {raw_key}")
    print(f"Key ID: {api_key.key_id}")
    print(f"Rate limit: {api_key.rate_limit} req/min")
    
    # Create JWT
    jwt_token = auth.create_jwt("user123")
    print(f"JWT: {jwt_token[:50]}...")
    
    # Verify JWT
    claims = auth.verify_jwt(jwt_token)
    print(f"JWT claims: {claims}")
