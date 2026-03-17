"""
Database connection and session management.

Supports PostgreSQL and SQLite with connection pooling.

Usage:
    from code.utils.database import init_db, get_db, SessionLocal
    
    # Initialize database
    init_db("postgresql://user:pass@localhost/dbname")
    # or
    init_db("sqlite:///./app.db")
    
    # Use in FastAPI dependency
    @app.get("/users")
    def get_users(db: Session = Depends(get_db)):
        return db.query(User).all()
    
    # Use directly
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.user_id == "user123").first()
    finally:
        db.close()
"""

import os
import logging
from typing import Generator, Optional
from contextlib import contextmanager
from sqlalchemy import create_engine, event, Engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool, QueuePool

from code.utils.models import Base


logger = logging.getLogger(__name__)


# Global engine and session
_engine: Optional[Engine] = None
SessionLocal: Optional[sessionmaker] = None


def get_database_url() -> str:
    """
    Get database URL from environment or use SQLite default.
    
    Returns:
        Database URL string
    """
    return os.getenv(
        "DATABASE_URL",
        "sqlite:///./thai_regulatory_ai.db"
    )


def init_db(database_url: Optional[str] = None, echo: bool = False) -> Engine:
    """
    Initialize database connection.
    
    Args:
        database_url: Database connection string
        echo: Enable SQL query logging
        
    Returns:
        SQLAlchemy engine
    """
    global _engine, SessionLocal
    
    if database_url is None:
        database_url = get_database_url()
    
    logger.info(f"Initializing database: {database_url.split('@')[-1]}")
    
    # Engine configuration
    engine_kwargs = {
        "echo": echo,
        "future": True
    }
    
    # Configure based on database type
    if database_url.startswith("sqlite"):
        # SQLite configuration
        engine_kwargs.update({
            "connect_args": {
                "check_same_thread": False,
                "timeout": 30
            },
            "poolclass": StaticPool
        })
    else:
        # PostgreSQL/MySQL configuration
        engine_kwargs.update({
            "pool_size": 10,
            "max_overflow": 20,
            "pool_pre_ping": True,
            "pool_recycle": 3600,
            "poolclass": QueuePool
        })
    
    # Create engine
    _engine = create_engine(database_url, **engine_kwargs)
    
    # Register event listeners
    _register_event_listeners(_engine)
    
    # Create session factory
    SessionLocal = sessionmaker(
        autocommit=False,
        autoflush=False,
        bind=_engine
    )
    
    # Create all tables
    Base.metadata.create_all(bind=_engine)
    
    logger.info("Database initialized successfully")
    return _engine


def _register_event_listeners(engine: Engine):
    """Register SQLAlchemy event listeners."""
    
    # Enable SQLite foreign keys
    if engine.url.drivername == "sqlite":
        @event.listens_for(engine, "connect")
        def set_sqlite_pragma(dbapi_conn, connection_record):
            cursor = dbapi_conn.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.close()
    
    # Log slow queries
    @event.listens_for(engine, "before_cursor_execute")
    def receive_before_cursor_execute(conn, cursor, statement, params, context, executemany):
        conn.info.setdefault('query_start_time', []).append(
            __import__('time').time()
        )
    
    @event.listens_for(engine, "after_cursor_execute")
    def receive_after_cursor_execute(conn, cursor, statement, params, context, executemany):
        total = __import__('time').time() - conn.info['query_start_time'].pop()
        if total > 1.0:  # Log queries slower than 1 second
            logger.warning(f"Slow query ({total:.2f}s): {statement[:100]}...")


def get_db() -> Generator[Session, None, None]:
    """
    Get database session (FastAPI dependency).
    
    Yields:
        Database session
    """
    if SessionLocal is None:
        raise RuntimeError("Database not initialized. Call init_db() first.")
    
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def get_db_context():
    """
    Get database session (context manager).
    
    Usage:
        with get_db_context() as db:
            user = db.query(User).first()
    """
    if SessionLocal is None:
        raise RuntimeError("Database not initialized. Call init_db() first.")
    
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def reset_db():
    """Drop and recreate all tables (for testing only)."""
    if _engine is None:
        raise RuntimeError("Database not initialized")
    
    logger.warning("Resetting database - dropping all tables")
    Base.metadata.drop_all(bind=_engine)
    Base.metadata.create_all(bind=_engine)
    logger.info("Database reset complete")


def close_db():
    """Close database connection."""
    global _engine, SessionLocal
    
    if _engine:
        _engine.dispose()
        _engine = None
        SessionLocal = None
        logger.info("Database connection closed")


# Migration utilities
def create_migration(message: str) -> str:
    """
    Create database migration script.
    
    Args:
        message: Migration description
        
    Returns:
        Migration file path
    """
    import datetime
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"migrations/{timestamp}_{message.replace(' ', '_')}.sql"
    
    # Create migrations directory
    os.makedirs("migrations", exist_ok=True)
    
    with open(filename, "w") as f:
        f.write(f"-- Migration: {message}\n")
        f.write(f"-- Created: {datetime.datetime.now()}\n\n")
        f.write("-- Add your SQL statements here\n")
    
    logger.info(f"Created migration: {filename}")
    return filename


def get_db_stats() -> dict:
    """
    Get database statistics.
    
    Returns:
        Dict with database stats
    """
    if _engine is None:
        return {"error": "Database not initialized"}
    
    stats = {
        "url": str(_engine.url).split("@")[-1],
        "pool_size": _engine.pool.size() if hasattr(_engine.pool, 'size') else None,
        "checked_out": _engine.pool.checkedout() if hasattr(_engine.pool, 'checkedout') else None,
        "overflow": _engine.pool.overflow() if hasattr(_engine.pool, 'overflow') else None,
        "checked_in": _engine.pool.checkedin() if hasattr(_engine.pool, 'checkedin') else None
    }
    
    return stats


if __name__ == "__main__":
    # Example usage
    from code.utils.models import User, Session as SessionModel, Message
    
    # Initialize database
    init_db("sqlite:///./test.db", echo=True)
    
    # Create test data
    with get_db_context() as db:
        # Create user
        user = User(
            user_id="user123",
            email="user@example.com",
            tier="premium"
        )
        db.add(user)
        db.commit()
        
        # Create session
        session = SessionModel(
            session_id="session123",
            user_id="user123",
            persona="practical"
        )
        db.add(session)
        db.commit()
        
        # Create message
        message = Message(
            session_id="session123",
            role="user",
            content="Test message",
            tokens=10,
            cost=0.001
        )
        db.add(message)
        db.commit()
        
        # Query
        users = db.query(User).all()
        print(f"Users: {users}")
        
        sessions = db.query(SessionModel).all()
        print(f"Sessions: {sessions}")
    
    # Get stats
    stats = get_db_stats()
    print(f"DB Stats: {stats}")
    
    # Close
    close_db()
