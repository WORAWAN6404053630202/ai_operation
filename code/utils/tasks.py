"""
Celery task queue for async processing in Thai Regulatory AI.

Features:
- Document ingestion (async)
- Batch processing
- Report generation
- Scheduled tasks
- Task monitoring
- Retry logic

Usage:
    from code.utils.tasks import celery_app, ingest_document_async, generate_report
    
    # Start worker:
    # celery -A code.utils.tasks worker --loglevel=info
    
    # Queue task
    task = ingest_document_async.delay(file_path="/path/to/doc.pdf")
    
    # Check status
    result = task.get(timeout=300)
"""

import os
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from celery import Celery, Task
from celery.schedules import crontab
from kombu import Queue


logger = logging.getLogger(__name__)


# Celery configuration
BROKER_URL = os.getenv("CELERY_BROKER_URL", "pyamqp://guest@localhost//")
RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/0")


# Create Celery app
celery_app = Celery(
    "thai_regulatory_ai",
    broker=BROKER_URL,
    backend=RESULT_BACKEND
)


# Celery configuration
celery_app.conf.update(
    # Task settings
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="Asia/Bangkok",
    enable_utc=False,
    
    # Task execution
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    task_time_limit=3600,  # 1 hour
    task_soft_time_limit=3300,  # 55 minutes
    
    # Result backend
    result_expires=86400,  # 24 hours
    result_persistent=True,
    
    # Worker settings
    worker_prefetch_multiplier=4,
    worker_max_tasks_per_child=1000,
    
    # Task routing
    task_routes={
        "code.utils.tasks.ingest_document_async": {"queue": "ingestion"},
        "code.utils.tasks.generate_report": {"queue": "reports"},
        "code.utils.tasks.cleanup_old_sessions": {"queue": "maintenance"}
    },
    
    # Queues
    task_queues=(
        Queue("default", routing_key="default"),
        Queue("ingestion", routing_key="ingestion"),
        Queue("reports", routing_key="reports"),
        Queue("maintenance", routing_key="maintenance")
    ),
    
    # Beat schedule (periodic tasks)
    beat_schedule={
        "cleanup-sessions-daily": {
            "task": "code.utils.tasks.cleanup_old_sessions",
            "schedule": crontab(hour=2, minute=0),  # 2 AM daily
        },
        "update-metrics-hourly": {
            "task": "code.utils.tasks.update_metrics",
            "schedule": crontab(minute=0),  # Every hour
        },
        "warm-cache-morning": {
            "task": "code.utils.tasks.warm_cache",
            "schedule": crontab(hour=6, minute=0),  # 6 AM daily
        }
    }
)


class CallbackTask(Task):
    """Base task with callbacks."""
    
    def on_success(self, retval, task_id, args, kwargs):
        """Called on successful task completion."""
        logger.info(f"Task {task_id} succeeded: {self.name}")
    
    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Called on task failure."""
        logger.error(f"Task {task_id} failed: {self.name} - {exc}")
    
    def on_retry(self, exc, task_id, args, kwargs, einfo):
        """Called on task retry."""
        logger.warning(f"Task {task_id} retrying: {self.name} - {exc}")


@celery_app.task(
    bind=True,
    base=CallbackTask,
    max_retries=3,
    default_retry_delay=60
)
def ingest_document_async(self, file_path: str, collection_name: str = "thai_regulatory") -> Dict[str, Any]:
    """
    Ingest document asynchronously.
    
    Args:
        file_path: Path to document
        collection_name: Vector store collection name
        
    Returns:
        Ingestion result
    """
    try:
        logger.info(f"Ingesting document: {file_path}")
        
        # Import here to avoid circular dependencies
        from code.service.local_vector_store import LocalVectorStore
        from code.utils.semantic_chunker import SemanticChunker
        
        # Load document
        from langchain_community.document_loaders import PyPDFLoader, TextLoader
        
        if file_path.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        else:
            loader = TextLoader(file_path)
        
        documents = loader.load()
        
        # Chunk documents
        chunker = SemanticChunker(
            min_chunk_chars=200,
            max_chunk_chars=600,
            overlap_chars=50
        )
        chunks = chunker.chunk_documents(documents)
        
        # Store in vector database
        vector_store = LocalVectorStore(collection_name=collection_name)
        vector_store.add_documents(chunks)
        
        result = {
            "file_path": file_path,
            "document_count": len(documents),
            "chunk_count": len(chunks),
            "status": "success",
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"Ingestion complete: {len(chunks)} chunks")
        return result
        
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        # Retry with exponential backoff
        raise self.retry(exc=e)


@celery_app.task(bind=True, base=CallbackTask)
def batch_ingest_documents(self, file_paths: List[str], collection_name: str = "thai_regulatory") -> Dict[str, Any]:
    """
    Batch ingest multiple documents.
    
    Args:
        file_paths: List of file paths
        collection_name: Vector store collection
        
    Returns:
        Batch ingestion result
    """
    results = {
        "total": len(file_paths),
        "success": 0,
        "failed": 0,
        "results": []
    }
    
    for file_path in file_paths:
        try:
            result = ingest_document_async.delay(file_path, collection_name)
            results["results"].append({
                "file": file_path,
                "task_id": result.id,
                "status": "queued"
            })
            results["success"] += 1
        except Exception as e:
            logger.error(f"Failed to queue {file_path}: {e}")
            results["failed"] += 1
            results["results"].append({
                "file": file_path,
                "status": "failed",
                "error": str(e)
            })
    
    return results


@celery_app.task(bind=True, base=CallbackTask)
def generate_report(
    self,
    report_type: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    user_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generate analytics report.
    
    Args:
        report_type: Type of report (usage, cost, performance)
        start_date: Start date (ISO format)
        end_date: End date (ISO format)
        user_id: Filter by user ID
        
    Returns:
        Report data
    """
    logger.info(f"Generating {report_type} report")
    
    try:
        from code.utils.database import get_db_context
        from code.utils.models import Metric, Message
        from sqlalchemy import func
        
        with get_db_context() as db:
            # Build query based on report type
            if report_type == "usage":
                # Usage statistics
                query = db.query(
                    func.count(Message.id).label("total_messages"),
                    func.sum(Message.tokens).label("total_tokens"),
                    func.avg(Message.latency_ms).label("avg_latency")
                )
                
                if user_id:
                    from code.utils.models import Session
                    query = query.join(Session).filter(Session.user_id == user_id)
                
                if start_date:
                    query = query.filter(Message.created_at >= start_date)
                if end_date:
                    query = query.filter(Message.created_at <= end_date)
                
                result = query.first()
                
                return {
                    "report_type": report_type,
                    "total_messages": result.total_messages or 0,
                    "total_tokens": result.total_tokens or 0,
                    "avg_latency_ms": round(result.avg_latency or 0, 2),
                    "generated_at": datetime.now().isoformat()
                }
            
            elif report_type == "cost":
                # Cost analysis
                query = db.query(
                    func.sum(Message.cost).label("total_cost"),
                    func.count(Message.id).label("message_count")
                )
                
                if start_date:
                    query = query.filter(Message.created_at >= start_date)
                if end_date:
                    query = query.filter(Message.created_at <= end_date)
                
                result = query.first()
                
                return {
                    "report_type": report_type,
                    "total_cost": round(result.total_cost or 0, 4),
                    "message_count": result.message_count or 0,
                    "avg_cost_per_message": round((result.total_cost or 0) / max(result.message_count or 1, 1), 4),
                    "generated_at": datetime.now().isoformat()
                }
            
            else:
                return {"error": f"Unknown report type: {report_type}"}
                
    except Exception as e:
        logger.error(f"Report generation failed: {e}")
        raise self.retry(exc=e)


@celery_app.task
def cleanup_old_sessions(days: int = 30) -> Dict[str, Any]:
    """
    Clean up old inactive sessions.
    
    Args:
        days: Delete sessions older than N days
        
    Returns:
        Cleanup result
    """
    logger.info(f"Cleaning up sessions older than {days} days")
    
    try:
        from code.utils.database import get_db_context
        from code.utils.models import Session
        
        cutoff_date = datetime.now() - timedelta(days=days)
        
        with get_db_context() as db:
            deleted = db.query(Session).filter(
                Session.created_at < cutoff_date,
                Session.is_active == False
            ).delete()
            
            db.commit()
            
            logger.info(f"Deleted {deleted} old sessions")
            
            return {
                "deleted_count": deleted,
                "cutoff_date": cutoff_date.isoformat(),
                "timestamp": datetime.now().isoformat()
            }
            
    except Exception as e:
        logger.error(f"Cleanup failed: {e}")
        return {"error": str(e)}


@celery_app.task
def update_metrics() -> Dict[str, Any]:
    """
    Update aggregated metrics.
    
    Returns:
        Update result
    """
    logger.info("Updating metrics")
    
    try:
        from code.utils.database import get_db_context
        from code.utils.models import Metric, Message
        from sqlalchemy import func
        
        with get_db_context() as db:
            # Calculate hourly metrics
            now = datetime.now()
            hour_ago = now - timedelta(hours=1)
            
            # Message count
            message_count = db.query(func.count(Message.id)).filter(
                Message.created_at >= hour_ago
            ).scalar()
            
            # Average latency
            avg_latency = db.query(func.avg(Message.latency_ms)).filter(
                Message.created_at >= hour_ago
            ).scalar()
            
            # Total cost
            total_cost = db.query(func.sum(Message.cost)).filter(
                Message.created_at >= hour_ago
            ).scalar()
            
            # Store metrics
            metrics = [
                Metric(
                    metric_type="messages",
                    metric_name="hourly_count",
                    value=message_count or 0,
                    unit="count"
                ),
                Metric(
                    metric_type="latency",
                    metric_name="hourly_avg",
                    value=avg_latency or 0,
                    unit="ms"
                ),
                Metric(
                    metric_type="cost",
                    metric_name="hourly_total",
                    value=total_cost or 0,
                    unit="usd"
                )
            ]
            
            db.add_all(metrics)
            db.commit()
            
            return {
                "metrics_updated": len(metrics),
                "timestamp": now.isoformat()
            }
            
    except Exception as e:
        logger.error(f"Metrics update failed: {e}")
        return {"error": str(e)}


@celery_app.task
def warm_cache() -> Dict[str, Any]:
    """
    Warm cache with frequently accessed data.
    
    Returns:
        Warming result
    """
    logger.info("Warming cache")
    
    try:
        from code.utils.redis_cache import get_redis_cache
        
        cache = get_redis_cache()
        
        # Cache frequently accessed data
        warm_data = {
            "system:status": "operational",
            "system:last_warmup": datetime.now().isoformat()
        }
        
        cache.warm_cache(warm_data, ttl=3600)
        
        return {
            "cached_items": len(warm_data),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Cache warming failed: {e}")
        return {"error": str(e)}


if __name__ == "__main__":
    # Start worker:
    # celery -A code.utils.tasks worker --loglevel=info
    # 
    # Start beat (scheduler):
    # celery -A code.utils.tasks beat --loglevel=info
    # 
    # Monitor:
    # celery -A code.utils.tasks flower
    
    pass
