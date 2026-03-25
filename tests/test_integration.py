"""
Integration tests for API endpoints and full RAG pipeline.
Tests end-to-end functionality with mocked LLM calls.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


@pytest.fixture
def mock_llm_response():
    """Mock LLM response for testing."""
    mock = MagicMock()
    mock.content = "สวัสดีครับ! ผมชื่อน้องโคโค่ ผมพร้อมช่วยเหลือคุณเรื่องการจดทะเบียนธุรกิจ"
    mock.usage_metadata = {
        "input_tokens": 1000,
        "output_tokens": 200
    }
    return mock


@pytest.fixture
def mock_retriever():
    """Mock vector store retriever."""
    from langchain_core.documents import Document
    
    mock = Mock()
    mock.invoke = Mock(return_value=[
        Document(
            page_content="การจดทะเบียนร้านอาหาร: ยื่นคำขอที่สำนักงานเขต",
            metadata={"source": "test", "department": "สำนักงานเขต"}
        )
    ])
    return mock


@pytest.fixture
def client(mock_llm_response, mock_retriever):
    """FastAPI test client with mocked dependencies."""
    with patch('code.router.route_v1.supervisor') as mock_supervisor, \
         patch('code.router.route_v1.state_manager') as mock_state_manager:
        
        # Setup mock supervisor
        mock_supervisor.handle = Mock(return_value=(
            MagicMock(session_id="test_123", persona_id="practical", messages=[]),
            "สวัสดีครับ! ยินดีให้บริการครับ"
        ))
        
        # Setup mock state manager
        mock_state_manager.load = Mock(return_value=None)
        mock_state_manager.save = Mock()
        mock_state_manager.list_sessions = Mock(return_value=[])
        
        from code.app import app
        return TestClient(app)


class TestAPIEndpoints:
    """Test API endpoints."""
    
    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/api/operation/healthcheck")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "timestamp" in data
        assert "cache" in data
        assert "rate_limit" in data
    
    def test_greeting_endpoint(self, client):
        """Test greeting endpoint creates new session."""
        response = client.post("/api/operation/greeting")
        
        assert response.status_code == 200
        data = response.json()
        assert "session_id" in data
        assert "response" in data
        assert data["persona_id"] in ["practical", "academic"]
    
    def test_greeting_with_persona(self, client):
        """Test greeting with specific persona."""
        response = client.post(
            "/api/operation/greeting",
            json={"persona_id": "academic"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["persona_id"] == "academic"
    
    def test_chat_endpoint(self, client):
        """Test chat endpoint."""
        response = client.post(
            "/api/operation/chat",
            json={
                "message": "จดทะเบียนร้านอาหารต้องทำอย่างไร",
                "session_id": "test_123"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "response" in data
        assert "session_id" in data
    
    def test_chat_empty_message(self, client):
        """Test chat with empty message returns 400."""
        response = client.post(
            "/api/operation/chat",
            json={
                "message": "",
                "session_id": "test_123"
            }
        )
        
        assert response.status_code == 400
    
    def test_chat_creates_session_if_not_exists(self, client):
        """Test chat creates session if session_id not provided."""
        response = client.post(
            "/api/operation/chat",
            json={"message": "สวัสดี"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "session_id" in data
        assert data["session_id"].startswith("s_")
    
    def test_reset_session(self, client):
        """Test session reset."""
        response = client.post(
            "/api/operation/reset",
            json={"session_id": "test_123"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "session_id" in data
    
    def test_list_sessions(self, client):
        """Test list sessions endpoint."""
        response = client.get("/api/operation/sessions")
        
        assert response.status_code == 200
        data = response.json()
        assert "sessions" in data


class TestRateLimitingIntegration:
    """Test rate limiting in API."""
    
    def test_rate_limit_blocks_excess_requests(self, client):
        """Test rate limiter blocks after limit exceeded."""
        session_id = "rate_test_123"
        
        # Make requests up to limit
        for i in range(10):
            response = client.post(
                "/api/operation/chat",
                json={
                    "message": f"test message {i}",
                    "session_id": session_id
                }
            )
            assert response.status_code == 200
        
        # Next request should be rate limited
        response = client.post(
            "/api/operation/chat",
            json={
                "message": "excess request",
                "session_id": session_id
            }
        )
        
        assert response.status_code == 429
        assert "retry" in response.text.lower() or "rate limit" in response.text.lower()


class TestCacheIntegration:
    """Test caching in API."""
    
    def test_cache_hit_on_duplicate_question(self, client):
        """Test cache returns cached response for duplicate questions."""
        session_id = "cache_test_123"
        question = "จดทะเบียนร้านอาหารต้องทำอย่างไร"
        
        # First request - cache miss
        response1 = client.post(
            "/api/operation/chat",
            json={"message": question, "session_id": session_id}
        )
        
        assert response1.status_code == 200
        data1 = response1.json()
        
        # Second request - should be cache hit
        response2 = client.post(
            "/api/operation/chat",
            json={"message": question, "session_id": session_id}
        )
        
        assert response2.status_code == 200
        data2 = response2.json()
        
        # Check cache stats increased
        health = client.get("/api/operation/healthcheck").json()
        assert health["cache"]["hits"] >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
