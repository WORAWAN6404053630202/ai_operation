"""
pytest configuration and shared fixtures.
"""

import pytest
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


@pytest.fixture
def sample_session_id():
    """Sample session ID for testing."""
    return "test_session_123"


@pytest.fixture
def sample_question():
    """Sample Thai question for testing."""
    return "จดทะเบียนร้านอาหารต้องทำอย่างไร"


@pytest.fixture
def sample_documents():
    """Sample documents for RAG testing."""
    from langchain_core.documents import Document
    
    return [
        Document(
            page_content="การจดทะเบียนร้านอาหาร: ยื่นคำขอที่สำนักงานเขต นำบัตรประชาชน ทะเบียนบ้าน",
            metadata={
                "department": "สำนักงานเขต",
                "license_type": "ร้านอาหาร",
                "source": "test"
            }
        ),
        Document(
            page_content="ค่าธรรมเนียมการจดทะเบียนร้านอาหาร 500 บาท ระยะเวลา 7-14 วันทำการ",
            metadata={
                "department": "สำนักงานเขต",
                "license_type": "ร้านอาหาร",
                "fees": "500 บาท",
                "source": "test"
            }
        ),
    ]
