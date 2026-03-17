"""
Test 92: Performance & Load Testing with Quality Metrics
Tests system performance under load while measuring answer quality
"""
from __future__ import annotations

import pytest
import time
import asyncio
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from model.conversation_state import ConversationState
from model.state_manager import StateManager
from model.persona_supervisor import PersonaSupervisor


class PerformanceMetrics:
    """Collect and analyze performance metrics"""
    
    def __init__(self):
        self.response_times = []
        self.success_count = 0
        self.error_count = 0
        self.quality_scores = []
    
    def record_response(self, response_time: float, success: bool, quality_score: float = None):
        """Record a response"""
        self.response_times.append(response_time)
        
        if success:
            self.success_count += 1
        else:
            self.error_count += 1
        
        if quality_score is not None:
            self.quality_scores.append(quality_score)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        if not self.response_times:
            return {"error": "No data collected"}
        
        sorted_times = sorted(self.response_times)
        
        return {
            "total_requests": len(self.response_times),
            "success_count": self.success_count,
            "error_count": self.error_count,
            "success_rate": (self.success_count / len(self.response_times)) * 100,
            "response_times": {
                "min": min(self.response_times),
                "max": max(self.response_times),
                "mean": statistics.mean(self.response_times),
                "median": statistics.median(self.response_times),
                "p95": sorted_times[int(len(sorted_times) * 0.95)],
                "p99": sorted_times[int(len(sorted_times) * 0.99)],
            },
            "quality": {
                "avg_score": statistics.mean(self.quality_scores) if self.quality_scores else None,
                "min_score": min(self.quality_scores) if self.quality_scores else None,
            }
        }


@pytest.fixture()
def supervisor(retriever):
    """Create supervisor for testing"""
    return PersonaSupervisor(retriever=retriever)


@pytest.fixture
def state_manager():
    """Create state manager"""
    return StateManager()


class TestPerformance:
    """Performance testing"""
    
    def test_single_request_performance(self, supervisor, state_manager):
        """Test single request performance baseline"""
        session_id = "perf_test_single"
        state = ConversationState(session_id=session_id, persona_id="practical", context={})
        
        # Measure greeting
        start = time.time()
        state, reply = supervisor.handle(state, "")
        greeting_time = time.time() - start
        
        # Measure question answering
        start = time.time()
        state, reply = supervisor.handle(state, "จดทะเบียนร้านอาหารยังไง")
        answer_time = time.time() - start
        
        print(f"\n⚡ Single Request Performance:")
        print(f"   Greeting: {greeting_time:.3f}s")
        print(f"   Question: {answer_time:.3f}s")
        
        # Performance targets
        assert greeting_time < 5.0, f"Greeting too slow ({greeting_time:.3f}s > 5s)"
        assert answer_time < 10.0, f"Answer too slow ({answer_time:.3f}s > 10s)"
    
    def test_concurrent_requests(self, supervisor, state_manager):
        """Test handling concurrent requests"""
        num_concurrent = 5
        metrics = PerformanceMetrics()
        
        def process_request(request_id: int):
            """Process single request"""
            try:
                session_id = f"concurrent_{request_id}"
                state = ConversationState(
                    session_id=session_id,
                    persona_id="practical",
                    context={}
                )
                
                start = time.time()
                state, reply = supervisor.handle(state, "เปิดร้านอาหารต้องทำอะไรบ้าง")
                elapsed = time.time() - start
                
                # Calculate simple quality score (has content)
                quality = 1.0 if len(reply) > 50 else 0.5
                
                return True, elapsed, quality
                
            except Exception as e:
                print(f"❌ Request {request_id} failed: {e}")
                return False, 0.0, 0.0
        
        # Run concurrent requests
        print(f"\n🔄 Testing {num_concurrent} concurrent requests...")
        
        with ThreadPoolExecutor(max_workers=num_concurrent) as executor:
            futures = [
                executor.submit(process_request, i)
                for i in range(num_concurrent)
            ]
            
            for future in as_completed(futures):
                success, elapsed, quality = future.result()
                metrics.record_response(elapsed, success, quality)
        
        # Print results
        summary = metrics.get_summary()
        print(f"\n📊 Concurrent Request Results:")
        print(f"   Total: {summary['total_requests']}")
        print(f"   Success rate: {summary['success_rate']:.1f}%")
        print(f"   Response times:")
        print(f"      Mean: {summary['response_times']['mean']:.3f}s")
        print(f"      P95: {summary['response_times']['p95']:.3f}s")
        print(f"      P99: {summary['response_times']['p99']:.3f}s")
        
        # All requests should succeed
        assert summary['success_rate'] == 100.0, "Some concurrent requests failed"
        
        # P95 should be reasonable
        assert summary['response_times']['p95'] < 15.0, "P95 response time too high"
    
    def test_sustained_load(self, supervisor, state_manager):
        """Test sustained load over time"""
        num_requests = 20
        delay_between = 0.5  # seconds
        
        metrics = PerformanceMetrics()
        
        print(f"\n⏱️ Testing sustained load ({num_requests} requests)...")
        
        for i in range(num_requests):
            try:
                session_id = f"sustained_{i}"
                state = ConversationState(
                    session_id=session_id,
                    persona_id="practical",
                    context={}
                )
                
                start = time.time()
                state, reply = supervisor.handle(state, "ขอคำแนะนำเรื่องร้านอาหาร")
                elapsed = time.time() - start
                
                quality = 1.0 if len(reply) > 50 else 0.5
                metrics.record_response(elapsed, True, quality)
                
                time.sleep(delay_between)
                
            except Exception as e:
                print(f"❌ Request {i} failed: {e}")
                metrics.record_response(0.0, False, 0.0)
        
        # Print results
        summary = metrics.get_summary()
        print(f"\n📊 Sustained Load Results:")
        print(f"   Total requests: {summary['total_requests']}")
        print(f"   Success rate: {summary['success_rate']:.1f}%")
        print(f"   Avg response time: {summary['response_times']['mean']:.3f}s")
        print(f"   Avg quality: {summary['quality']['avg_score']:.2f}")
        
        # Should maintain high success rate
        assert summary['success_rate'] > 95.0, "Success rate dropped under sustained load"


class TestMemoryLeaks:
    """Test for memory leaks during extended use"""
    
    def test_session_cleanup(self, state_manager):
        """Test that old sessions are cleaned up"""
        # Create many sessions
        num_sessions = 100
        
        print(f"\n🗑️ Creating {num_sessions} sessions...")
        
        for i in range(num_sessions):
            session_id = f"cleanup_test_{i}"
            state = ConversationState(
                session_id=session_id,
                persona_id="practical",
                context={"test": "data"}
            )
            state_manager.save(session_id, state)
        
        # List sessions
        sessions = state_manager.list_sessions()
        print(f"   Sessions created: {len(sessions)}")
        
        # Cleanup old sessions (use 0 days to force cleanup)
        state_manager.purge_older_than_days(0)
        
        # Check cleanup worked
        remaining = state_manager.list_sessions()
        print(f"   Sessions after cleanup: {len(remaining)}")
        
        assert len(remaining) < num_sessions, "Cleanup did not remove old sessions"


class TestChaosEngineering:
    """Test system resilience under failure conditions"""
    
    def test_handle_missing_retriever_docs(self, supervisor, state_manager):
        """Test handling when retriever returns no documents"""
        session_id = "chaos_no_docs"
        state = ConversationState(
            session_id=session_id,
            persona_id="practical",
            context={}
        )
        
        # Ask question that might not match any docs
        state, reply = supervisor.handle(state, "xyz123abc nonexistent topic xyz")
        
        print(f"\n🔍 No docs scenario: {reply[:150]}...")
        
        # Should handle gracefully, not crash
        assert reply, "Should provide fallback response when no docs found"
        assert len(reply) > 20, "Fallback response should be meaningful"
    
    def test_handle_malformed_state(self, supervisor, state_manager):
        """Test handling of corrupted state"""
        session_id = "chaos_bad_state"
        
        # Create state with unexpected structure
        state = ConversationState(
            session_id=session_id,
            persona_id="practical",
            context={"corrupted": {"nested": {"deep": "data"}}}
        )
        
        # Should handle without crashing
        try:
            state, reply = supervisor.handle(state, "สวัสดี")
            success = True
        except Exception as e:
            print(f"❌ Failed on malformed state: {e}")
            success = False
        
        assert success, "Should handle malformed state gracefully"
    
    def test_handle_rapid_persona_switching(self, supervisor, state_manager):
        """Test rapid switching between personas"""
        session_id = "chaos_rapid_switch"
        state = ConversationState(
            session_id=session_id,
            persona_id="practical",
            context={}
        )
        
        # Rapidly switch personas
        messages = [
            "สวัสดี",
            "ขอแบบละเอียด",  # Try to switch to academic
            "เอาแบบสั้นๆ",    # Switch back
            "ขอเชิงลึก",       # Switch again
        ]
        
        print(f"\n🔄 Rapid persona switching...")
        
        for msg in messages:
            try:
                state, reply = supervisor.handle(state, msg)
                print(f"   {msg[:20]}... → Persona: {state.persona_id}")
            except Exception as e:
                pytest.fail(f"Failed during rapid switching: {e}")
        
        # Should complete without errors
        assert True


class TestAnswerQualityUnderLoad:
    """Test that answer quality doesn't degrade under load"""
    
    def test_quality_consistency(self, supervisor, state_manager):
        """Test answer quality remains consistent under load"""
        num_tests = 10
        question = "จดทะเบียนร้านอาหารต้องทำอะไรบ้าง"
        
        answers = []
        
        print(f"\n📏 Testing quality consistency ({num_tests} iterations)...")
        
        for i in range(num_tests):
            session_id = f"quality_{i}"
            state = ConversationState(
                session_id=session_id,
                persona_id="practical",
                context={}
            )
            
            state, reply = supervisor.handle(state, question)
            answers.append(reply)
        
        # Analyze consistency
        lengths = [len(answer) for answer in answers]
        
        print(f"\n📊 Answer length statistics:")
        print(f"   Min: {min(lengths)}")
        print(f"   Max: {max(lengths)}")
        print(f"   Mean: {statistics.mean(lengths):.0f}")
        print(f"   Std dev: {statistics.stdev(lengths):.0f}")
        
        # Answers should be relatively consistent
        # (some variation is OK, but not wildly different)
        std_dev = statistics.stdev(lengths)
        mean_length = statistics.mean(lengths)
        
        # Coefficient of variation should be reasonable
        cv = (std_dev / mean_length) * 100
        print(f"   Coefficient of variation: {cv:.1f}%")
        
        assert cv < 50, f"Answer length too inconsistent (CV: {cv:.1f}%)"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-s"])
