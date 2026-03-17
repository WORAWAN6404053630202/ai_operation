# Production Readiness Testing Guide

## 🎯 Overview

Comprehensive test suite for Thai Regulatory AI system testing **both Practical and Academic personas** with real data analysis and quality metrics.

### Key Features
- ✅ **NO HARDCODED VALUES** - All data extracted from real ChromaDB
- ✅ **Metadata Analysis** - Analyzes actual fields (แนวคำตอบ, เงื่อนไขและหลักเกณฑ์, หัวข้อการดำเนินการย่อย)
- ✅ **E2E Conversation Testing** - Complete flows for both personas
- ✅ **Back-Navigation Support** - Tests user going back to previous choices
- ✅ **Quality Metrics** - RAGAS, BERTScore integration ready
- ✅ **Performance Testing** - Load testing, chaos engineering
- ✅ **Fresh Environment** - Follows user's workflow (rm -rf → ingest → test)

---

## 📁 Test Files Created

### **test_90_metadata_analysis.py**
**Purpose:** Analyze actual metadata structure from ChromaDB

**Tests:**
1. `test_metadata_fields_exist` - Verify critical metadata fields exist
2. `test_topic_fields_coverage` - Check topic field coverage (operation_topic, แนวคำตอบ, etc.)
3. `test_entity_type_normalization` - Validate entity_type data quality
4. `test_extract_menu_worthy_topics` - Extract real menu topics (NO hardcoding)
5. `test_slot_data_availability` - Check slot-related data (location, entity_type, business_type)
6. `test_document_content_length` - Validate document content quality
7. `test_thai_content_ratio` - Ensure Thai language content
8. `test_regulatory_keywords` - Check regulatory keyword presence

**Key Insight:** This test discovers what metadata fields actually exist and their quality, so menu generation uses REAL data.

---

### **test_91_conversation_quality.py**
**Purpose:** E2E conversation testing with quality validation

**Practical Persona Tests:**
1. `test_practical_greeting_to_answer_flow` - Complete flow: greeting → topic → slots → answer
2. `test_practical_back_navigation` - User can change topic mid-conversation ✅
3. `test_practical_follow_up_questions` - Context retention across 3+ turns
4. Answer validation:
   - Concise (< 500 words)
   - Has action items
   - Minimal citations

**Academic Persona Tests:**
1. `test_academic_full_flow` - Complete: trigger → slots → sections → answer
2. `test_academic_back_to_section_selection` - User can go back to choose more sections ✅
3. `test_academic_auto_return_to_practical` - Auto-return after completion
4. Answer validation:
   - Detailed (> 50 words)
   - Has evidence markers
   - Comprehensive structure

**Edge Cases:**
- Gibberish input handling
- Very long questions
- Mixed Thai-English

**RAGAS/BERTScore Integration:**
- Framework ready, pending dataset preparation

---

### **test_92_performance_quality.py**
**Purpose:** Performance testing under load

**Tests:**
1. `test_single_request_performance` - Baseline performance (< 5s greeting, < 10s answer)
2. `test_concurrent_requests` - 5 concurrent users (100% success rate)
3. `test_sustained_load` - 20 sequential requests (> 95% success rate)
4. `test_session_cleanup` - Memory leak detection
5. **Chaos Engineering:**
   - No documents found scenario
   - Malformed state handling
   - Rapid persona switching
6. `test_quality_consistency` - Answer quality remains stable under load

---

### **run_production_tests.py**
**Purpose:** Orchestrate all test phases with reporting

**Features:**
- Runs all test phases in sequence
- Generates comprehensive JSON report
- Shows pass/fail summary
- Option to continue on failure

**Report includes:**
- Environment info
- Per-phase duration
- Success/failure status
- Full stdout/stderr logs

---

### **setup_test_env.sh**
**Purpose:** Prepare testing environment (follows user's workflow)

**Steps:**
1. `rm -rf code/local_chroma_v2` - Clean old data
2. `python -m code.scripts.ingest_local` - Ingest fresh documents
3. Verify ChromaDB created successfully
4. Ready for testing

---

## 🚀 How to Run Tests

### **Option 1: Full Production Readiness Suite**

```bash
# Step 1: Setup fresh environment
cd /Users/w.worawan/Downloads/ai-operation-microservice3_v2ori
./code/tests_enterprise/setup_test_env.sh

# Step 2: Run all tests
python code/tests_enterprise/run_production_tests.py
```

**Output:**
- Phase-by-phase results
- Comprehensive report saved to `test_results/production_readiness_YYYYMMDD_HHMMSS.json`
- Overall pass/fail verdict

---

### **Option 2: Run Individual Test Files**

```bash
# Metadata analysis only
pytest code/tests_enterprise/test_90_metadata_analysis.py -v -s

# Conversation quality only
pytest code/tests_enterprise/test_91_conversation_quality.py -v -s

# Performance only
pytest code/tests_enterprise/test_92_performance_quality.py -v -s
```

---

### **Option 3: Run Specific Tests**

```bash
# Test practical conversation only
pytest code/tests_enterprise/test_91_conversation_quality.py::TestPracticalConversation -v -s

# Test academic flow only
pytest code/tests_enterprise/test_91_conversation_quality.py::TestAcademicConversation -v -s

# Test performance only
pytest code/tests_enterprise/test_92_performance_quality.py::TestPerformance -v -s
```

---

## 📊 Understanding Test Results

### **Metadata Analysis Results**

```
📊 Found 12 unique metadata fields:
   - operation_topic: 85.3% coverage (412/483)
   - แนวคำตอบ: 78.5% coverage (379/483)
   - หัวข้อการดำเนินการย่อย: 62.1% coverage (300/483)
   ...

✨ Menu-worthy topics found: 47
Top 10 topics:
   1. การจดทะเบียนพาณิชย์ (from: operation_topic, แนวคำตอบ)
   2. ใบอนุญาตประกอบกิจการร้านอาหาร (from: operation_topic)
   ...
```

**What to look for:**
- ✅ > 50% coverage on topic fields
- ✅ > 5 menu-worthy topics found
- ✅ Thai content ratio > 80%
- ❌ If coverage < 50%, data ingestion may have issues

---

### **Conversation Quality Results**

```
Turn 1 [START] - Persona: practical
  Bot: สวัสดีครับ! ผมพร้อมช่วยเรื่องร้านอาหารนะครับ...

Turn 2 [USER] - Persona: practical
  User: 1
  Bot: เรื่องการจดทะเบียนพาณิชย์ใช่ไหมครับ...

✅ Turn 1 - Greeting received
✅ Turn 2 - Topic selection successful
✅ Turn 3 - Slot answered

📊 Validating Practical answer quality...
   Word count: 245
   Has action items: True
   Citation markers: 1
```

**What to look for:**
- ✅ Practical: < 500 words, has action items
- ✅ Academic: > 50 words, has evidence
- ✅ Context retained across turns
- ✅ Back-navigation works

---

### **Performance Results**

```
⚡ Single Request Performance:
   Greeting: 0.432s
   Question: 2.156s

📊 Concurrent Request Results:
   Total: 5
   Success rate: 100.0%
   Response times:
      Mean: 2.341s
      P95: 3.012s
      P99: 3.102s
```

**What to look for:**
- ✅ Greeting < 5s
- ✅ Answer < 10s
- ✅ Success rate > 95%
- ✅ P95 < 15s

---

## 🎓 Test Philosophy

### **1. No Hardcoding**
All test data comes from actual ChromaDB metadata:
```python
# ❌ BAD: Hardcoded topics
topics = ["จดทะเบียน", "ใบอนุญาต"]

# ✅ GOOD: Extracted from metadata
topics = extract_from_metadata(collection_data, field='operation_topic')
```

### **2. Real Conversation Flows**
Tests simulate actual user behavior:
- Greeting → Topic → Slots → Answer
- Follow-up questions with context
- Changing mind mid-conversation
- Going back to previous choices

### **3. Both Personas Tested**
- **Practical:** Fast, concise, action-oriented
- **Academic:** Detailed, evidence-based, comprehensive

### **4. Quality Metrics**
- Length appropriateness
- Content structure
- Evidence presence
- Consistency under load

---

## 🔧 Troubleshooting

### **Problem: "No collections found in ChromaDB"**

**Solution:**
```bash
# Re-run setup
./code/tests_enterprise/setup_test_env.sh

# Verify ingestion worked
ls -la code/local_chroma_v2/
```

---

### **Problem: "RAGAS not available"**

**Solution:**
```bash
# Install dependencies
pip install ragas bert-score deepeval
```

Note: RAGAS tests are currently skipped (marked as placeholders). They require additional dataset preparation.

---

### **Problem: Tests fail with "retriever not found"**

**Solution:**
```bash
# Make sure you're using the fixture correctly
# The test should have:
@pytest.fixture(scope="module")
def retriever():
    from service.local_vector_store import get_retriever
    return get_retriever(fail_if_empty=False)
```

---

## 📈 Next Steps

### **Phase 1-5 Completed ✅**

1. ✅ **Phase 1:** Testing infrastructure setup
2. ✅ **Phase 2:** Answer quality tests (hallucination detection, completeness)
3. ✅ **Phase 3:** E2E conversation tests (multi-turn, back-navigation)
4. ✅ **Phase 4:** Performance & monitoring (load testing, chaos engineering)
5. ✅ **Phase 5:** Ready for production readiness report

### **Additional Improvements (Optional)**

1. **RAGAS Full Integration**
   - Prepare reference dataset
   - Calculate faithfulness scores
   - Set quality thresholds

2. **BERTScore Reference Answers**
   - Create golden answer dataset
   - Benchmark semantic similarity

3. **Real-time Monitoring**
   - Integrate with Grafana dashboard
   - Alert on quality degradation

4. **A/B Testing**
   - Compare prompt variations
   - Measure user satisfaction

---

## 📝 Test Maintenance

### **Adding New Tests**

```python
# In test_91_conversation_quality.py

class TestNewFeature:
    """Test new feature description"""
    
    def test_new_behavior(self, supervisor, state_manager, fresh_session):
        """Test specific behavior"""
        tester = ConversationTester(supervisor, state_manager)
        
        # Your test logic here
        ...
```

### **Updating Quality Thresholds**

If performance improves, update thresholds in tests:

```python
# test_92_performance_quality.py
assert greeting_time < 3.0  # Updated from 5.0s
assert answer_time < 5.0    # Updated from 10.0s
```

---

## 🎯 Success Criteria

**System is production-ready when:**

- ✅ All metadata tests pass (> 50% coverage on topic fields)
- ✅ All conversation tests pass (both personas)
- ✅ Performance tests pass (< 10s answer time, > 95% success rate)
- ✅ Chaos engineering tests pass (graceful degradation)
- ✅ Quality metrics stable under load

---

## 📞 Support

For questions or issues:
1. Check this documentation
2. Review test output logs
3. Check `test_results/` directory for detailed reports
4. Review existing tests in `tests_enterprise/` for examples

---

**Last Updated:** March 16, 2026
**Test Suite Version:** 1.0
**Compatible with:** Python 3.11+, ChromaDB 0.5.0+
