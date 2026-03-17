#!/bin/bash
# Production Readiness Test - Environment Setup Script
# Prepares fresh environment for testing as per user's workflow:
# 1. Clean local_chroma
# 2. Run ingest_local
# 3. Run test suite

set -e  # Exit on error

echo "=================================="
echo "Production Readiness Test Setup"
echo "=================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

cd "$PROJECT_ROOT"

echo -e "${YELLOW}Step 1: Cleaning local ChromaDB...${NC}"
if [ -d "code/local_chroma_v2" ]; then
    rm -rf code/local_chroma_v2
    echo -e "${GREEN}✅ Cleaned code/local_chroma_v2${NC}"
else
    echo "ℹ️  code/local_chroma_v2 does not exist, skipping"
fi

if [ -d "local_chroma_v2" ]; then
    rm -rf local_chroma_v2
    echo -e "${GREEN}✅ Cleaned local_chroma_v2${NC}"
else
    echo "ℹ️  local_chroma_v2 does not exist, skipping"
fi

echo ""
echo -e "${YELLOW}Step 2: Ingesting documents...${NC}"
python -m code.scripts.ingest_local

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Document ingestion successful${NC}"
else
    echo -e "${RED}❌ Document ingestion failed${NC}"
    exit 1
fi

echo ""
echo -e "${YELLOW}Step 3: Checking ChromaDB...${NC}"
if [ -d "code/local_chroma_v2" ]; then
    DOC_COUNT=$(find code/local_chroma_v2 -type f | wc -l)
    echo -e "${GREEN}✅ ChromaDB exists with $DOC_COUNT files${NC}"
else
    echo -e "${RED}❌ ChromaDB not created${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}=================================="
echo "✅ Environment setup complete!"
echo "=================================="${NC}
echo ""
echo "You can now run tests with:"
echo "  python code/tests_enterprise/run_production_tests.py"
echo ""
echo "Or run individual test files:"
echo "  pytest code/tests_enterprise/test_90_metadata_analysis.py -v -s"
echo "  pytest code/tests_enterprise/test_91_conversation_quality.py -v -s"
echo "  pytest code/tests_enterprise/test_92_performance_quality.py -v -s"
