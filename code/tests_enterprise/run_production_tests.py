"""
Test Runner Script for Production Readiness Testing
Runs all tests in proper sequence and generates comprehensive report
"""
import subprocess
import sys
import json
from pathlib import Path
from datetime import datetime
import os


class TestRunner:
    """Run test suite and generate reports"""
    
    def __init__(self):
        self.results = {
            "test_date": datetime.now().isoformat(),
            "environment": self._get_environment_info(),
            "phases": {}
        }
    
    def _get_environment_info(self):
        """Get environment information"""
        return {
            "python_version": sys.version,
            "platform": sys.platform,
            "cwd": os.getcwd()
        }
    
    def run_phase(self, phase_name: str, test_file: str, description: str):
        """Run a test phase"""
        print(f"\n{'='*80}")
        print(f"PHASE: {phase_name}")
        print(f"Description: {description}")
        print(f"{'='*80}\n")
        
        # Run pytest
        cmd = [
            sys.executable,
            "-m",
            "pytest",
            test_file,
            "-v",
            "--tb=short",
            "-s",
            "--color=yes"
        ]
        
        start_time = datetime.now()
        result = subprocess.run(cmd, capture_output=True, text=True)
        end_time = datetime.now()
        
        duration = (end_time - start_time).total_seconds()
        
        # Store results
        self.results["phases"][phase_name] = {
            "description": description,
            "test_file": test_file,
            "duration_seconds": duration,
            "exit_code": result.returncode,
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr
        }
        
        # Print output
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        if result.returncode == 0:
            print(f"\n✅ {phase_name} PASSED ({duration:.1f}s)")
        else:
            print(f"\n❌ {phase_name} FAILED ({duration:.1f}s)")
        
        return result.returncode == 0
    
    def generate_report(self):
        """Generate comprehensive test report"""
        print(f"\n\n{'='*80}")
        print("PRODUCTION READINESS TEST REPORT")
        print(f"{'='*80}\n")
        
        print(f"Test Date: {self.results['test_date']}")
        print(f"Environment: Python {sys.version.split()[0]} on {sys.platform}")
        print()
        
        # Phase summary
        print("Phase Results:")
        print("-" * 80)
        
        total_duration = 0
        phases_passed = 0
        phases_total = len(self.results["phases"])
        
        for phase_name, phase_data in self.results["phases"].items():
            status = "✅ PASS" if phase_data["success"] else "❌ FAIL"
            duration = phase_data["duration_seconds"]
            total_duration += duration
            
            if phase_data["success"]:
                phases_passed += 1
            
            print(f"{status} | {phase_name:40} | {duration:6.1f}s")
        
        print("-" * 80)
        print(f"Total: {phases_passed}/{phases_total} phases passed | {total_duration:.1f}s total")
        print()
        
        # Overall result
        overall_pass = phases_passed == phases_total
        
        if overall_pass:
            print("🎉 ALL TESTS PASSED - SYSTEM READY FOR PRODUCTION")
        else:
            print(f"⚠️  {phases_total - phases_passed} PHASE(S) FAILED - REVIEW REQUIRED")
        
        print()
        
        # Save detailed report
        report_file = Path("test_results") / f"production_readiness_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        report_file.parent.mkdir(exist_ok=True)
        
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        print(f"📄 Detailed report saved to: {report_file}")
        
        return overall_pass


def main():
    """Main test runner"""
    print("\n" + "="*80)
    print("PRODUCTION READINESS TEST SUITE")
    print("Testing both Practical and Academic personas")
    print("="*80 + "\n")
    
    runner = TestRunner()
    
    # Define test phases
    phases = [
        {
            "name": "Phase 0: Metadata Analysis",
            "file": "code/tests_enterprise/test_90_metadata_analysis.py",
            "description": "Analyze actual metadata structure and quality from ChromaDB"
        },
        {
            "name": "Phase 1: Conversation Quality",
            "file": "code/tests_enterprise/test_91_conversation_quality.py",
            "description": "Test E2E conversation flows, back-navigation, context retention"
        },
        {
            "name": "Phase 2: Performance & Load",
            "file": "code/tests_enterprise/test_92_performance_quality.py",
            "description": "Test performance under load, chaos engineering, quality consistency"
        },
        {
            "name": "Phase 3: Existing Contract Tests",
            "file": "code/tests_enterprise/test_00_contracts.py",
            "description": "Verify core contracts and interfaces"
        },
        {
            "name": "Phase 4: Practical Behavior",
            "file": "code/tests_enterprise/test_30_practical_behavior.py",
            "description": "Test Practical persona behavior"
        },
        {
            "name": "Phase 5: Academic Flow",
            "file": "code/tests_enterprise/test_40_academic_flow.py",
            "description": "Test Academic persona complete flow"
        }
    ]
    
    # Run each phase
    all_passed = True
    for phase in phases:
        success = runner.run_phase(
            phase["name"],
            phase["file"],
            phase["description"]
        )
        
        if not success:
            all_passed = False
            
            # Ask if should continue
            print("\n⚠️  This phase failed. Continue with remaining phases? (y/n): ", end="")
            if input().lower() != 'y':
                print("Test suite aborted.")
                break
    
    # Generate final report
    runner.generate_report()
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
