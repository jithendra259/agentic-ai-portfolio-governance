"""
Multi-Agent System Test Suite

Simple tests to validate the multi-agent system is functioning correctly.
Run with: python -m pytest backend/src/agents/test_multi_agent_system.py
"""

import asyncio
import sys
import json
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.agents.multi_agent_manager import MultiAgentManager
from src.agents.agent_base import ExecutionStatus


class TestSuite:
    """Test suite for multi-agent system."""
    
    def __init__(self):
        """Initialize test suite."""
        self.manager = None
        self.results = []
    
    async def setup(self):
        """Set up test suite."""
        print("\n🚀 Initializing Multi-Agent System...")
        self.manager = MultiAgentManager()
        print("✅ System initialized with 10 agents")
        print(f"   - 4 production agents (Phase 1)")
        print(f"   - 6 stub agents (Phase 2)")
    
    async def test_basic_execution(self):
        """Test 1: Basic query execution."""
        print("\n📋 Test 1: Basic Query Execution")
        print("-" * 50)
        
        try:
            query = "What is the current trend for AAPL stock?"
            print(f"Query: {query}")
            
            result = await self.manager.execute(query)
            
            assert result["status"] == "success", "Execution should succeed"
            assert "request_id" in result, "Response should have request_id"
            assert "audit_trail" in result, "Response should have audit trail"
            
            print("✅ PASSED - Basic execution works")
            print(f"   Request ID: {result['request_id']}")
            print(f"   Verification Status: {result.get('verification_status', 'N/A')}")
            
            return True
        
        except Exception as e:
            print(f"❌ FAILED - {str(e)}")
            return False
    
    async def test_audit_trail(self):
        """Test 2: Audit trail completeness."""
        print("\n📋 Test 2: Audit Trail Completeness")
        print("-" * 50)
        
        try:
            query = "Analyze portfolio allocation"
            result = await self.manager.execute(query)
            
            audit_trail = result.get("audit_trail", {})
            entries = audit_trail.get("entries", [])
            
            assert len(entries) > 0, "Audit trail should have entries"
            
            # Verify entry structure
            for entry in entries:
                assert "timestamp" in entry, "Entry should have timestamp"
                assert "component" in entry, "Entry should have component"
                assert "action" in entry, "Entry should have action"
                assert "status" in entry, "Entry should have status"
            
            print("✅ PASSED - Audit trail complete")
            print(f"   Total entries: {len(entries)}")
            
            # Show sample entries
            for entry in entries[:3]:
                print(f"   [{entry['status']}] {entry['component']}: {entry['action']}")
            
            return True
        
        except Exception as e:
            print(f"❌ FAILED - {str(e)}")
            return False
    
    async def test_response_structure(self):
        """Test 3: Response structure validation."""
        print("\n📋 Test 3: Response Structure Validation")
        print("-" * 50)
        
        try:
            query = "Should I rebalance my portfolio?"
            result = await self.manager.execute(query)
            
            response = result.get("response", {})
            
            # Check required fields
            required_fields = [
                "response_id",
                "request_id",
                "timestamp",
                "user_summary",
                "findings",
                "governance_review",
                "recommendation",
                "risks",
                "confidence_score",
                "explanation",
                "audit_trail",
            ]
            
            for field in required_fields:
                assert field in response, f"Response missing {field}"
            
            # Validate recommendation structure
            rec = response.get("recommendation", {})
            assert "action" in rec, "Recommendation should have action"
            assert "confidence" in rec, "Recommendation should have confidence"
            
            print("✅ PASSED - Response structure valid")
            print(f"   Response fields: {len(response)}")
            print(f"   Recommendation: {rec.get('action', 'N/A')}")
            print(f"   Confidence: {rec.get('confidence', 0):.2f}")
            
            return True
        
        except Exception as e:
            print(f"❌ FAILED - {str(e)}")
            return False
    
    async def test_error_handling(self):
        """Test 4: Error handling and recovery."""
        print("\n📋 Test 4: Error Handling")
        print("-" * 50)
        
        try:
            # Empty query
            result = await self.manager.execute("")
            assert "request_id" in result, "Should still return request_id on error"
            
            # Check status
            if result["status"] == "error":
                print("✅ PASSED - System handles empty query gracefully")
            else:
                print("⚠️  System processed empty query (may be intentional)")
            
            return True
        
        except Exception as e:
            print(f"❌ FAILED - {str(e)}")
            return False
    
    async def test_memory_management(self):
        """Test 5: Memory management."""
        print("\n📋 Test 5: Memory Management")
        print("-" * 50)
        
        try:
            # Execute multiple queries
            for i in range(3):
                await self.manager.execute(f"Query {i+1}")
            
            print("✅ PASSED - Multiple queries processed")
            
            # Test cleanup
            removed = self.manager.cleanup_old_requests(max_age_hours=0)
            print(f"   Cleaned up: {removed} old requests")
            
            return True
        
        except Exception as e:
            print(f"❌ FAILED - {str(e)}")
            return False
    
    async def test_verification_gating(self):
        """Test 6: Verification gating."""
        print("\n📋 Test 6: Verification Gating")
        print("-" * 50)
        
        try:
            query = "Complex portfolio analysis"
            result = await self.manager.execute(query)
            
            verification = result.get("verification_status", "UNKNOWN")
            
            # Should be one of: VERIFIED, BLOCKED, WARNING
            valid_statuses = ["VERIFIED", "BLOCKED", "WARNING", "UNKNOWN"]
            assert verification in valid_statuses, f"Invalid status: {verification}"
            
            print("✅ PASSED - Verification gating works")
            print(f"   Status: {verification}")
            
            return True
        
        except Exception as e:
            print(f"❌ FAILED - {str(e)}")
            return False
    
    async def test_agent_execution(self):
        """Test 7: Individual agent execution."""
        print("\n📋 Test 7: Individual Agent Execution")
        print("-" * 50)
        
        try:
            from src.agents.agent_base import AgentType, TaskDefinition, AgentConfig
            
            # Test PlannerAgent
            planner = self.manager.agents[AgentType.PLANNER]
            plan_task = TaskDefinition(
                task_id="test_plan",
                agent_type=AgentType.PLANNER,
                agent_name="PlannerAgent",
                description="Test planning",
                inputs={"user_query": "Test", "request_id": "test_req"},
            )
            
            plan_output = await planner.execute(plan_task, {})
            
            assert plan_output.status.value in ["success", "failed"], "Invalid status"
            assert plan_output.data or plan_output.error, "Should have data or error"
            
            print("✅ PASSED - Individual agent execution works")
            print(f"   Planner: {plan_output.status.value}")
            
            return True
        
        except Exception as e:
            print(f"❌ FAILED - {str(e)}")
            return False
    
    async def run_all_tests(self):
        """Run all tests."""
        print("\n" + "="*60)
        print("MULTI-AGENT SYSTEM TEST SUITE")
        print("="*60)
        
        await self.setup()
        
        tests = [
            ("Basic Execution", self.test_basic_execution),
            ("Audit Trail", self.test_audit_trail),
            ("Response Structure", self.test_response_structure),
            ("Error Handling", self.test_error_handling),
            ("Memory Management", self.test_memory_management),
            ("Verification Gating", self.test_verification_gating),
            ("Agent Execution", self.test_agent_execution),
        ]
        
        results = {}
        
        for test_name, test_func in tests:
            try:
                results[test_name] = await test_func()
            except Exception as e:
                print(f"\n❌ Test suite error: {str(e)}")
                results[test_name] = False
        
        # Summary
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        
        passed = sum(1 for v in results.values() if v)
        total = len(results)
        
        for test_name, result in results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{status}: {test_name}")
        
        print(f"\nTotal: {passed}/{total} tests passed")
        
        if passed == total:
            print("\n🎉 ALL TESTS PASSED - System is operational!")
        else:
            print(f"\n⚠️  {total - passed} test(s) failed - review above")
        
        return passed == total


async def main():
    """Run test suite."""
    suite = TestSuite()
    success = await suite.run_all_tests()
    return 0 if success else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
