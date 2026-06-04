import unittest
from fastapi.testclient import TestClient
import json
from api.main import app

class TestAnalyticsRouter(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_eda_endpoint(self):
        response = self.client.get("/api/analytics/eda?tickers=AAPL,MSFT,NVDA&start_date=2024-01-01&end_date=2024-06-30")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("tickers", data)
        self.assertIn("adjusted_close", data)
        self.assertIn("normalized_price", data)
        self.assertIn("log_returns", data)
        self.assertIn("correlation_heatmap", data)
        self.assertIn("rolling_correlation", data)
        print("EDA Endpoint Passed!")

    def test_instability_endpoint(self):
        response = self.client.get("/api/analytics/instability?tickers=AAPL,MSFT&start_date=2024-01-01&end_date=2024-06-30")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("instability_index", data)
        self.assertIn("regime_timeline", data)
        self.assertIn("volatility_spike", data)
        print("Instability Endpoint Passed!")

    def test_advisory_allocation_endpoint(self):
        response = self.client.get("/api/analytics/advisory-allocation?tickers=AAPL,MSFT&start_date=2024-01-01&end_date=2024-06-30")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("ticker_allocation", data)
        self.assertIn("sector_allocation", data)
        self.assertIn("advisory_pie", data)
        print("Advisory Allocation Endpoint Passed!")

    def test_diversification_endpoint(self):
        response = self.client.get("/api/analytics/diversification?tickers=AAPL,MSFT&start_date=2024-01-01&end_date=2024-06-30")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("hhi_index", data)
        self.assertIn("effective_holdings", data)
        self.assertIn("ticker_concentration", data)
        print("Diversification Endpoint Passed!")

    def test_risk_governance_endpoint(self):
        response = self.client.get("/api/analytics/risk-governance?tickers=AAPL,MSFT&start_date=2024-01-01&end_date=2024-06-30")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("drawdown_curves", data)
        self.assertIn("max_drawdown", data)
        self.assertIn("cvar_comparison", data)
        self.assertIn("sharpe_comparison", data)
        print("Risk Governance Endpoint Passed!")

    def test_contagion_endpoint(self):
        response = self.client.get("/api/analytics/contagion?tickers=AAPL,MSFT&start_date=2024-01-01&end_date=2024-06-30")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("nodes", data)
        self.assertIn("edges", data)
        self.assertIn("eigenvector_centrality", data)
        self.assertIn("sigmoid_curve", data)
        print("Contagion Endpoint Passed!")

    def test_agent_governance_endpoint(self):
        response = self.client.get("/api/analytics/agent-governance")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("pipeline_status", data)
        self.assertIn("audit_trail", data)
        self.assertIn("hitl_triggers", data)
        self.assertIn("compliance_matrix", data)
        print("Agent Governance Endpoint Passed!")

    def test_backtesting_endpoint(self):
        response = self.client.get("/api/analytics/backtesting?tickers=AAPL,MSFT&start_date=2024-01-01&end_date=2024-06-30")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("equity_curves", data)
        self.assertIn("performance", data)
        self.assertIn("ablation_study", data)
        print("Backtesting Endpoint Passed!")

if __name__ == '__main__':
    unittest.main()
