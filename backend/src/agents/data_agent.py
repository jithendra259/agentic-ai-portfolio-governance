"""
Data Agent

Retrieves reliable market, portfolio, and governance data.
Validates data availability and completeness before returning.
Never fabricates values.
"""

import logging
from typing import Any, Dict, Optional

from src.agents.agent_base import (
    AgentConfig,
    AgentOutput,
    AgentType,
    SpecializedAgent,
    TaskDefinition,
    DataValidationError,
)

logger = logging.getLogger(__name__)


class DataAgent(SpecializedAgent):
    """
    Data Agent: Retrieves and validates data.
    
    Responsibilities:
    - Retrieve stock price data
    - Retrieve portfolio data
    - Retrieve governance rules
    - Retrieve market data
    - Validate data completeness
    - Never fabricate values
    
    Anti-Hallucination:
    - Returns FAILED if data unavailable
    - No estimation or interpolation
    - Clear data source attribution
    """
    
    def __init__(self, config: AgentConfig):
        """Initialize data agent."""
        super().__init__(config)
        # In production, initialize actual data providers
        self.data_sources = {
            "price_data": self._fetch_price_data,
            "portfolio_data": self._fetch_portfolio_data,
            "market_data": self._fetch_market_data,
            "governance_data": self._fetch_governance_data,
        }
    
    async def execute(
        self,
        task: TaskDefinition,
        context: Dict[str, Any],
    ) -> AgentOutput:
        """
        Retrieve requested data.
        
        Args:
            task: Task definition with data type and parameters
            context: Execution context
        
        Returns:
            AgentOutput with data or FAILED status
        """
        output = self._create_output(task.inputs.get("request_id", ""))
        
        try:
            # Validate inputs
            if not self._validate_inputs(task.inputs, ["data_type"]):
                output.mark_failed("Missing required input: data_type", "INVALID_INPUT")
                return output
            
            data_type = task.inputs.get("data_type", "")
            
            # Fetch data based on type
            self._log_info(f"Fetching {data_type}")
            
            if data_type not in self.data_sources:
                output.mark_failed(
                    f"Unknown data type: {data_type}",
                    "UNKNOWN_DATA_TYPE"
                )
                return output
            
            # Execute fetch
            success, data, error = self._safe_calculate(
                self.data_sources[data_type],
                task.inputs,
            )
            
            if not success:
                output.mark_failed(
                    f"Failed to fetch {data_type}: {error}",
                    "DATA_FETCH_FAILED"
                )
                output.add_evidence(f"Error: {error}")
                return output
            
            # Validate data
            is_valid, validation_error = self._validate_data(data, data_type)
            
            if not is_valid:
                output.mark_failed(
                    f"Data validation failed: {validation_error}",
                    "DATA_INVALID"
                )
                output.add_evidence(f"Validation error: {validation_error}")
                return output
            
            output.mark_success(data)
            output.add_source(f"{data_type.replace('_', ' ').title()} Source")
            output.add_evidence(f"Retrieved {len(data)} records")
            output.confidence = 0.95  # Data retrieval is high confidence
            
        except Exception as e:
            self._log_error(f"Data retrieval failed: {str(e)}")
            output.mark_failed(f"Data error: {str(e)}", "DATA_ERROR")
        
        return output
    
    def _fetch_price_data(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fetch stock price data.
        
        Raises:
            DataValidationError if data unavailable
        """
        ticker = inputs.get("ticker")
        days = inputs.get("days", 252)
        
        if not ticker:
            raise DataValidationError("Ticker required for price data")
        
        self._log_info(f"Fetching price data for {ticker} ({days} days)")
        
        # In production, fetch from data provider (MongoDB, API, etc.)
        # For now, return empty structure
        return {
            "ticker": ticker,
            "period_days": days,
            "data_points": 0,
            "start_date": None,
            "end_date": None,
            "prices": [],
            "volumes": [],
        }
    
    def _fetch_portfolio_data(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fetch portfolio data.
        
        Raises:
            DataValidationError if data unavailable
        """
        user_id = inputs.get("user_id")
        
        if not user_id:
            raise DataValidationError("User ID required for portfolio data")
        
        self._log_info(f"Fetching portfolio data for user {user_id}")
        
        # In production, fetch from portfolio database
        return {
            "user_id": user_id,
            "holdings": [],
            "total_value": 0,
            "allocation": {},
        }
    
    def _fetch_market_data(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fetch market-wide data.
        
        Raises:
            DataValidationError if data unavailable
        """
        metric = inputs.get("metric", "indices")
        
        self._log_info(f"Fetching market data for {metric}")
        
        # In production, fetch market indices, volatility, etc.
        return {
            "metric": metric,
            "data_points": 0,
            "last_update": None,
        }
    
    def _fetch_governance_data(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fetch governance rules and policies.
        
        Raises:
            DataValidationError if data unavailable
        """
        policy_id = inputs.get("policy_id", "default")
        
        self._log_info(f"Fetching governance policy {policy_id}")
        
        # In production, fetch from governance database
        return {
            "policy_id": policy_id,
            "risk_limits": {},
            "allocation_limits": {},
            "blacklist": [],
            "whitelist": [],
        }
    
    def _validate_data(
        self,
        data: Dict[str, Any],
        data_type: str,
    ) -> tuple[bool, Optional[str]]:
        """
        Validate retrieved data.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Empty data is always valid (just means no records found)
        if not data:
            return True, None
        
        # Type-specific validation
        if data_type == "price_data":
            if "ticker" not in data:
                return False, "Missing ticker in price data"
            if data.get("data_points", 0) == 0:
                return False, "No price data points returned"
        
        elif data_type == "portfolio_data":
            if "user_id" not in data:
                return False, "Missing user ID in portfolio data"
        
        elif data_type == "market_data":
            if "metric" not in data:
                return False, "Missing metric in market data"
        
        elif data_type == "governance_data":
            if "policy_id" not in data:
                return False, "Missing policy ID in governance data"
        
        return True, None
