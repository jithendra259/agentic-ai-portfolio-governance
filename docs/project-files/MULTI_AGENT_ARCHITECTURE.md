# Multi-Agent Adaptive Portfolio Governance System

## Architecture Overview

```
User Query
    ↓
Planner Agent
    ├─ Analyze intent
    ├─ Create execution plan
    ├─ Break into subtasks
    ├─ Prioritize & sequence
    └─ Store plan in audit log
    ↓
Task Execution Engine
    ├─ Task 1: Data Retrieval (Data Agent)
    ├─ Task 2: Technical Analysis (Technical Analysis Agent)
    ├─ Task 3: Regime Detection (Regime Agent)
    ├─ Task 4: Instability Analysis (Instability Agent)
    ├─ Task 5: Portfolio Optimization (Portfolio Agent)
    ├─ Task 6: Governance Validation (Governance Agent)
    └─ Task 7: Explainability Generation (Explainability Agent)
    ↓
Verification Framework
    ├─ Verification Agent validates all outputs
    ├─ Anti-hallucination rules enforced
    ├─ Evidence tracking required
    └─ Governance approval confirmed
    ↓
Response Construction
    └─ Response Agent assembles final answer
    ↓
User Response (with full audit trail)
```

## Agent Responsibilities

### 1. Planner Agent
**Role**: High-level task orchestration
**Inputs**: User query, context
**Outputs**: Execution plan (JSON)
**Validates**:
- Query clarity
- Task decomposition
- Dependency mapping

**Plan Structure**:
```json
{
  "request_id": "uuid",
  "user_intent": "string",
  "detected_intent_type": "string",
  "task_count": int,
  "tasks": [
    {
      "task_id": "T001",
      "name": "string",
      "agent": "string",
      "dependencies": ["T000"],
      "priority": int,
      "estimated_tokens": int,
      "timeout_seconds": int
    }
  ],
  "estimated_complexity": "LOW|MEDIUM|HIGH",
  "requires_governance_approval": bool,
  "timestamp": "ISO8601"
}
```

### 2. Data Agent
**Role**: Reliable data retrieval
**Inputs**: Ticker, data type, date range
**Outputs**: JSON with data or FAILED status
**Validates**:
- Data availability
- Data completeness
- Data freshness

**Never**:
- Fabricates values
- Interpolates missing data
- Makes assumptions

**Response Structure**:
```json
{
  "status": "SUCCESS|FAILED",
  "data_type": "string",
  "data": {},
  "metadata": {
    "source": "string",
    "retrieved_at": "ISO8601",
    "data_points": int,
    "missing_data": bool,
    "confidence": 0.0-1.0
  },
  "error": "string (if FAILED)"
}
```

### 3. Technical Analysis Agent
**Role**: Calculate technical indicators
**Inputs**: Price data, indicator configs
**Outputs**: Indicators + signals
**Calculates**:
- SMA, EMA, DEMA
- RSI, Stochastic
- MACD, Signal Line, Histogram
- Bollinger Bands, ATR, ADX
- VWAP, OBV

**Detects**:
- Golden Cross / Death Cross
- Breakouts / Breakdowns
- Trend Reversals
- Support / Resistance
- Consolidation Zones

**Response Structure**:
```json
{
  "indicator_name": "string",
  "values": [float],
  "signals": [
    {
      "date": "ISO8601",
      "type": "GOLDEN_CROSS|DEATH_CROSS|BREAKOUT|REVERSAL|etc",
      "strength": 0.0-1.0,
      "evidence": "string"
    }
  ],
  "calculation_method": "string",
  "lookback_period": int
}
```

### 4. Regime Detection Agent
**Role**: Market regime classification
**Inputs**: Technical indicators, market data
**Outputs**: Regime + confidence + evidence
**Detects**:
- Bull Market (SMA alignment + positive momentum)
- Bear Market (SMA inverted + negative momentum)
- Sideways Market (volatility range)
- High Volatility (VIX-like metric)
- Low Volatility (Bollinger Band contraction)

**Response Structure**:
```json
{
  "regime": "BULL|BEAR|SIDEWAYS|HIGH_VOL|LOW_VOL",
  "confidence": 0.0-1.0,
  "primary_evidence": ["string"],
  "secondary_evidence": ["string"],
  "regime_duration_days": int,
  "regime_strength": "WEAK|MODERATE|STRONG"
}
```

### 5. Instability Analysis Agent
**Role**: Detect market/portfolio instability
**Inputs**: Price series, correlation matrix, volatility history
**Outputs**: Instability score + components
**Computes**:
- Volatility Drift: Changes in realized volatility
- Correlation Drift: Changes in asset correlations
- Covariance Drift: Changes in covariance structure
- Rolling Sharpness: Degradation in risk-adjusted returns

**Response Structure**:
```json
{
  "composite_instability_score": 0.0-1.0,
  "level": "LOW|MEDIUM|HIGH",
  "components": {
    "volatility_drift": {
      "score": 0.0-1.0,
      "direction": "UP|DOWN|STABLE",
      "magnitude": float
    },
    "correlation_drift": {
      "score": 0.0-1.0,
      "changed_pairs": int
    },
    "covariance_drift": {
      "score": 0.0-1.0,
      "frobenius_norm": float
    }
  },
  "risk_warning": "string (if HIGH)"
}
```

### 6. Portfolio Optimization Agent
**Role**: Recommend portfolio adjustments
**Inputs**: Holdings, objectives, constraints, market data
**Outputs**: Allocation recommendation
**Validates**:
- Risk limits
- Allocation limits
- Rebalance frequency
- Tax implications (flagged)

**Response Structure**:
```json
{
  "current_allocation": {},
  "recommended_allocation": {},
  "changes": [
    {
      "ticker": "string",
      "current_weight": float,
      "recommended_weight": float,
      "change_amount": float,
      "rationale": "string"
    }
  ],
  "expected_return": float,
  "expected_volatility": float,
  "sharpe_ratio": float,
  "rebalance_needed": bool,
  "rebalance_urgency": "LOW|MEDIUM|HIGH"
}
```

### 7. Governance Agent
**Role**: Compliance validation
**Inputs**: Recommendation, governance rules, risk limits
**Outputs**: APPROVED | REJECTED | REQUIRES_REVIEW

**Validates**:
- Risk level <= policy maximum
- Allocation % <= policy limit
- Concentration rules
- Blacklist/Whitelist compliance
- AML/KYC requirements

**Response Structure**:
```json
{
  "status": "APPROVED|REJECTED|REQUIRES_REVIEW",
  "validation_results": [
    {
      "rule": "string",
      "status": "PASS|FAIL|WARNING",
      "message": "string"
    }
  ],
  "compliance_score": 0.0-1.0,
  "required_approvals": ["human_review"],
  "escalation_level": "NONE|LOW|MEDIUM|HIGH"
}
```

### 8. Explainability Agent
**Role**: Generate user-facing explanations
**Inputs**: All agent outputs, technical details
**Outputs**: Technical + beginner explanations

**Generates**:
- Why decision was made
- Which indicators contributed (with percentages)
- Which regime was detected (implications)
- Which governance rules were applied
- What risks were identified
- Confidence breakdowns

**Response Structure**:
```json
{
  "executive_summary": "string",
  "technical_explanation": {
    "key_indicators": [
      {
        "name": "string",
        "value": float,
        "contribution_pct": float,
        "interpretation": "string"
      }
    ],
    "regime_context": "string",
    "instability_impact": "string"
  },
  "beginner_explanation": "string",
  "confidence_breakdown": {
    "data_quality": 0.0-1.0,
    "indicator_agreement": 0.0-1.0,
    "regime_clarity": 0.0-1.0,
    "governance_certainty": 0.0-1.0,
    "overall": 0.0-1.0
  }
}
```

### 9. Verification Agent
**Role**: Anti-hallucination gatekeeper
**Inputs**: All intermediate outputs
**Outputs**: VERIFIED | BLOCKED with reason

**Verifies**:
- Data retrieved successfully
- Evidence exists for all claims
- Calculations completed
- Governance approval obtained
- Confidence >= threshold
- No contradictions

**Response Structure**:
```json
{
  "verification_status": "VERIFIED|BLOCKED",
  "checks_passed": int,
  "checks_failed": int,
  "blocked_items": [
    {
      "item": "string",
      "reason": "INSUFFICIENT_EVIDENCE|MISSING_DATA|LOW_CONFIDENCE|GOVERNANCE_REJECTED|CONTRADICTION",
      "evidence_available": bool
    }
  ],
  "timestamp": "ISO8601"
}
```

### 10. Response Agent
**Role**: Assemble final response
**Inputs**: All verified outputs
**Outputs**: User-facing response

**Response Structure**:
```json
{
  "response_id": "uuid",
  "timestamp": "ISO8601",
  "user_summary": "string",
  "findings": {
    "technical_analysis": {},
    "regime_analysis": {},
    "instability_analysis": {},
    "portfolio_recommendation": {}
  },
  "governance_review": {
    "status": "APPROVED|REJECTED|REQUIRES_REVIEW",
    "details": "string"
  },
  "recommendation": {
    "action": "BUY|SELL|HOLD|REBALANCE",
    "confidence": 0.0-1.0,
    "key_levels": {}
  },
  "risks": ["string"],
  "confidence_score": 0.0-1.0,
  "explanation": "string",
  "audit_trail": {
    "plan_id": "uuid",
    "task_results": {}
  }
}
```

## Anti-Hallucination Rules

**Rule 1: Never Invent Values**
- If data unavailable → return "Unable to determine"
- No extrapolation or estimation
- No "most likely" guesses

**Rule 2: Data-Driven Recommendations**
- Every recommendation requires source data
- Every signal requires evidence
- Every claim requires calculation proof

**Rule 3: Confidence Thresholding**
- If confidence < 0.5 → return warning
- If confidence < 0.3 → reject recommendation
- Flag low-confidence indicators

**Rule 4: Governance Pre-Approval**
- Zero recommendations without governance validation
- Even low-risk suggestions require check
- Escalate on governance failure

**Rule 5: Evidence Trail**
- Every claim links to source agent
- Every calculation shows method
- Every signal shows calculation details

**Rule 6: Contradiction Detection**
- If agents disagree → escalate to review
- If signals conflict → flag for user
- If confidence contradicts → highlight

**Rule 7: Completeness Check**
- If task incomplete → do not proceed
- If data partial → flag limitations
- If missing critical component → abort gracefully

## Task Execution Engine

### Dependency Management
```python
tasks = {
    "T001_data_retrieval": {
        "agent": "DataAgent",
        "depends_on": [],
        "priority": 1,
    },
    "T002_technical_analysis": {
        "agent": "TechnicalAnalysisAgent",
        "depends_on": ["T001_data_retrieval"],
        "priority": 2,
    },
    "T003_regime_detection": {
        "agent": "RegimeDetectionAgent",
        "depends_on": ["T002_technical_analysis"],
        "priority": 2,
    },
    "T004_instability_analysis": {
        "agent": "InstabilityAnalysisAgent",
        "depends_on": ["T001_data_retrieval", "T002_technical_analysis"],
        "priority": 2,
    },
    "T005_portfolio_optimization": {
        "agent": "PortfolioOptimizationAgent",
        "depends_on": ["T001_data_retrieval", "T002_technical_analysis", "T003_regime_detection"],
        "priority": 3,
    },
    "T006_governance_validation": {
        "agent": "GovernanceAgent",
        "depends_on": ["T005_portfolio_optimization"],
        "priority": 4,
    },
    "T007_explainability": {
        "agent": "ExplainabilityAgent",
        "depends_on": ["T002_technical_analysis", "T003_regime_detection", "T004_instability_analysis", "T005_portfolio_optimization", "T006_governance_validation"],
        "priority": 4,
    },
    "T008_verification": {
        "agent": "VerificationAgent",
        "depends_on": ["T001_data_retrieval", "T002_technical_analysis", "T003_regime_detection", "T004_instability_analysis", "T005_portfolio_optimization", "T006_governance_validation", "T007_explainability"],
        "priority": 5,
    },
    "T009_response": {
        "agent": "ResponseAgent",
        "depends_on": ["T008_verification"],
        "priority": 6,
    }
}
```

### Execution Flow
1. **Planner** creates task DAG
2. **Executor** respects dependencies
3. **Parallel** execution where possible
4. **Cascade** on failure (don't start dependent tasks)
5. **Rollback** state if verification fails

## Memory & Audit System

### Stored in MongoDB
```javascript
{
  request_id: UUID,
  user_id: string,
  timestamp: ISO8601,
  query: string,
  
  // Planning Phase
  plan: {
    task_count: int,
    tasks: [...],
    complexity: string
  },
  
  // Execution Phase
  task_results: {
    T001: { status, output, duration_ms },
    T002: { status, output, duration_ms },
    ...
  },
  
  // Verification Phase
  verification: {
    status: "VERIFIED|BLOCKED",
    checks_passed: int,
    checks_failed: int,
    blocked_reasons: [...]
  },
  
  // Final Response
  response: { ... },
  
  // Audit Trail
  decision_trail: [
    { component, action, timestamp, confidence },
    ...
  ]
}
```

### Auditability Features
- Every decision logged with component ID
- Confidence scores attached to each step
- Evidence links preserved
- Timestamps for latency analysis
- Task dependencies recorded
- Verification gates shown

## Adaptability

### Market Regime-Based Adaptation
```python
if regime == "BULL_MARKET":
    # More aggressive optimization
    portfolio_agent.risk_tolerance = 0.7
    response_agent.tone = "opportunistic"
    
elif regime == "BEAR_MARKET":
    # Defensive positioning
    portfolio_agent.risk_tolerance = 0.3
    response_agent.tone = "protective"
    
elif regime == "HIGH_VOLATILITY":
    # Tighter stops
    governance_agent.max_position = 0.05
    verification_agent.confidence_threshold = 0.7
```

### Instability-Based Adaptation
```python
if instability_score > 0.7:
    # High instability → higher verification thresholds
    verification_agent.confidence_threshold = 0.8
    planner.complexity_multiplier = 1.5
    explainability_agent.detail_level = "VERBOSE"
    
elif instability_score < 0.3:
    # Low instability → faster response
    planner.allow_shortcuts = True
    verification_agent.parallel_checks = True
```

### Portfolio Risk-Based Adaptation
```python
if portfolio_risk > policy_max:
    # High portfolio risk → more conservative
    optimization_agent.max_allocation = 0.03
    governance_agent.escalation_level = "HIGH"
    
else if portfolio_risk < policy_min:
    # Low portfolio risk → more aggressive
    optimization_agent.max_allocation = 0.10
    governance_agent.escalation_level = "LOW"
```

## Integration with Existing System

### Preserves Current Capabilities
- Technical indicators (from technical_indicators.py)
- Technical reports (from technical_report_generator.py)
- LangChain tools integration
- MongoDB memory layer
- LangGraph orchestration

### Adds New Layers
- Task planning and execution
- Multi-agent coordination
- Verification framework
- Comprehensive audit trail
- Adaptive behavior

### Migration Path
1. Keep existing orchestrator operational
2. Add multi-agent framework alongside
3. New queries route through multi-agent system
4. Legacy queries fall back to existing orchestrator
5. Gradually migrate all queries to multi-agent

## Implementation Priorities

**Phase 1 (Critical)**:
1. Agent base classes
2. Task execution engine
3. Planner + Data agent
4. Verification agent
5. Response agent

**Phase 2 (High)**:
1. Technical analysis agent
2. Regime detection agent
3. Instability analysis agent
4. Governance agent
5. Explainability agent

**Phase 3 (Medium)**:
1. Portfolio optimization agent
2. Memory/audit system
3. Adaptive behavior rules
4. Testing & validation

**Phase 4 (Polish)**:
1. Performance optimization
2. Caching & memoization
3. Parallel execution tuning
4. Monitoring & alerting

## Success Metrics

✅ **Hallucination Rate**: < 5% (verified claims without evidence)
✅ **Explanation Quality**: 90%+ user satisfaction on clarity
✅ **Governance Compliance**: 100% (zero policy violations)
✅ **Auditability**: 100% decision trail preservation
✅ **Response Latency**: < 10 seconds (90th percentile)
✅ **Task Completion**: 95%+ (non-critical tasks can fail gracefully)
✅ **Confidence Calibration**: ±10% (stated vs actual accuracy)
