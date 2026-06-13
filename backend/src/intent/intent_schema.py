from dataclasses import asdict, dataclass, field
from typing import Optional


MAIN_INTENTS = {
    "general_chat",
    "methodology_explanation",
    "portfolio_analysis",
    "advisory_allocation",
    "plot_request",
    "dashboard_navigation",
    "response_validation",
    "system_debug",
}

SUB_INTENTS = {
    "unknown",
    "data_quality",
    "stock_eda_full",
    "eda",
    "correlation_covariance",
    "instability_regime",
    "diversification",
    "risk_governance",
    "advisory_allocation",
    "graph_contagion",
    "hitl_governance",
    "backtesting",
    "smart_plot_selection",
    "full_plot_coverage",
    "response_validation",
    "chatbot_accuracy_debug",
}

RESPONSE_MODES = {"brief", "standard", "technical", "viva", "debug", "full_report"}


@dataclass
class RouterEntities:
    universe: Optional[str] = None
    tickers: list[str] = field(default_factory=list)
    current_weights: dict[str, float] = field(default_factory=dict)
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    benchmark: Optional[str] = None
    sectors: list[str] = field(default_factory=list)
    metrics: list[str] = field(default_factory=list)
    plot_names: list[str] = field(default_factory=list)
    response_mode: str = "standard"
    user_intent_keywords: list[str] = field(default_factory=list)


@dataclass
class RouterExecution:
    needs_math_agent: bool = False
    needs_optimizer: bool = False
    needs_allocation: bool = False
    needs_graph_network: bool = False
    needs_rag: bool = False
    needs_plot_selector: bool = False
    needs_validator: bool = True
    needs_frontend_trigger: bool = False


@dataclass
class RouterTarget:
    endpoint: Optional[str] = None
    default_tab: Optional[str] = None
    plot_mode: str = "none"
    max_plots: int = 0
    allowed_plots: list[str] = field(default_factory=list)


@dataclass
class RouterResponse:
    mode: str = "standard"
    include_formulas: bool = False
    include_traceability: bool = True
    max_bullets: int = 8


@dataclass
class RouterSafety:
    advisory_only: bool = True
    forbidden_terms_check: bool = True
    modules_skipped: list[str] = field(default_factory=list)
    rag_top_k: int = 3


@dataclass
class RouterCache:
    reuse_previous_analysis: bool = False
    analysis_id: Optional[str] = None


@dataclass
class RouterClarification:
    needed: bool = False
    question: Optional[str] = None


@dataclass
class RouterPlan:
    main_intent: str
    sub_intent: str
    confidence: float
    entities: RouterEntities = field(default_factory=RouterEntities)
    execution: RouterExecution = field(default_factory=RouterExecution)
    routing: RouterTarget = field(default_factory=RouterTarget)
    response: RouterResponse = field(default_factory=RouterResponse)
    safety: RouterSafety = field(default_factory=RouterSafety)
    cache: RouterCache = field(default_factory=RouterCache)
    clarification: RouterClarification = field(default_factory=RouterClarification)

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["confidence"] = round(float(self.confidence), 3)
        return payload
