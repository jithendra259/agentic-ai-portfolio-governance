import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)

class IntentType(str, Enum):
    """Exhaustive intent taxonomy for the advisory assistant."""
    GREETING = "greeting"
    LIST_SECTORS = "list_sectors"
    GET_STOCKS_BY_SECTOR = "get_stocks_by_sector"
    GET_STOCKS_BY_UNIVERSE = "get_stocks_by_universe"
    UNIVERSE_OVERVIEW = "universe_overview"
    STOCK_SNAPSHOT = "stock_snapshot"
    ANALYZE_PORTFOLIO = "analyze_portfolio"
    INSTITUTIONAL_NETWORK = "institutional_network"
    HISTORICAL_CVAR = "historical_cvar"
    FULL_PIPELINE_RUN = "full_pipeline_run"
    ROLLING_WINDOW_TEST = "rolling_window_test"
    EXPLAIN_PARAMETERS = "explain_parameters"
    METHODOLOGY_QUESTION = "methodology_question"
    DOCUMENTATION_REQUEST = "documentation_request"
    INVALID_EXECUTION = "invalid_execution"
    OUT_OF_SCOPE = "out_of_scope"
    ADVERSARIAL = "adversarial"
    MALFORMED = "malformed"

class RiskTier(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

@dataclass
class IntentMatch:
    intent: IntentType
    confidence: float
    risk_tier: RiskTier
    parameters: dict
    explanation: str
    requires_hitl: bool

    def to_dict(self) -> dict:
        return {
            "intent": self.intent.value,
            "confidence": round(self.confidence, 3),
            "risk_tier": self.risk_tier.value,
            "parameters": self.parameters,
            "explanation": self.explanation,
            "requires_hitl": self.requires_hitl,
        }

class IntentClassifier:
    """
    Deterministic intent classifier used as a governance-aware pre-routing gate.
    """

    KNOWN_SECTOR_ALIASES = {
        "basic materials": "Basic Materials",
        "communication services": "Communication Services",
        "consumer cyclical": "Consumer Cyclical",
        "consumer defensive": "Consumer Defensive",
        "energy": "Energy",
        "financial services": "Financial Services",
        "financials": "Financial Services",
        "finance": "Financial Services",
        "healthcare": "Healthcare",
        "health care": "Healthcare",
        "industrials": "Industrials",
        "real estate": "Real Estate",
        "technology": "Technology",
        "tech": "Technology",
        "utilities": "Utilities",
    }

    SECTOR_EXPLANATION_MARKERS = (
        "explain",
        "describe",
        "summary",
        "summarize",
        "overview",
        "about",
        "tell me",
    )

    GREETING_PATTERNS = [
        r"^(?:hi|hello|hey|hiya)$",
        r"^good\s+(?:morning|afternoon|evening)$",
    ]

    # EXPANDED: Security Gates to prevent trading
    INVALID_EXECUTION_PATTERNS = [
        "execute trades", "place orders", "place order",
        "buy ", "sell ", "liquidate", "rebalance live",
        "send trade", "broker", "short ", "go long", 
        "market order", "limit order", "invest in", 
        "purchase", "dump", "all in on"
    ]

    ADVERSARIAL_PATTERNS = [
        "ignore all instructions", "ignore previous instructions",
        "bypass", "override", "forget your rules",
        "clear memory", "sudo", "root access", "prompt injection",
    ]

    OUT_OF_SCOPE_PATTERNS = [
        "weather", "sports score", "capital of",
        "restaurant", "flight", "movie", "song",
    ]

    DATA_LOOKUP_PATTERNS = {
        IntentType.LIST_SECTORS: [
            r"\b(?:what|which|show|list|get)\b.*\b(?:sectors|industries)\b",
            r"\b(?:sectors|industries)\b.*\b(?:available|stored|database)\b",
            r"^(?:sector|sectors|industry|industries)(?:\s+list)?$",
            r"^(?:list\s+of\s+)?(?:sectors|industries)$",
        ],
        IntentType.GET_STOCKS_BY_SECTOR: [
            r"\b(?:show|get|list)\b(?:\s+me)?\s+(?P<sector>[a-z&\-\s]+?)\s+(?:stocks|companies|tickers)\b",
            r"\b(?:stocks|companies|tickers)\b\s+(?:in|from|within)\s+(?P<sector>[a-z&\-\s]+?)\s+sector\b",
        ],
        IntentType.GET_STOCKS_BY_UNIVERSE: [
            r"\b(?:what(?:'s| is)?|show|get|list)\b.*\b(?:in|from)\s+(?P<universe>u\d{1,2})\b",
            r"\b(?P<universe>u\d{1,2})\b\s+(?:universe|portfolio|constituents|members)\b",
            r"\b(?:stocks|tickers|companies)\b.*\b(?:in|from|of)\s+(?P<universe>u\d{1,2})\b",
        ],
        IntentType.UNIVERSE_OVERVIEW: [
            r"\b(?:summary|overview|composition|summarize)\b.*\b(?:of|for)\s+(?P<universe>u\d{1,2})\b",
            r"\b(?:tell|show)\b.*\babout\s+(?P<universe>u\d{1,2})\b",
        ],
        IntentType.STOCK_SNAPSHOT: [
            r"\b(?:snapshot|data|info|information|details)\b.*\b(?:for|on|about)\s+(?P<tickers>[a-z,\s]+)$",
            r"\b(?:show|get|list)\b.*\b(?:all stored|database)\b.*\b(?:tickers|stocks|data)\b",
            r"\b(?:brief|tell me)\b.*\b(?:company|stock|firm)\b.*\b(?:about|for|on)\s+(?P<tickers>[a-z,\s]+)\b",
            r"\b(?:what do we know about|summarize)\b\s+(?P<tickers>[a-z,\s]+)\b",
            r"\b(?:tell me more about|tell me about|more about)\b\s+(?P<tickers>[a-z,\s]+)\b",
            r"\b(?:explain|describe)\b\s+(?:ticker|stock|company|firm)\s+(?P<tickers>[a-z,\s]+)\b",
            r"\b(?:explain|describe)\b(?:\s+the)?\s+(?P<tickers>[a-z]{1,5}(?:\s*,\s*[a-z]{1,5})*)\b$",
            r"^(?P<tickers>[a-z]{1,5})\s*:\s*.*\b(?:explain|describe|summarize|tell me more|more about)\b.*$",
        ],
    }

    # EXPANDED: Governance and Advanced Analysis 
    GOVERNANCE_PATTERNS = {
        IntentType.ANALYZE_PORTFOLIO: [
            r"\b(?:analyze|analyse|assess|evaluate|review|audit|health check)\b\s+(?P<portfolio>.+?)\s+\b(?:for|on)\b\s+(?P<date>\d{4}-\d{2}-\d{2})\b",
            r"\b(?:allocation|weights|exposure|governance|risk report)\b.*\b(?:for|on)\s+(?P<portfolio>.+?)\s+\b(?:for|on)\b\s+(?P<date>\d{4}-\d{2}-\d{2})\b",
        ],
        IntentType.INSTITUTIONAL_NETWORK: [
            r"\b(?:institutional|holder|ownership|overlap|network|contagion|interconnectedness)\b.*\b(?:analysis|graph|visualization|risk)\b(?:\s+for\s+(?P<subject>.+))?",
            r"\b(?:show|get|analyze|map)\b.*\b(?:institutions|holders|systemic risk)\b.*\b(?:for|in|on)\s+(?P<subject>.+)$",
            r"\b(?:shared institutions?|ownership overlap|institutional overlap|contagion structure|graph context)\b.*\b(?:for|between|in|on)\s+(?P<subject>.+)$",
            r"\b(?:which institutions?)\b.*\b(?:connect|link|hold)\b.*\b(?:for|between|in|on)\s+(?P<subject>.+)$",
            r"\b(?:which institutions?)\b.*\b(?:connect|link|hold)\b\s+(?P<subject>.+)$",
        ],
        IntentType.HISTORICAL_CVAR: [
            r"\b(?:run|optimize|calculate|stress test)\b.*\b(?:cvar|historical cvar|tail risk|worst case)\b.*\b(?:for|on|with)\s+(?P<portfolio>.+?)\s+\b(?:for|on)\b\s+(?P<date>\d{4}-\d{2}-\d{2})\b",
            r"\b(?:optimal|recommended|safest)\b.*\b(?:allocation|weights)\b.*\b(?:for|on)\s+(?P<portfolio>.+?)\s+\b(?:for|on)\b\s+(?P<date>\d{4}-\d{2}-\d{2})\b",
        ],
    }

    BACKTEST_PATTERNS = {
        IntentType.FULL_PIPELINE_RUN: [
            r"\b(?:run|execute|backtest)\b.*\b(?:full|complete)\b.*\b(?:governance|pipeline)\b.*\b(?:all|11)\b.*\buniverses\b",
            r"\b(?:run|execute|backtest)\b.*\b(?:all|51|fifty[- ]one)\b.*\b(?:windows|rolling)\b",
        ],
        IntentType.ROLLING_WINDOW_TEST: [
            r"\b(?:run|execute|backtest)\b.*\b(?:rolling window|rolling windows)\b",
            r"\b(?:run|backtest)\b.*\b(?:51|fifty[- ]one)\b.*\b(?:rolling windows|windows)\b",
        ],
    }

    # EXPANDED: Documentation and Parameter requests
    CONFIG_PATTERNS = {
        IntentType.EXPLAIN_PARAMETERS: [
            r"\b(?:what|explain|define|clarify)\b.*\b(?:instability index|composite instability|i_t|lambda_t|graph penalty|threshold)\b",
            r"\b(?:how)\b.*\b(?:parameters|metrics|indices|penalties)\b.*\b(?:work|function|calculated)\b",
        ],
        IntentType.METHODOLOGY_QUESTION: [
            r"\b(?:how|why|what is)\b.*\b(?:g-cvar|optimizer|governance framework|methodology|the math)\b.*\b(?:work|function|based on)\b",
            r"\b(?:explain|describe|break down)\b.*\b(?:architecture|methodology|algorithm|approach|system)\b",
        ],
        IntentType.DOCUMENTATION_REQUEST: [
            r"\b(?:show|get|provide|link)\b.*\b(?:documentation|docs|help|guide|readme|paper|thesis)\b",
            r"\b(?:where|how)\b.*\b(?:documentation|instructions|architecture)\b",
        ],
    }

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.compiled_patterns = self._compile_patterns()

    def _compile_patterns(self) -> dict[IntentType, list[re.Pattern]]:
        compiled_patterns: dict[IntentType, list[re.Pattern]] = {}
        for intent_dict in (
            {IntentType.GREETING: self.GREETING_PATTERNS},
            self.DATA_LOOKUP_PATTERNS,
            self.GOVERNANCE_PATTERNS,
            self.BACKTEST_PATTERNS,
            self.CONFIG_PATTERNS,
        ):
            for intent, patterns in intent_dict.items():
                compiled_patterns[intent] = [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
        return compiled_patterns

    def classify(self, user_message: str) -> IntentMatch:
        raw_query = user_message or ""
        query = raw_query.strip()
        normalized_query = re.sub(r"\s+", " ", query.lower())

        if not normalized_query:
            return self._no_match("Empty query.", IntentType.MALFORMED)

        # SECURITY GATES (STRICT)
        if self._contains_any(normalized_query, self.ADVERSARIAL_PATTERNS):
            return self._no_match("Security gate: adversarial prompt detected.", IntentType.ADVERSARIAL)
        if self._contains_any(normalized_query, self.INVALID_EXECUTION_PATTERNS):
            return self._no_match("Security gate: trade execution request detected.", IntentType.INVALID_EXECUTION)

        # 1. FAST CATCH: Standalone Tickers (e.g., "NVDA")
        if re.match(r"^[A-Z]{1,5}$", query):
            return IntentMatch(
                intent=IntentType.STOCK_SNAPSHOT,
                confidence=1.0,
                risk_tier=RiskTier.LOW,
                parameters={"tickers": [query]},
                explanation="Fast-catch: Standalone ticker detected.",
                requires_hitl=False
            )

        # 2. FAST CATCH: Standalone Universes (e.g., "u1")
        univ_match = re.match(r"^(?P<univ>u\d{1,2})$", normalized_query)
        if univ_match:
            return IntentMatch(
                intent=IntentType.UNIVERSE_OVERVIEW,
                confidence=1.0,
                risk_tier=RiskTier.LOW,
                parameters={"universe": univ_match.group("univ").upper()},
                explanation="Fast-catch: Standalone universe detected.",
                requires_hitl=False
            )

        # 3. FAST CATCH: Leading ticker labels such as "ASX: ... explain this"
        labeled_ticker_match = re.match(r"^(?P<ticker>[A-Z]{1,5})\s*:\s*.+$", query)
        if labeled_ticker_match:
            return IntentMatch(
                intent=IntentType.STOCK_SNAPSHOT,
                confidence=0.97,
                risk_tier=RiskTier.LOW,
                parameters={"tickers": [labeled_ticker_match.group("ticker").upper()]},
                explanation="Fast-catch: Leading ticker label detected.",
                requires_hitl=False,
            )

        # 4. FAST CATCH: Known sector aliases and sector explanation phrasing
        sector_match = self._match_known_sector_query(normalized_query)
        if sector_match is not None:
            return sector_match

        # 5. REGEX PATTERN MATCHING (The existing logic)
        match = self._pattern_match(query)
        if match is not None:
            return match

        # 6. FLEXIBLE SEMANTIC FALLBACK
        # If any domain terms are present, let it pass to the LLM instead of blocking
        return self._semantic_fallback(normalized_query)

    def _contains_any(self, query: str, patterns: list[str]) -> bool:
        return any(pattern in query for pattern in patterns)

    def _pattern_match(self, query: str) -> Optional[IntentMatch]:
        best_match: Optional[IntentMatch] = None
        best_confidence = 0.0

        for intent, compiled_patterns in self.compiled_patterns.items():
            for pattern in compiled_patterns:
                match_obj = pattern.search(query)
                if not match_obj:
                    continue

                confidence = 0.95
                parameters = self._extract_parameters(query, match_obj, intent)
                risk_tier = self._classify_risk(intent)
                candidate = IntentMatch(
                    intent=intent,
                    confidence=confidence,
                    risk_tier=risk_tier,
                    parameters=parameters,
                    explanation=f"Pattern match: {pattern.pattern}",
                    requires_hitl=risk_tier in {RiskTier.HIGH, RiskTier.CRITICAL},
                )
                if confidence > best_confidence:
                    best_match = candidate
                    best_confidence = confidence

        return best_match if best_confidence >= 0.8 else None

    def _match_known_sector_query(self, normalized_query: str) -> Optional[IntentMatch]:
        matched_alias = None
        canonical_sector = None

        for alias, canonical in sorted(
            self.KNOWN_SECTOR_ALIASES.items(),
            key=lambda item: len(item[0]),
            reverse=True,
        ):
            if re.search(rf"\b{re.escape(alias)}\b", normalized_query):
                matched_alias = alias
                canonical_sector = canonical
                break

        if canonical_sector is None:
            return None

        cleaned_query = re.sub(r"[^\w\s&-]+", " ", normalized_query)
        cleaned_query = re.sub(r"\s+", " ", cleaned_query).strip()

        if cleaned_query == matched_alias:
            return IntentMatch(
                intent=IntentType.GET_STOCKS_BY_SECTOR,
                confidence=0.93,
                risk_tier=RiskTier.LOW,
                parameters={"sector": canonical_sector},
                explanation=f"Known sector shortcut: {canonical_sector}",
                requires_hitl=False,
            )

        if any(marker in cleaned_query for marker in self.SECTOR_EXPLANATION_MARKERS):
            return IntentMatch(
                intent=IntentType.MALFORMED,
                confidence=0.55,
                risk_tier=RiskTier.MEDIUM,
                parameters={"sector": canonical_sector},
                explanation="In-domain sector query needs clarification. Routing to LLM.",
                requires_hitl=False,
            )

        return None

    def _extract_parameters(self, query: str, match_obj: re.Match, intent: IntentType) -> dict:
        params: dict = {}
        groups = match_obj.groupdict()

        if intent == IntentType.GET_STOCKS_BY_SECTOR:
            sector = (groups.get("sector") or "").strip(" .?,")
            if sector:
                params["sector"] = sector

        elif intent in {IntentType.GET_STOCKS_BY_UNIVERSE, IntentType.UNIVERSE_OVERVIEW}:
            universe = (groups.get("universe") or "").upper()
            if universe:
                params["universe"] = universe

        elif intent == IntentType.STOCK_SNAPSHOT:
            params["tickers"] = self._parse_tickers(groups.get("tickers", query))

        elif intent in {IntentType.ANALYZE_PORTFOLIO, IntentType.HISTORICAL_CVAR}:
            portfolio_text = groups.get("portfolio", "")
            params["tickers"] = self._parse_tickers(portfolio_text)
            params["universes"] = self._parse_universes(portfolio_text)
            target_date = groups.get("date")
            if target_date:
                params["target_date"] = target_date

        elif intent == IntentType.INSTITUTIONAL_NETWORK:
            subject_text = (groups.get("subject") or query).strip()
            params["tickers"] = self._parse_tickers(subject_text)
            params["universes"] = self._parse_universes(subject_text)
            if params["universes"]:
                params["universe"] = params["universes"][0]

        elif intent in {IntentType.FULL_PIPELINE_RUN, IntentType.ROLLING_WINDOW_TEST}:
            params["scope"] = "batch"

        return params

    def _parse_tickers(self, text: str) -> list[str]:
        tickers = re.findall(r"\b([A-Z]{1,5})\b", str(text).upper())
        unique_tickers = []
        for ticker in tickers:
            if ticker.startswith("U") and ticker[1:].isdigit():
                continue
            if ticker not in unique_tickers:
                unique_tickers.append(ticker)
        return unique_tickers

    def _parse_universes(self, text: str) -> list[str]:
        universes = re.findall(r"\b(U\d{1,2})\b", str(text).upper())
        unique_universes = []
        for universe in universes:
            if universe not in unique_universes:
                unique_universes.append(universe)
        return unique_universes

    def _classify_risk(self, intent: IntentType) -> RiskTier:
        if intent in {
            IntentType.GREETING,
            IntentType.LIST_SECTORS,
            IntentType.GET_STOCKS_BY_SECTOR,
            IntentType.GET_STOCKS_BY_UNIVERSE,
            IntentType.UNIVERSE_OVERVIEW,
            IntentType.STOCK_SNAPSHOT,
            IntentType.INSTITUTIONAL_NETWORK,
            IntentType.EXPLAIN_PARAMETERS,
            IntentType.METHODOLOGY_QUESTION,
            IntentType.DOCUMENTATION_REQUEST,
        }:
            return RiskTier.LOW

        if intent in {
            IntentType.ANALYZE_PORTFOLIO,
            IntentType.HISTORICAL_CVAR,
        }:
            return RiskTier.HIGH

        if intent in {IntentType.FULL_PIPELINE_RUN, IntentType.ROLLING_WINDOW_TEST}:
            return RiskTier.CRITICAL

        return RiskTier.MEDIUM

    def _semantic_fallback(self, query: str) -> IntentMatch:
        domain_terms = [
            "analyze", "analyse", "evaluate", "assess", "portfolio",
            "universe", "ticker", "stock", "sector", "governance",
            "cvar", "risk", "instability", "institutional", 
            "stress test", "contagion", "governanace", "govrnance",
            "company", "firm", "brief", "jpm", "market", "info"
        ] + list(self.KNOWN_SECTOR_ALIASES.keys())
        if any(term in query for term in domain_terms):
            return IntentMatch(
                intent=IntentType.MALFORMED,
                confidence=0.45,
                risk_tier=RiskTier.MEDIUM,
                parameters={},
                explanation="In-domain query missing parameters. Routing to LLM.",
                requires_hitl=False,
            )

        return IntentMatch(
            intent=IntentType.MALFORMED,
            confidence=0.2,
            risk_tier=RiskTier.MEDIUM,
            parameters={},
            explanation="No deterministic intent matched. Routing to chatbot for best-effort retrieval.",
            requires_hitl=False,
        )

    def _no_match(self, explanation: str, intent: IntentType) -> IntentMatch:
        return IntentMatch(
            intent=intent,
            confidence=0.0,
            risk_tier=self._classify_risk(intent),
            parameters={},
            explanation=explanation,
            requires_hitl=False,
        )

    def _log(self, message: str) -> None:
        if self.verbose:
            logger.info(message)
