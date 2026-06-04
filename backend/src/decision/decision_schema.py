from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class DecisionRequest(BaseModel):
    message: str = ""
    tickers: list[str] = Field(default_factory=list)
    current_weights: dict[str, float] = Field(default_factory=dict)
    metrics: dict[str, float] = Field(default_factory=dict)
    graph: dict[str, Any] = Field(default_factory=dict)
    previous_analysis: dict[str, Any] | None = None
    use_sample_data_when_missing: bool = True
    decision_context: dict[str, Any] = Field(default_factory=dict)


class ValidationRequest(BaseModel):
    response_text: str = ""
    decision: dict[str, Any] = Field(default_factory=dict)


class RegimeResult(BaseModel):
    label: str
    instability_index: float | None = None
    threshold: str
    confidence: float


class MethodSelection(BaseModel):
    method: str
    reason_codes: list[str] = Field(default_factory=list)
    confidence: float = 0.8


class HitlAction(BaseModel):
    required: bool
    level: str
    reason_codes: list[str] = Field(default_factory=list)
    message: str


class ValidationResult(BaseModel):
    valid: bool
    issues: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class PlotSelection(BaseModel):
    plot_id: str
    title: str
    reason: str
    tab: str


class DecisionTrace(BaseModel):
    decision_id: str
    claims: list[dict[str, Any]] = Field(default_factory=list)
    source_modules: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
