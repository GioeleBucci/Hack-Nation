"""Handoff schemas for everything agents write to the research record.

Agents may only write the kinds in ``AGENT_KINDS``. The remaining kinds (plans, experiments,
results, stats, recommendations, approvals) are written by tools, so a result can never be
typed in by a model. See docs/record.md.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from firelab.placement import STRATEGIES

SOURCE_ID = re.compile(r"^(openalex:W\d+|doi:10\.\S+/\S+)$")


class _Item(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str | None = Field(None, description="Leave empty to get the next id; reuse an id to write a new version.")


class EvidenceCard(_Item):
    claim: str
    source_id: str = Field(description="openalex:W123... or doi:10.xxxx/...")
    title: str | None = None
    year: int | None = None
    summary: str
    confidence: Literal["low", "medium", "high"]
    gap: bool = Field(False, description="True if this card records an open gap rather than a finding.")

    @field_validator("source_id")
    @classmethod
    def _source(cls, v: str) -> str:
        v = v.strip()
        if v.lower().startswith("https://openalex.org/"):
            v = "openalex:" + v.rsplit("/", 1)[-1]
        if v.lower().startswith("https://doi.org/"):
            v = "doi:" + v[len("https://doi.org/"):]
        if not SOURCE_ID.match(v):
            raise ValueError("source_id must look like 'openalex:W123456' or 'doi:10.1234/abc'")
        return v


class Hypothesis(_Item):
    statement: str
    label: Literal["agent-generated"]
    strategy: str = Field(description=f"One of {list(STRATEGIES)}")
    params: dict = Field(default_factory=dict)
    evidence_ids: list[str] = Field(default_factory=list)
    finding_ids: list[str] = Field(default_factory=list)
    metric: Literal["burned_ha", "asset_cells"] = "burned_ha"
    predicted_change_pct: float = Field(description="Predicted relative change vs the random control, e.g. -15.")
    falsified_if: str
    status: Literal["proposed", "testing", "supported", "rejected", "reopened", "revised"] = "proposed"
    parent_id: str | None = Field(None, description="Hypothesis this one revises.")
    revision_note: str | None = None

    @field_validator("strategy")
    @classmethod
    def _strategy(cls, v: str) -> str:
        if v not in STRATEGIES:
            raise ValueError(f"strategy must be one of {list(STRATEGIES)}")
        return v

    @model_validator(mode="after")
    def _grounded(self):
        if not self.evidence_ids and not self.finding_ids:
            raise ValueError("A hypothesis must cite at least one evidence_id or finding_id.")
        return self


class PlanSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    strategy: str
    params: dict = Field(default_factory=dict)
    seed: int = 0


class ExperimentOption(_Item):
    round: int = Field(ge=0, description="Loop round; a chosen option needs a rival option in the same round.")
    tests: list[str] = Field(min_length=1, description="Hypothesis ids this option tests.")
    design: Literal["screen", "deep", "sweep", "transfer", "other"]
    plans: list[PlanSpec] = Field(min_length=1, description="Plans to build, controls included.")
    budget_pct: float
    ignition_set: str = "train"
    weather_set: str = "train"
    n_ignitions: int = Field(ge=1)
    n_weather: int = Field(ge=1)
    cost_sims: int
    expected_learning: str
    feasibility: str
    chosen: bool = False
    why: str = Field(description="Why this option was chosen or not chosen.")

    @model_validator(mode="after")
    def _checks(self):
        expected = len(self.plans) * self.n_ignitions * self.n_weather
        if self.cost_sims != expected:
            raise ValueError(f"cost_sims must equal plans x n_ignitions x n_weather = {expected}")
        strategies = {p.strategy for p in self.plans}
        if not {"none", "random"} <= strategies:
            raise ValueError("Every option must include the 'none' and 'random' controls.")
        return self


class HypothesisUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    hypothesis_id: str
    observed_change_pct: float | None = None
    verdict: Literal["supported", "rejected", "inconclusive"]


class Finding(_Item):
    experiment_id: str
    stats_id: str | None = None
    summary: str
    hypothesis_updates: list[HypothesisUpdate] = Field(default_factory=list)
    surprise: bool = False
    reopen: list[str] = Field(default_factory=list, description="Hypothesis ids or assumptions to reopen.")
    confidence: Literal["low", "medium", "high"]


class Decision(_Item):
    based_on: list[str] = Field(min_length=1, description="Ids of findings/experiments this decision rests on.")
    decision: str
    rationale: str
    next: str = Field(description="The next experiment or action.")


class RiskFlag(_Item):
    plan_id: str
    category: Literal["escape", "smoke", "ecology", "assets", "model_limits", "feasibility", "other"]
    severity: Literal["low", "medium", "high"]
    description: str
    mitigation: str


AGENT_KINDS: dict[str, type[_Item]] = {
    "evidence": EvidenceCard,
    "hypothesis": Hypothesis,
    "experiment_option": ExperimentOption,
    "finding": Finding,
    "decision": Decision,
    "risk_flag": RiskFlag,
}

SYSTEM_KINDS = ("plan", "experiment", "result", "stats", "recommendation", "approval")

ID_PREFIX = {
    "evidence": "E", "hypothesis": "H", "experiment_option": "X", "finding": "F",
    "decision": "D", "risk_flag": "R", "experiment": "EXP", "stats": "ST", "recommendation": "REC",
    "approval": "A",
}
