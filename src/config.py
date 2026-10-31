"""Runtime configuration injected by the platform."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RuntimeConfig:
    """Filesystem roots and LLM gateway settings for one evaluation run."""

    data_root: Path
    result_root: Path
    team: str
    llm_url: str
    llm_api_key: str
    llm_model: str
    llm_timeout_s: float = 120.0

    def __post_init__(self) -> None:
        team = self.team.strip()
        if not team or team in (".", "..") or "/" in team or "\\" in team:
            raise ValueError("team must be a single directory name")
        if (
            isinstance(self.llm_timeout_s, bool)
            or not isinstance(self.llm_timeout_s, (int, float))
            or not math.isfinite(self.llm_timeout_s)
            or not 0 < self.llm_timeout_s <= 170
        ):
            raise ValueError("llm_timeout_s must be finite and between 0 and 170 seconds")
        object.__setattr__(self, "team", team)

    @classmethod
    def from_env(cls) -> "RuntimeConfig":
        result_root = Path(os.getenv("RESEARCH_RESULT_ROOT", "/output"))
        workspace_team = os.getenv("RESEARCH_WORKSPACE_TEAM")
        if workspace_team:
            # This is the platform's identity, even before its result directory
            # exists. Shared library folders must never override it.
            team = workspace_team
        else:
            env_team = os.getenv("TEAM_ID")
            inferred = _infer_team(result_root)
            if not env_team:
                team = inferred
            elif (result_root / "data" / env_team).is_dir() or inferred in (
                "local-team",
                env_team,
            ):
                team = env_team
            else:
                team = inferred
        return cls(
            data_root=Path(os.getenv("DATA_PATH", "/data")),
            result_root=result_root,
            team=team,
            # The platform documents CHALLENGE_LLM_* as the recommended names
            # and also injects the PLATFORM_* compatible names. Accept both so
            # either injection style reaches the same gateway.
            llm_url=(
                os.getenv("PLATFORM_LLM_URL")
                or os.getenv("CHALLENGE_LLM_BASE_URL")
                or ""
            ),
            llm_api_key=(
                os.getenv("PLATFORM_API_KEY")
                or os.getenv("CHALLENGE_LLM_API_KEY")
                or ""
            ),
            llm_model=(
                os.getenv("AGENT_MODEL_NAME")
                or os.getenv("PLATFORM_LLM_MODEL")
                or os.getenv("CHALLENGE_LLM_MODEL")
                or ""
            ),
            llm_timeout_s=float(os.getenv("AGENT_LLM_TIMEOUT_S") or "120"),
        )

    @property
    def team_dir(self) -> Path:
        return self.result_root / "data" / self.team

    @property
    def experiments_dir(self) -> Path:
        return self.data_root / "experiments"


def _infer_team(result_root: Path) -> str:
    """Infer the single team directory when the platform omits the team env."""
    base = result_root / "data"
    try:
        candidates = [
            child.name
            for child in base.iterdir()
            if child.is_dir()
            and child.name not in {"expert", "figure", "论文库", "papers", "paper_library"}
        ]
    except OSError:
        return "local-team"
    return candidates[0] if len(candidates) == 1 else "local-team"
