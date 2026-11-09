"""Start the AgentHack-compatible HTTP runtime: GET /health and POST /solve."""

from src.pilot import ResearchPilotAgent
from src.runtime import FlexibleAgentRuntime


def main() -> None:
    FlexibleAgentRuntime(ResearchPilotAgent()).run(host="0.0.0.0", port=8080)
