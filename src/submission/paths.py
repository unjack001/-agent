"""Canonical submission paths, relative to the workspace root."""

from __future__ import annotations


def result_json(team: str, case_id: str) -> str:
    return f"data/{team}/{case_id}.json"


def survey_markdown(team: str, case_id: str) -> str:
    return f"data/expert/{case_id}-综述-{team}.md"


def idea_markdown(team: str, case_id: str) -> str:
    return f"data/expert/{case_id}-灵感Idea报告-{team}.md"


def figure_dir(team: str) -> str:
    return f"data/figure/{team}"


def figure_path(team: str, filename: str) -> str:
    return f"{figure_dir(team)}/{filename}"


def work_dir(team: str) -> str:
    return f"data/{team}/work"
