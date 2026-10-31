"""Required top-level keys for each output_schema_ref."""

FUNCTION_BY_CASE_PREFIX: dict[str, str] = {
    "A": "Chat",
    "B": "Literature_Retrieval",
    "C": "Survey",
    "D": "Reproduction",
    "E": "Idea",
}

OUTPUT_FIELDS: dict[str, tuple[str, ...]] = {
    "Chat": ("case_id", "answer", "clarifying_question"),
    "Literature_Retrieval": ("case_id", "papers"),
    "Survey": ("case_id", "path", "markdown"),
    "Reproduction": ("case_id", "status", "metrics", "figures"),
    "Idea": ("case_id", "path", "markdown"),
}

REPRODUCTION_STATUSES = ("completed", "failed", "materials_missing")
