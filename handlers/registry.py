"""B-only test image: retain other source files without enabling handlers."""

from src.handlers.base import CaseHandler, HandlerContext, HandlerError
from src.handlers.literature_retrieval import LiteratureRetrievalHandler

ENABLED_FUNCTIONS = ("Literature_Retrieval",)
DISABLED_FUNCTIONS = frozenset(("Chat", "Survey", "Reproduction", "Idea"))


class CategoryDisabledError(HandlerError):
    """This image intentionally refuses non-B tasks before doing task work."""


def ensure_enabled(function=None, case_id=None) -> None:
    if (isinstance(function, str) and function in DISABLED_FUNCTIONS or
            isinstance(case_id, str) and case_id.startswith(("A-", "C-", "D-", "E-"))):
        raise CategoryDisabledError("This test image enables B / Literature_Retrieval only")

HANDLER_CLASSES: dict[str, type[CaseHandler]] = {
    "Literature_Retrieval": LiteratureRetrievalHandler,
}


def build_handlers(context: HandlerContext) -> dict[str, CaseHandler]:
    return {name: handler_class(context) for name, handler_class in HANDLER_CLASSES.items()}
