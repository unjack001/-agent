"""Maps case function names to handler classes."""

from src.handlers.base import CaseHandler, HandlerContext
from src.handlers.chat import ChatHandler
from src.handlers.idea import IdeaHandler
from src.handlers.literature_retrieval import LiteratureRetrievalHandler
from src.handlers.reproduction import ReproductionHandler
from src.handlers.survey import SurveyHandler

HANDLER_CLASSES: dict[str, type[CaseHandler]] = {
    "Chat": ChatHandler,
    "Literature_Retrieval": LiteratureRetrievalHandler,
    "Survey": SurveyHandler,
    "Reproduction": ReproductionHandler,
    "Idea": IdeaHandler,
}


def build_handlers(context: HandlerContext) -> dict[str, CaseHandler]:
    return {name: handler_class(context) for name, handler_class in HANDLER_CLASSES.items()}
