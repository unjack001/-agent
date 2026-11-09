"""Bind a fixed workflow to its authorized PDF materials before library access."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.materials.access import CasePolicy
from src.materials.models import PaperRef

if TYPE_CHECKING:
    from src.dispatch.models import CaseQuestion
    from src.handlers.base import HandlerContext


class CaseMaterials:
    """Resolve only current-case PDFs; never let a legacy resolver scan broadly."""

    def __init__(self, context: HandlerContext, question: CaseQuestion) -> None:
        self.context = context
        self.question = question
        self.policy = CasePolicy(context.config, question.case_id,
            scan_cache=getattr(context.library, 'material_scan_cache', None),
            scan_lock=getattr(context.library, '_lock', None))
        self._papers: list[PaperRef] | None = None

    def resolve(self) -> list[PaperRef]:
        if self._papers is not None:
            return list(self._papers)
        paths = self.policy.material_files(
            self.question.materials,
            allow_library_fallback=self.question.function == "Literature_Retrieval",
        )
        if not paths:
            self._papers = []
            return []
        refs = self.context.library.resolve_materials(
            [str(path) for path in paths],
            allow_library_fallback=False,
            strict_names=True,
        )
        expected = {path.resolve() for path in paths}
        actual = {ref.path.resolve() for ref in refs}
        if actual != expected or len(refs) != len(expected):
            raise PermissionError("material resolver returned PDFs outside the current case")
        for ref in refs:
            self.policy.authorize_paper(ref.path)
            if ref.filename != ref.path.name:
                raise ValueError("resolved PDF filename does not match its actual file")
        by_path = {ref.path.resolve(): ref for ref in refs}
        self._papers = [by_path[path.resolve()] for path in paths]
        if self.question.function == 'Literature_Retrieval':
            # Durable diagnostic snapshot; process-local scan cache is the
            # reused index, never trust a model-supplied manifest as authority.
            from src.submission.writer import ResultWriter, SubmissionError
            try:
                relative=f'{self.policy.work_prefix}/material-snapshot.json'
                self.policy.work_path(relative, for_write=True)
                ResultWriter(self.context.config).write_json(relative, {
                    'case_id': self.question.case_id,
                    'papers': [{'filename':p.filename,'path':str(p.path)} for p in self._papers]})
            except (OSError, ValueError, SubmissionError):
                pass
        return list(self._papers)
