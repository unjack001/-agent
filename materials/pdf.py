"""PDF text/page extraction for allowed papers."""

from __future__ import annotations

from pathlib import Path
import logging
import shutil

from src.materials.models import MaterialError, normalize_evidence_text
from src.materials.ocr import check_budget, has_images, read_page
from src.materials.ocr import _command

logger = logging.getLogger(__name__)

MAX_TOTAL_CHARS = 500_000


def _native_pages(pdf_path: Path, max_pages: int | None) -> list[str] | None:
    """Use the bundled native PDF extractor, keeping form-feed page boundaries."""
    tool = shutil.which('pdftotext')
    if not tool:
        return None
    args = [tool, '-layout', '-enc', 'UTF-8', '-eol', 'unix']
    if max_pages is not None:
        args += ['-f', '1', '-l', str(max_pages)]
    try:
        text = _command([*args, str(pdf_path.resolve()), '-']).decode('utf-8')
    except (MaterialError, UnicodeDecodeError):
        return None  # The original extractor still diagnoses bad/encrypted PDFs.
    if not text.endswith('\f'):
        return None
    return text[:-1].split('\f')


def extract_pages(pdf_path: Path, max_pages: int | None = None, *,
                  prefer_native: bool = False, max_total_chars: int | None = MAX_TOTAL_CHARS) -> list[str]:
    """Preserve page numbers; use local OCR for sparse/broken image text layers."""
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise MaterialError("pypdf is not installed in the agent image") from exc

    try:
        check_budget()
        reader = PdfReader(str(pdf_path))
        if reader.is_encrypted:
            try:
                if reader.decrypt("") == 0:
                    raise MaterialError(f"材料问题：{pdf_path.name} 已加密，无法读取")
            except MaterialError:
                raise
            except Exception:
                raise MaterialError(f"材料问题：{pdf_path.name} 无法解密")

        native = _native_pages(pdf_path, max_pages) if prefer_native else None
        expected = min(len(reader.pages), max_pages) if max_pages is not None else len(reader.pages)
        if native is not None and len(native) != expected:
            native = None  # Never shift citation page numbers on a count mismatch.
        pages: list[str] = []
        total = 0
        for index, page in enumerate(reader.pages):
            if max_pages is not None and index >= max_pages:
                break
            check_budget()
            failed = False
            try:
                text = native[index] if native is not None else page.extract_text() or ""
                if native is not None:
                    contents = page.get_contents()
                    raw_size = len(contents.get_data()) if contents is not None else 0
                    # Native extractors may omit text outside the crop box or
                    # unsupported text operators. Use the original parser on
                    # suspiciously short pages, rather than losing late evidence.
                    if raw_size > max(1024, len(text) * 8):
                        text = page.extract_text() or text
            except Exception:
                text, failed = "", True
            text = text.strip()
            clean = normalize_evidence_text(text)
            meaningful = sum(c.isalnum() for c in clean)
            broken = bool(clean) and clean.count("\ufffd") > max(2, len(clean) // 20)
            if failed or broken or (has_images(page) and meaningful < 200) or (not meaningful and page.get("/Contents")):
                try:
                    text = read_page(pdf_path, index + 1)
                except MaterialError as exc:
                    raise MaterialError(f"材料问题：{pdf_path.name} 第 {index + 1} 页 OCR 失败：{exc}") from exc
                if not any(c.isalnum() for c in text) and (failed or broken or meaningful):
                    raise MaterialError(f"材料问题：{pdf_path.name} 第 {index + 1} 页 OCR 无可用正文")
                logger.info("PDF OCR filename=%s page=%d chars=%d", pdf_path.name, index + 1, len(text))
            pages.append(text if max_total_chars is None else text[:max_total_chars - total])
            total += len(pages[-1])
            if max_total_chars is not None and total >= max_total_chars:
                break
        if not any(any(c.isalnum() for c in page) for page in pages):
            raise MaterialError(f"材料问题：{pdf_path.name} 原生提取和可用 OCR 均未获得正文，请提供可解析材料")
        return pages
    except (MaterialError, TimeoutError):
        raise
    except Exception as exc:
        raise MaterialError(f"材料问题：无法读取 PDF {pdf_path.name} ({type(exc).__name__})") from exc
