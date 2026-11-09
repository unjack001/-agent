"""Local, page-wise OCR. No network or changes to mounted input files."""
from functools import lru_cache
from pathlib import Path
import shutil
import subprocess
import os
import math
from tempfile import TemporaryDirectory

from src.materials.models import MaterialError, normalize_evidence_text
from src.time_limits import remaining_seconds


def check_budget():
    left = remaining_seconds()
    if left <= 0:
        raise TimeoutError("PDF reading exhausted the current case budget")
    return left


def _command(args):
    timeout = check_budget()
    if not math.isfinite(timeout):
        timeout = None
    try:
        result = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                timeout=timeout, check=False, stdin=subprocess.DEVNULL,
                                env={**os.environ, "OMP_THREAD_LIMIT": "1"})
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError("PDF/OCR exhausted the current case budget") from exc
    except OSError as exc:
        raise MaterialError("PDF/OCR tool could not start") from exc
    if result.returncode:
        raise MaterialError(f"PDF/OCR tool {Path(args[0]).name} failed (exit={result.returncode})")
    return result.stdout


@lru_cache(maxsize=1)
def _tools():
    renderer, engine = shutil.which("pdftoppm"), shutil.which("tesseract")
    if not renderer or not engine:
        raise MaterialError("扫描版 PDF 需要 OCR，但镜像缺少 pdftoppm/tesseract")
    languages = set(_command([engine, "--list-langs"]).decode("utf-8", "replace").split())
    if not {"eng", "chi_sim"}.issubset(languages):
        raise MaterialError("OCR 缺少 eng/chi_sim 语言资源，无法可靠读取中英文扫描材料")
    return renderer, engine


def has_images(page):
    def visit(resources, depth=0):
        if not resources or depth > 6:
            return False
        resources = resources.get_object()
        objects = resources.get("/XObject")
        for value in objects.get_object().values() if objects else ():
            obj = value.get_object()
            if obj.get("/Subtype") == "/Image":
                return True
            if obj.get("/Subtype") == "/Form" and visit(obj.get("/Resources"), depth + 1):
                return True
        return False
    try:
        return visit(page.get("/Resources"))
    except Exception:
        return True


def read_page(pdf_path: Path, page_number: int) -> str:
    renderer, engine = _tools()
    with TemporaryDirectory(prefix="agent-pdf-ocr-") as temp:
        prefix = Path(temp) / "page"
        _command([renderer, "-f", str(page_number), "-l", str(page_number),
                  "-singlefile", "-r", "180", "-scale-to", "2400", "-gray",
                  str(pdf_path.resolve()), str(prefix)])
        image = prefix.with_suffix(".pgm")
        if not image.is_file():
            raise MaterialError("PDF renderer produced no OCR image")
        # Poppler emits a three-line P5 header, followed by grayscale bytes.
        pixels = image.read_bytes().split(b"\n", 3)[-1]
        if pixels and sum(value < 235 for value in pixels) / len(pixels) < 0.0005:
            return ""  # Scanned blank page.
        text = _command([engine, str(image), "stdout", "-l", "chi_sim+eng", "--psm", "3"])
        clean = normalize_evidence_text(text.decode("utf-8", "replace"))
        if not any(c.isalnum() for c in clean):
            raise MaterialError("OCR 未识别到正文；页面可能仅含图形或扫描质量不足")
        return clean
