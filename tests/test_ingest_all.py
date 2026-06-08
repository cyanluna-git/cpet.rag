"""tests/test_ingest_all.py — scripts/ingest_all.py 단위 테스트 (mock — GPU/key/real PDF 불필요).

모든 테스트는 base 의존성만으로 동작한다 (pyarrow/lancedb 불필요):
  uv run pytest tests/test_ingest_all.py -q

패치 전략:
    ingest_all.main() 의 lazy import 중 `core.vectorstore` 는 pyarrow 를 module-level 에서
    임포트하므로 실제 모듈 로딩을 막아야 base-only 환경에서 통과한다.
    sys.modules 에 가짜 모듈(MagicMock)을 미리 주입하여 모든 `from core.vectorstore import ...`
    호출을 실제 pyarrow 없이 처리한다.

    JinaEmbedder / ProcessedRegistry / ingest_corpus 는 base 에서 임포트 가능하므로
    unittest.mock.patch 로 소스 모듈 네임스페이스를 직접 패치한다.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

from ingestion.pipeline import IngestResult

# ---------------------------------------------------------------------------
# Patch targets (base-safe modules — importable without extras)
# ---------------------------------------------------------------------------

_PATCH_EMBEDDER = "ingestion.embed.JinaEmbedder"
_PATCH_REGISTRY = "ingestion.load.ProcessedRegistry"
_PATCH_INGEST = "ingestion.ingest_corpus"

# ---------------------------------------------------------------------------
# sys.modules injection helpers
#
# `core.vectorstore` imports pyarrow at module level → cannot be patched via
# mock.patch in a base-only env. Instead, inject a fake module into sys.modules
# before the lazy import fires so the real pyarrow is never loaded.
# ---------------------------------------------------------------------------


def _fake_vectorstore_module(mock_store_instance: MagicMock) -> MagicMock:
    """core.vectorstore 를 대체할 가짜 모듈을 반환한다."""
    fake_mod = MagicMock()
    fake_mod.LanceDBStore.return_value = mock_store_instance
    return fake_mod


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CORPUS_HEADERS = [
    "doi",
    "title",
    "first_author",
    "year",
    "journal",
    "source",
    "file",
    "oa_status",
    "added_by",
    "added_at",
]


def _write_corpus(path: Path, rows: list[dict[str, str]]) -> None:
    """테스트용 임시 corpus CSV를 작성한다."""
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_CORPUS_HEADERS)
        writer.writeheader()
        for row in rows:
            full_row = {k: row.get(k, "") for k in _CORPUS_HEADERS}
            writer.writerow(full_row)


def _make_fake_pdf(pdf_dir: Path, filename: str) -> Path:
    """최소 가짜 PDF 파일을 생성한다."""
    p = pdf_dir / filename
    p.write_bytes(b"%PDF-1.4 fake content for test")
    return p


def _make_results(statuses: list[str]) -> list[IngestResult]:
    results = []
    for i, status in enumerate(statuses):
        results.append(
            IngestResult(
                paper_key=f"key_{i}",
                doi=f"10.0/{i}",
                status=status,  # type: ignore[arg-type]
                n_chunks=3 if status == "ingested" else 0,
                error="mock error" if status == "failed" else None,
            )
        )
    return results


def _mock_store() -> MagicMock:
    s = MagicMock()
    s.count.return_value = 0
    return s


# ---------------------------------------------------------------------------
# Context manager: inject fake core.vectorstore into sys.modules
# ---------------------------------------------------------------------------


class _fake_vs_modules:
    """sys.modules 에 가짜 vectorstore 모듈을 주입/복원하는 컨텍스트 매니저."""

    def __init__(self, mock_store_instance: MagicMock) -> None:
        self._mock_store_instance = mock_store_instance
        self._saved: dict[str, Any] = {}

    def __enter__(self) -> MagicMock:
        fake_mod = _fake_vectorstore_module(self._mock_store_instance)
        for key in ("core.vectorstore", "core.vectorstore.lancedb_store"):
            self._saved[key] = sys.modules.get(key)
            sys.modules[key] = fake_mod  # type: ignore[assignment]
        return fake_mod

    def __exit__(self, *args: Any) -> None:
        for key, original in self._saved.items():
            if original is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = original


# ---------------------------------------------------------------------------
# Test: --dry-run
# ---------------------------------------------------------------------------


class TestDryRun:
    """--dry-run: PDF 존재 여부만 점검, 실제 인입 호출 없음."""

    def test_dry_run_counts_pdfs(self, tmp_path: Path, capsys: Any) -> None:
        """PDF가 있는 논문 수를 정확히 세고 ingest_corpus 는 호출하지 않는다."""
        pdf_dir = tmp_path / "pdfs"
        pdf_dir.mkdir()

        # 3편: 2개는 PDF 존재, 1개는 없음
        _make_fake_pdf(pdf_dir, "paper_a.pdf")
        _make_fake_pdf(pdf_dir, "paper_b.pdf")

        corpus_csv = tmp_path / "corpus.csv"
        _write_corpus(
            corpus_csv,
            [
                {"doi": "10.1/a", "title": "Paper A", "source": "src_a", "file": "paper_a.pdf"},
                {"doi": "10.1/b", "title": "Paper B", "source": "src_b", "file": "paper_b.pdf"},
                {"doi": "10.1/c", "title": "Paper C", "source": "src_c", "file": "missing.pdf"},
            ],
        )

        ingest_called: list[Any] = []

        # dry-run 은 lazy import 분기 자체에 진입하지 않으므로 vectorstore 모듈 불필요
        with patch(_PATCH_INGEST, side_effect=lambda *a, **kw: ingest_called.append(True) or []):
            import scripts.ingest_all as mod

            ret = mod.main(["--pdf-dir", str(pdf_dir), "--corpus", str(corpus_csv), "--dry-run"])

        assert ret == 0, "dry-run 은 항상 0 반환"
        assert len(ingest_called) == 0, "dry-run 에서 ingest_corpus 가 호출됐습니다"

        out = capsys.readouterr().out
        # 2편 존재, 1편 누락
        assert "2" in out
        assert "1" in out

    def test_dry_run_no_store_created(self, tmp_path: Path) -> None:
        """dry-run 에서 LanceDBStore 생성자를 호출하지 않는다."""
        pdf_dir = tmp_path / "pdfs"
        pdf_dir.mkdir()
        _make_fake_pdf(pdf_dir, "paper_a.pdf")

        corpus_csv = tmp_path / "corpus.csv"
        _write_corpus(
            corpus_csv,
            [{"doi": "10.1/a", "title": "Paper A", "source": "src_a", "file": "paper_a.pdf"}],
        )

        store_init_calls: list[Any] = []
        mock_store_inst = MagicMock(side_effect=lambda *a, **kw: store_init_calls.append(True))

        # dry-run: lazy import 분기 진입 전에 리턴 → vectorstore 로딩 없음
        # sys.modules 주입은 필요 없지만 일관성을 위해 포함
        with _fake_vs_modules(mock_store_inst):
            import scripts.ingest_all as mod

            mod.main(["--pdf-dir", str(pdf_dir), "--corpus", str(corpus_csv), "--dry-run"])

        assert len(store_init_calls) == 0, "dry-run 에서 LanceDBStore 가 생성됐습니다"


# ---------------------------------------------------------------------------
# Test: ingest_corpus 인자 전달 검증
# ---------------------------------------------------------------------------


class TestIngestCorpusArgs:
    """main이 ingest_corpus를 올바른 인자로 호출하는지 monkeypatch로 검증한다."""

    def _setup(self, tmp_path: Path) -> tuple[Path, Path]:
        pdf_dir = tmp_path / "pdfs"
        pdf_dir.mkdir()
        _make_fake_pdf(pdf_dir, "paper_a.pdf")
        _make_fake_pdf(pdf_dir, "paper_b.pdf")
        _make_fake_pdf(pdf_dir, "paper_c.pdf")

        corpus_csv = tmp_path / "corpus.csv"
        _write_corpus(
            corpus_csv,
            [
                {"doi": "10.1/a", "title": "Paper A", "source": "src_a", "file": "paper_a.pdf"},
                {"doi": "10.1/b", "title": "Paper B", "source": "src_b", "file": "paper_b.pdf"},
                {"doi": "10.1/c", "title": "Paper C", "source": "src_c", "file": "paper_c.pdf"},
            ],
        )
        return pdf_dir, corpus_csv

    def test_ingest_corpus_called_with_correct_args(self, tmp_path: Path) -> None:
        """ingest_corpus가 have / pdf_dir / store / embedder / registry 를 받는다."""
        pdf_dir, corpus_csv = self._setup(tmp_path)

        mock_store_inst = _mock_store()
        mock_embedder_inst = MagicMock()
        mock_registry_inst = MagicMock()
        captured: dict[str, Any] = {}

        def fake_ingest(
            papers: list,
            pdf_dir_arg: Any,
            *,
            store: Any,
            embedder: Any,
            registry: Any,
            limit: Any,
            force: Any,
            **kw: Any,
        ) -> list:
            captured["papers"] = papers
            captured["pdf_dir"] = pdf_dir_arg
            captured["store"] = store
            captured["embedder"] = embedder
            captured["registry"] = registry
            captured["limit"] = limit
            captured["force"] = force
            return _make_results(["ingested"] * len(papers))

        with _fake_vs_modules(mock_store_inst):
            with patch(_PATCH_EMBEDDER, return_value=mock_embedder_inst):
                with patch(_PATCH_REGISTRY, return_value=mock_registry_inst):
                    with patch(_PATCH_INGEST, side_effect=fake_ingest):
                        import scripts.ingest_all as mod

                        ret = mod.main(["--pdf-dir", str(pdf_dir), "--corpus", str(corpus_csv)])

        assert ret == 0
        assert len(captured["papers"]) == 3  # 3 PDFs exist
        assert Path(captured["pdf_dir"]) == pdf_dir
        assert captured["limit"] is None  # default
        assert captured["force"] is False  # default

    def test_limit_passed_to_ingest_corpus(self, tmp_path: Path) -> None:
        """--limit 인자가 ingest_corpus 에 전달된다."""
        pdf_dir, corpus_csv = self._setup(tmp_path)

        captured: dict[str, Any] = {}

        def fake_ingest(
            papers: list,
            pdf_dir_arg: Any,
            *,
            store: Any,
            embedder: Any,
            registry: Any,
            limit: Any,
            force: Any,
            **kw: Any,
        ) -> list:
            captured["limit"] = limit
            return _make_results(["ingested"])

        with _fake_vs_modules(_mock_store()):
            with patch(_PATCH_EMBEDDER, return_value=MagicMock()):
                with patch(_PATCH_REGISTRY, return_value=MagicMock()):
                    with patch(_PATCH_INGEST, side_effect=fake_ingest):
                        import scripts.ingest_all as mod

                        mod.main(
                            ["--pdf-dir", str(pdf_dir), "--corpus", str(corpus_csv), "--limit", "2"]
                        )

        assert captured["limit"] == 2

    def test_force_passed_to_ingest_corpus(self, tmp_path: Path) -> None:
        """--force 인자가 ingest_corpus 에 전달된다."""
        pdf_dir, corpus_csv = self._setup(tmp_path)

        captured: dict[str, Any] = {}

        def fake_ingest(
            papers: list,
            pdf_dir_arg: Any,
            *,
            store: Any,
            embedder: Any,
            registry: Any,
            limit: Any,
            force: Any,
            **kw: Any,
        ) -> list:
            captured["force"] = force
            return _make_results(["ingested"])

        with _fake_vs_modules(_mock_store()):
            with patch(_PATCH_EMBEDDER, return_value=MagicMock()):
                with patch(_PATCH_REGISTRY, return_value=MagicMock()):
                    with patch(_PATCH_INGEST, side_effect=fake_ingest):
                        import scripts.ingest_all as mod

                        mod.main(
                            ["--pdf-dir", str(pdf_dir), "--corpus", str(corpus_csv), "--force"]
                        )

        assert captured["force"] is True


# ---------------------------------------------------------------------------
# Test: 누락 PDF 처리
# ---------------------------------------------------------------------------


class TestMissingPdf:
    """PDF 파일이 없는 논문은 필터에서 걸러지고 no_pdf 집계에 포함된다."""

    def test_missing_pdfs_filtered_before_ingest(self, tmp_path: Path, capsys: Any) -> None:
        """존재하지 않는 PDF paper는 have 목록에서 제외, ingest_corpus 에 전달 안 됨."""
        pdf_dir = tmp_path / "pdfs"
        pdf_dir.mkdir()
        _make_fake_pdf(pdf_dir, "present.pdf")

        corpus_csv = tmp_path / "corpus.csv"
        _write_corpus(
            corpus_csv,
            [
                {"doi": "10.1/a", "title": "Present", "source": "src_a", "file": "present.pdf"},
                {"doi": "10.1/b", "title": "Missing", "source": "src_b", "file": "missing.pdf"},
                {"doi": "10.1/c", "title": "No file", "source": "src_c", "file": ""},
            ],
        )

        captured_papers: list[Any] = []

        def fake_ingest(
            papers: list,
            pdf_dir_arg: Any,
            *,
            store: Any,
            embedder: Any,
            registry: Any,
            limit: Any,
            force: Any,
            **kw: Any,
        ) -> list:
            captured_papers.extend(papers)
            return _make_results(["ingested"] * len(papers))

        with _fake_vs_modules(_mock_store()):
            with patch(_PATCH_EMBEDDER, return_value=MagicMock()):
                with patch(_PATCH_REGISTRY, return_value=MagicMock()):
                    with patch(_PATCH_INGEST, side_effect=fake_ingest):
                        import scripts.ingest_all as mod

                        mod.main(["--pdf-dir", str(pdf_dir), "--corpus", str(corpus_csv)])

        # ingest_corpus 에는 PDF 존재하는 1편만 전달
        assert len(captured_papers) == 1
        assert captured_papers[0].doi == "10.1/a"

        out = capsys.readouterr().out
        # no_pdf 개수가 출력에 포함돼야 함
        assert "no_pdf" in out


# ---------------------------------------------------------------------------
# Test: 실패 CSV 저장
# ---------------------------------------------------------------------------


class TestFailuresCsv:
    """ingest_corpus 가 failed 결과를 반환할 때 failures CSV를 저장한다."""

    def test_failures_csv_written(self, tmp_path: Path, monkeypatch: Any) -> None:
        """failed IngestResult 가 있으면 failures CSV 를 저장한다."""
        pdf_dir = tmp_path / "pdfs"
        pdf_dir.mkdir()
        _make_fake_pdf(pdf_dir, "paper_a.pdf")

        corpus_csv = tmp_path / "corpus.csv"
        _write_corpus(
            corpus_csv,
            [{"doi": "10.1/a", "title": "Paper A", "source": "src_a", "file": "paper_a.pdf"}],
        )

        failures_path = tmp_path / "failures.csv"

        import scripts.ingest_all as mod

        monkeypatch.setattr(mod, "FAILURES_CSV", failures_path)

        def fake_ingest(
            papers: list,
            pdf_dir_arg: Any,
            *,
            store: Any,
            embedder: Any,
            registry: Any,
            limit: Any,
            force: Any,
            **kw: Any,
        ) -> list:
            return [
                IngestResult(
                    paper_key="key_a",
                    doi="10.1/a",
                    status="failed",
                    error="mock parse error",
                )
            ]

        with _fake_vs_modules(_mock_store()):
            with patch(_PATCH_EMBEDDER, return_value=MagicMock()):
                with patch(_PATCH_REGISTRY, return_value=MagicMock()):
                    with patch(_PATCH_INGEST, side_effect=fake_ingest):
                        ret = mod.main(["--pdf-dir", str(pdf_dir), "--corpus", str(corpus_csv)])

        # 실패가 있으므로 exit code 1
        assert ret == 1
        assert failures_path.exists(), "failures CSV 가 생성되지 않았습니다"

        with failures_path.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)

        assert len(rows) == 1
        assert rows[0]["doi"] == "10.1/a"
        assert "mock parse error" in rows[0]["error"]

    def test_no_failures_csv_when_no_failures(self, tmp_path: Path, monkeypatch: Any) -> None:
        """실패가 없으면 failures CSV를 생성하지 않는다."""
        pdf_dir = tmp_path / "pdfs"
        pdf_dir.mkdir()
        _make_fake_pdf(pdf_dir, "paper_a.pdf")

        corpus_csv = tmp_path / "corpus.csv"
        _write_corpus(
            corpus_csv,
            [{"doi": "10.1/a", "title": "Paper A", "source": "src_a", "file": "paper_a.pdf"}],
        )

        failures_path = tmp_path / "no_failures.csv"

        import scripts.ingest_all as mod

        monkeypatch.setattr(mod, "FAILURES_CSV", failures_path)

        def fake_ingest(
            papers: list,
            pdf_dir_arg: Any,
            *,
            store: Any,
            embedder: Any,
            registry: Any,
            limit: Any,
            force: Any,
            **kw: Any,
        ) -> list:
            return _make_results(["ingested"])

        with _fake_vs_modules(_mock_store()):
            with patch(_PATCH_EMBEDDER, return_value=MagicMock()):
                with patch(_PATCH_REGISTRY, return_value=MagicMock()):
                    with patch(_PATCH_INGEST, side_effect=fake_ingest):
                        ret = mod.main(["--pdf-dir", str(pdf_dir), "--corpus", str(corpus_csv)])

        assert ret == 0
        assert not failures_path.exists(), "실패 없는데 CSV가 생성됐습니다"
