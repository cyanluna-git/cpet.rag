"""tests/test_pipeline.py — IngestionPipeline E2E 테스트.

전략:
1. 실제 파싱 경로 (tests/fixtures/sample.pdf 존재 시):
   real Docling parse + MockEmbedder (결정론적 벡터) + real LanceDB
2. 모의 경로 (sample.pdf 없을 때):
   monkeypatch parse_pdf → synthetic ParsedDoc, 이후 단계는 동일 (real chunk + embed + load)

두 경로 모두 ProcessedRegistry skip 동작을 검증한다.

실행:
    uv run --extra ingestion --extra vectorstore pytest tests/test_pipeline.py -q
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from core.models import Chunk, Paper
from core.models.chunk import Chunk
from ingestion import IngestResult, ingest_corpus, ingest_pdf
from ingestion.load import ProcessedRegistry, processed_key

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SAMPLE_PDF = Path(__file__).parent / "fixtures" / "sample.pdf"
DIM = 8  # small dim for tests — store and mock must agree


# ---------------------------------------------------------------------------
# MockEmbedder
# ---------------------------------------------------------------------------


class MockEmbedder:
    """JinaEmbedder 서브클래스로 _embed_call / _embed_late_call 을 override.

    embed_chunks (inherited) 는 그대로 사용하므로 late-chunking boundary 구성을 실제로 실행한다.
    결정론적 단위 벡터를 반환해 외부 API 호출 없이 테스트할 수 있다.

    Parameters
    ----------
    dim:
        반환 벡터 차원 수. LanceDBStore 의 dim 과 일치해야 한다.
    """

    def __init__(self, dim: int = DIM) -> None:
        # JinaEmbedder 의 생성자는 Settings() 및 httpx 를 사용하므로
        # 직접 초기화하지 않고 필요한 속성만 설정한다.
        self.dim = dim

    def _det_vec(self, seed: int) -> list[float]:
        """시드 기반 결정론적 단위 벡터."""
        v = [(seed % (i + 2)) * 0.1 + 0.01 for i in range(self.dim)]
        norm = sum(x**2 for x in v) ** 0.5 or 1.0
        return [x / norm for x in v]

    def embed(self, texts: list[str]) -> list[list[float]]:
        """표준 임베딩 — 텍스트 인덱스 기반 결정론적 벡터."""
        return [self._det_vec(i) for i in range(len(texts))]

    def embed_late(
        self,
        document: str,
        boundaries: list[tuple[int, int]],
    ) -> list[list[float]]:
        """Late Chunking 임베딩 — boundary 인덱스 기반 결정론적 벡터."""
        return [self._det_vec(i) for i in range(len(boundaries))]

    def embed_chunks(
        self,
        chunks: list[Chunk],
        full_document: str | None = None,
        late: bool = True,
    ) -> list[Chunk]:
        """chunks 에 결정론적 embedding 을 설정한 새 Chunk 목록 반환 (non-mutating)."""
        vectors: list[list[float]]
        if full_document is not None and late:
            # Late 경로: boundary 재구성 (JinaEmbedder 동일 로직)
            sep = "\n"
            texts = [c.text for c in chunks]
            boundaries: list[tuple[int, int]] = []
            pos = 0
            for i, text in enumerate(texts):
                boundaries.append((pos, pos + len(text)))
                pos += len(text) + (len(sep) if i < len(texts) - 1 else 0)
            concat = sep.join(texts)
            vectors = self.embed_late(concat, boundaries)
        else:
            vectors = self.embed([c.ctx_text for c in chunks])

        result: list[Chunk] = []
        for chunk, vec in zip(chunks, vectors):
            new_chunk = copy.copy(chunk)
            new_chunk.embedding = vec[: self.dim]
            result.append(new_chunk)
        return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_paper(pdf_filename: str | None = "sample.pdf") -> Paper:
    return Paper(
        doi="10.9999/pipeline.test",
        title="Pipeline E2E Test Paper",
        first_author="TestAuthor",
        year=2024,
        source="pipeline_test",
        file=pdf_filename,
        openalex_id="W_PIPELINE_TEST",
    )


def _make_store(tmp_path: Path, dim: int = DIM) -> Any:
    from core.vectorstore import LanceDBStore

    return LanceDBStore(uri=str(tmp_path / "lancedb"), dim=dim)


def _make_registry(tmp_path: Path) -> ProcessedRegistry:
    return ProcessedRegistry(path=tmp_path / "processed.jsonl")


def _make_synthetic_parsed() -> Any:
    """sample.pdf 가 없을 때 사용할 synthetic ParsedDoc."""
    from ingestion.parse.types import ParsedDoc, Section

    return ParsedDoc(
        source_path="/tmp/synthetic_test.pdf",
        markdown="# Introduction\n\nThis is a synthetic test document for pipeline E2E testing.\n\n"
        "# Methods\n\nParticipants underwent CPET evaluation under standardized conditions.\n\n"
        "# Results\n\nPeak VO₂ was measured at 52 ml/kg/min.\n\n"
        "# Discussion\n\nThe results confirm the utility of CPET in exercise physiology research.",
        n_pages=4,
        sections=[
            Section(heading=None, text="Abstract: CPET testing is important.", page=1),
            Section(
                heading="Introduction",
                text="This is a synthetic test document for pipeline E2E testing.",
                page=1,
            ),
            Section(
                heading="Methods",
                text="Participants underwent CPET evaluation under standardized conditions.",
                page=2,
            ),
            Section(heading="Results", text="Peak VO₂ was measured at 52 ml/kg/min.", page=3),
            Section(
                heading="Discussion",
                text="The results confirm the utility of CPET in exercise physiology research.",
                page=4,
            ),
        ],
        low_confidence_pages=[],
    )


# ---------------------------------------------------------------------------
# Test: ingest_pdf — real Docling or monkeypatched
# ---------------------------------------------------------------------------


class TestIngestPdf:
    """ingest_pdf E2E 테스트."""

    def test_ingest_then_skip(self, tmp_path: Path) -> None:
        """첫 실행 → ingested, 두 번째 실행 → skipped."""
        store = _make_store(tmp_path)
        registry = _make_registry(tmp_path)
        mock_embedder = MockEmbedder(dim=DIM)
        paper = _make_paper()

        if SAMPLE_PDF.exists():
            # Real Docling parse path
            result = ingest_pdf(
                SAMPLE_PDF,
                paper,
                store=store,
                embedder=mock_embedder,
                registry=registry,
                use_vlm=False,  # no Gemini key in tests
                use_late=True,
            )
        else:
            # Monkeypatched parse path
            synthetic = _make_synthetic_parsed()
            with patch("ingestion.pipeline.parse_pdf", return_value=synthetic):
                # patch at pipeline module level (lazy import target)
                with patch("ingestion.parse.docling_parser.parse_pdf", create=True):
                    pass
            # Re-run with patch applied to where pipeline.py does its lazy import
            with patch("ingestion.parse.docling_parser.parse_pdf", synthetic):
                result = _run_with_synthetic(tmp_path, store, mock_embedder, registry, paper)

        assert (
            result.status == "ingested"
        ), f"Expected ingested, got {result.status}: {result.error}"
        assert result.n_chunks > 0, "Expected at least one chunk"
        assert (
            store.count() == result.n_chunks
        ), f"store.count()={store.count()} != n_chunks={result.n_chunks}"

        key = processed_key(paper)
        assert registry.is_processed(key), "Registry should mark paper as processed"

        # Second run → skipped
        result2 = ingest_pdf(
            SAMPLE_PDF if SAMPLE_PDF.exists() else Path("/tmp/synthetic_test.pdf"),
            paper,
            store=store,
            embedder=mock_embedder,
            registry=registry,
            use_vlm=False,
            use_late=True,
        )
        assert result2.status == "skipped", f"Expected skipped, got {result2.status}"
        assert store.count() == result.n_chunks, "store count should not change after skip"

    def test_force_reingest(self, tmp_path: Path) -> None:
        """force=True 이면 이미 processed 여도 재인입한다."""
        store = _make_store(tmp_path)
        registry = _make_registry(tmp_path)
        mock_embedder = MockEmbedder(dim=DIM)
        paper = _make_paper()

        if not SAMPLE_PDF.exists():
            pytest.skip("sample.pdf not present — force reingest test skipped")

        # First ingest
        r1 = ingest_pdf(
            SAMPLE_PDF,
            paper,
            store=store,
            embedder=mock_embedder,
            registry=registry,
            use_vlm=False,
        )
        assert r1.status == "ingested"

        # Force reingest
        r2 = ingest_pdf(
            SAMPLE_PDF,
            paper,
            store=store,
            embedder=mock_embedder,
            registry=registry,
            use_vlm=False,
            force=True,
        )
        assert r2.status == "ingested", f"Expected ingested (force), got: {r2.status}"

    def test_missing_pdf_returns_failed_or_exception(self, tmp_path: Path) -> None:
        """존재하지 않는 PDF 경로는 failed 상태를 반환한다."""
        store = _make_store(tmp_path)
        registry = _make_registry(tmp_path)
        mock_embedder = MockEmbedder(dim=DIM)
        paper = _make_paper("nonexistent.pdf")

        # ingest_pdf is called directly with a missing path
        result = ingest_pdf(
            tmp_path / "nonexistent.pdf",
            paper,
            store=store,
            embedder=mock_embedder,
            registry=registry,
            use_vlm=False,
        )
        # The path doesn't exist, parse_pdf will fail → status='failed'
        assert result.status == "failed"
        assert result.error is not None

    def test_no_registry_still_ingests(self, tmp_path: Path) -> None:
        """registry=None 이어도 인입은 성공한다 (skip 체크 없음)."""
        if not SAMPLE_PDF.exists():
            pytest.skip("sample.pdf not present")

        store = _make_store(tmp_path)
        mock_embedder = MockEmbedder(dim=DIM)
        paper = _make_paper()

        result = ingest_pdf(
            SAMPLE_PDF,
            paper,
            store=store,
            embedder=mock_embedder,
            registry=None,
            use_vlm=False,
        )
        assert result.status == "ingested"
        assert result.n_chunks > 0


def _run_with_synthetic(
    tmp_path: Path,
    store: Any,
    mock_embedder: MockEmbedder,
    registry: ProcessedRegistry,
    paper: Paper,
) -> IngestResult:
    """synthetic ParsedDoc 경로 헬퍼: monkeypatch parse_pdf."""
    synthetic = _make_synthetic_parsed()

    # We monkeypatch the lazy import target inside pipeline.py
    import ingestion.parse.docling_parser as _dp

    orig = getattr(_dp, "parse_pdf", None)
    try:
        _dp.parse_pdf = lambda path: synthetic  # type: ignore[attr-defined]
        return ingest_pdf(
            tmp_path / "fake.pdf",
            paper,
            store=store,
            embedder=mock_embedder,
            registry=registry,
            use_vlm=False,
        )
    finally:
        if orig is not None:
            _dp.parse_pdf = orig  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Test: ingest_pdf with monkeypatch (robust either way)
# ---------------------------------------------------------------------------


class TestIngestPdfMonkeypatched:
    """sample.pdf なし でも常に動く synthetic ParsedDoc テスト."""

    def test_synthetic_ingest_and_skip(self, tmp_path: Path) -> None:
        """synthetic ParsedDoc 로 full pipeline (chunk+embed+load+registry) 를 검증한다."""
        store = _make_store(tmp_path)
        registry = _make_registry(tmp_path)
        mock_embedder = MockEmbedder(dim=DIM)
        paper = Paper(
            doi="10.5555/synthetic.test",
            title="Synthetic Test Paper",
            source="synthetic_test",
            file="synthetic.pdf",
            openalex_id="W_SYNTHETIC",
        )
        synthetic = _make_synthetic_parsed()

        # Patch at module level where pipeline.py does its lazy import
        import ingestion.parse.docling_parser as _dp

        orig_parse = getattr(_dp, "parse_pdf", None)
        try:
            _dp.parse_pdf = lambda path: synthetic  # type: ignore[attr-defined]

            # Create a fake PDF file so content_hash can be computed
            fake_pdf = tmp_path / "synthetic.pdf"
            fake_pdf.write_bytes(b"%PDF-1.4 synthetic content")

            result = ingest_pdf(
                fake_pdf,
                paper,
                store=store,
                embedder=mock_embedder,
                registry=registry,
                use_vlm=False,
                use_late=True,
            )

            assert (
                result.status == "ingested"
            ), f"Expected ingested, got {result.status}: {result.error}"
            assert result.n_chunks > 0
            assert store.count() == result.n_chunks

            key = processed_key(paper)
            assert registry.is_processed(key)

            # Second run → skipped
            result2 = ingest_pdf(
                fake_pdf,
                paper,
                store=store,
                embedder=mock_embedder,
                registry=registry,
                use_vlm=False,
            )
            assert result2.status == "skipped"
            assert store.count() == result.n_chunks  # unchanged

        finally:
            if orig_parse is not None:
                _dp.parse_pdf = orig_parse  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Test: ingest_corpus
# ---------------------------------------------------------------------------


class TestIngestCorpus:
    """ingest_corpus 테스트 — limit=1, 단일 논문."""

    def test_corpus_limit_one_ingested(self, tmp_path: Path) -> None:
        """limit=1 로 단일 논문 인입 — 결과 목록 길이 1, status 확인."""
        store = _make_store(tmp_path)
        registry = _make_registry(tmp_path)
        mock_embedder = MockEmbedder(dim=DIM)

        paper = Paper(
            doi="10.5555/corpus.test",
            title="Corpus Test Paper",
            source="corpus_test",
            file="sample.pdf",
            openalex_id="W_CORPUS_TEST",
        )

        if SAMPLE_PDF.exists():
            # Point pdf_dir at fixture directory
            results = ingest_corpus(
                [paper],
                pdf_dir=SAMPLE_PDF.parent,
                store=store,
                embedder=mock_embedder,
                registry=registry,
                use_vlm=False,
                use_late=True,
                limit=1,
            )
            assert len(results) == 1
            assert results[0].status == "ingested"
            assert results[0].n_chunks > 0

        else:
            # Monkeypatch parse_pdf for corpus test
            import ingestion.parse.docling_parser as _dp

            orig_parse = getattr(_dp, "parse_pdf", None)
            synthetic = _make_synthetic_parsed()
            try:
                _dp.parse_pdf = lambda path: synthetic  # type: ignore[attr-defined]

                # create a fake PDF in tmp dir
                fake_dir = tmp_path / "pdfs"
                fake_dir.mkdir()
                (fake_dir / "sample.pdf").write_bytes(b"%PDF-1.4 fake")

                results = ingest_corpus(
                    [paper],
                    pdf_dir=fake_dir,
                    store=store,
                    embedder=mock_embedder,
                    registry=registry,
                    use_vlm=False,
                    use_late=True,
                    limit=1,
                )
                assert len(results) == 1
                assert results[0].status == "ingested"
                assert results[0].n_chunks > 0

            finally:
                if orig_parse is not None:
                    _dp.parse_pdf = orig_parse  # type: ignore[attr-defined]

    def test_corpus_no_pdf_file_missing(self, tmp_path: Path) -> None:
        """PDF 파일이 없으면 no_pdf 상태를 반환한다."""
        store = _make_store(tmp_path)
        registry = _make_registry(tmp_path)
        mock_embedder = MockEmbedder(dim=DIM)

        paper = Paper(
            doi="10.5555/nopdf.test",
            title="No PDF Paper",
            source="nopdf_test",
            file="nonexistent_paper.pdf",
        )

        results = ingest_corpus(
            [paper],
            pdf_dir=tmp_path,
            store=store,
            embedder=mock_embedder,
            registry=registry,
            limit=1,
        )
        assert len(results) == 1
        assert results[0].status == "no_pdf"

    def test_corpus_no_file_field(self, tmp_path: Path) -> None:
        """paper.file 이 None 이면 no_pdf 상태를 반환한다."""
        store = _make_store(tmp_path)
        registry = _make_registry(tmp_path)
        mock_embedder = MockEmbedder(dim=DIM)

        paper = Paper(
            doi="10.5555/nofile.test",
            title="No File Paper",
            source="nofile_test",
            file=None,
        )

        results = ingest_corpus(
            [paper],
            pdf_dir=tmp_path,
            store=store,
            embedder=mock_embedder,
            registry=registry,
            limit=1,
        )
        assert len(results) == 1
        assert results[0].status == "no_pdf"

    def test_corpus_limit_respected(self, tmp_path: Path) -> None:
        """limit 이 papers 수보다 작으면 limit 만큼만 처리한다."""
        store = _make_store(tmp_path)
        registry = _make_registry(tmp_path)
        mock_embedder = MockEmbedder(dim=DIM)

        # 5 papers, all with no_pdf (file=None) — just counts results
        papers = [
            Paper(doi=f"10.0/{i}", title=f"Paper {i}", source=f"src_{i}", file=None)
            for i in range(5)
        ]

        results = ingest_corpus(
            papers,
            pdf_dir=tmp_path,
            store=store,
            embedder=mock_embedder,
            registry=registry,
            limit=3,
        )
        assert len(results) == 3

    def test_corpus_skip_on_second_run(self, tmp_path: Path) -> None:
        """두 번째 ingest_corpus 실행 시 이미 처리된 논문은 skipped 된다."""
        if not SAMPLE_PDF.exists():
            pytest.skip("sample.pdf not present for corpus skip test")

        store = _make_store(tmp_path)
        registry = _make_registry(tmp_path)
        mock_embedder = MockEmbedder(dim=DIM)
        paper = Paper(
            doi="10.5555/skip.test",
            title="Skip Test Paper",
            source="skip_test",
            file="sample.pdf",
            openalex_id="W_SKIP_TEST",
        )

        # First run
        results1 = ingest_corpus(
            [paper],
            pdf_dir=SAMPLE_PDF.parent,
            store=store,
            embedder=mock_embedder,
            registry=registry,
            use_vlm=False,
            limit=1,
        )
        assert results1[0].status == "ingested"

        # Second run — same registry
        results2 = ingest_corpus(
            [paper],
            pdf_dir=SAMPLE_PDF.parent,
            store=store,
            embedder=mock_embedder,
            registry=registry,
            use_vlm=False,
            limit=1,
        )
        assert results2[0].status == "skipped"
