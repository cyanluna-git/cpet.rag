"""tests/test_generate.py — Generator 유닛 테스트 (mock _generate_call).

모든 테스트는 실제 Bedrock API 없이 로컬에서 실행 가능하다.
`_generate_call` 을 mock 해 결정론적 답변을 주입하고 동작을 검증한다.
"""

from __future__ import annotations

from unittest.mock import patch

from core.models import Chunk, RetrievedChunk
from serving.generation import GenerationResult, Generator

# ---------------------------------------------------------------------------
# 헬퍼
# ---------------------------------------------------------------------------


def _make_chunk(chunk_id: str, doi: str = "10.0000/test", page: int = 1) -> Chunk:
    """테스트용 Chunk 생성."""
    return Chunk(
        id=chunk_id,
        doi=doi,
        section="Methods",
        text=f"This is the text content of chunk {chunk_id}. "
        f"It contains scientific information about exercise physiology.",
        ctx_text=f"[ctx] chunk {chunk_id}",
        page=page,
        chunk_index=int(chunk_id.lstrip("c") or 0),
        source="test_src",
    )


def _make_rc(chunk_id: str, doi: str = "10.0000/test", page: int = 1) -> RetrievedChunk:
    """RetrievedChunk 생성 (rerank_score=None)."""
    return RetrievedChunk(chunk=_make_chunk(chunk_id, doi=doi, page=page), score=0.9)


# ---------------------------------------------------------------------------
# build_prompt 테스트
# ---------------------------------------------------------------------------


def test_build_prompt_system_contains_strict_citation() -> None:
    """system 프롬프트에 Strict Citation 지시가 포함된다."""
    generator = Generator()
    chunks = [_make_rc("c1"), _make_rc("c2")]

    system, user = generator.build_prompt("What is VO2max?", chunks)

    # Strict Citation 핵심 지시 확인
    assert "ONLY the provided" in system or "ONLY" in system
    assert "citation" in system.lower()
    assert "[id]" in system or "brackets" in system.lower()
    assert "cannot answer from the provided sources" in system.lower()


def test_build_prompt_user_contains_numbered_context() -> None:
    """user 메시지에 번호+id 컨텍스트 블록이 포함된다."""
    generator = Generator()
    chunks = [_make_rc("c1"), _make_rc("c2")]

    system, user = generator.build_prompt("What is VO2max?", chunks)

    # [1], [2] 번호 포함
    assert "[1]" in user
    assert "[2]" in user
    # id= 참조 포함
    assert "id=c1" in user
    assert "id=c2" in user
    # 청크 원문 포함
    assert "chunk c1" in user
    assert "chunk c2" in user
    # 질의 포함
    assert "VO2max" in user


def test_build_prompt_context_includes_doi_page_section() -> None:
    """컨텍스트 헤더에 doi, page, section 이 포함된다."""
    generator = Generator()
    rc = _make_rc("c1", doi="10.1234/exercise", page=5)
    system, user = generator.build_prompt("query", [rc])

    assert "10.1234/exercise" in user
    assert "p.5" in user
    assert "§Methods" in user


# ---------------------------------------------------------------------------
# generate 기본 동작 테스트
# ---------------------------------------------------------------------------


def test_generate_returns_answer_en_and_citations() -> None:
    """generate 는 answer_en 을 반환하고 citations 가 chunk 에 매핑된다."""
    generator = Generator()
    chunks = [_make_rc("c1"), _make_rc("c2")]

    mock_answer = (
        "Endurance training increases mitochondrial density [c1]. "
        "Lactate threshold rises with aerobic conditioning [c2]."
    )

    with patch.object(generator, "_generate_call", return_value=mock_answer):
        result = generator.generate("What happens with endurance training?", chunks)

    assert isinstance(result, GenerationResult)
    assert result.answer_en == mock_answer
    assert result.refused is False

    # citations: c1, c2 모두 매핑됨
    citation_ids = {c.chunk_id for c in result.citations}
    assert "c1" in citation_ids
    assert "c2" in citation_ids


def test_generate_used_chunk_ids_matches_citations() -> None:
    """used_chunk_ids 는 citations 의 chunk_id 목록과 일치한다."""
    generator = Generator()
    chunks = [_make_rc("c1"), _make_rc("c2")]
    mock_answer = "Training effect [c1]. Recovery [c2]."

    with patch.object(generator, "_generate_call", return_value=mock_answer):
        result = generator.generate("query", chunks)

    assert set(result.used_chunk_ids) == {"c1", "c2"}
    assert result.used_chunk_ids == [c.chunk_id for c in result.citations]


def test_generate_citation_has_correct_doi_and_page() -> None:
    """citations 의 doi, page 가 원본 chunk 와 일치한다."""
    generator = Generator()
    rc = _make_rc("c1", doi="10.9999/sport", page=7)
    chunks = [rc]
    mock_answer = "Some finding [c1]."

    with patch.object(generator, "_generate_call", return_value=mock_answer):
        result = generator.generate("query", chunks)

    assert len(result.citations) == 1
    cit = result.citations[0]
    assert cit.chunk_id == "c1"
    assert cit.doi == "10.9999/sport"
    assert cit.page == 7


# ---------------------------------------------------------------------------
# 거부(refusal) 테스트
# ---------------------------------------------------------------------------


def test_generate_empty_chunks_returns_refused() -> None:
    """빈 chunks → refused=True, citations=[], _generate_call 호출 없음."""
    generator = Generator()

    with patch.object(generator, "_generate_call") as mock_call:
        result = generator.generate("query", [])

    assert result.refused is True
    assert result.citations == []
    assert result.used_chunk_ids == []
    mock_call.assert_not_called()


def test_generate_below_min_chunks_returns_refused() -> None:
    """chunks < min_chunks → refused=True, _generate_call 호출 없음."""
    generator = Generator()
    chunks = [_make_rc("c1")]  # 1개

    with patch.object(generator, "_generate_call") as mock_call:
        result = generator.generate("query", chunks, min_chunks=3)

    assert result.refused is True
    assert result.citations == []
    mock_call.assert_not_called()


def test_generate_refusal_answer_text() -> None:
    """거부 응답의 answer_en 은 표준 거부 문자열이다."""
    generator = Generator()

    result = generator.generate("query", [])

    assert "cannot answer" in result.answer_en.lower()
    assert result.answer_en == "I cannot answer from the provided sources."


def test_generate_exactly_min_chunks_succeeds() -> None:
    """chunks == min_chunks 이면 거부하지 않고 생성한다."""
    generator = Generator()
    chunks = [_make_rc("c1"), _make_rc("c2")]
    mock_answer = "Result [c1]."

    with patch.object(generator, "_generate_call", return_value=mock_answer):
        result = generator.generate("query", chunks, min_chunks=2)

    assert result.refused is False
    assert result.answer_en == mock_answer


# ---------------------------------------------------------------------------
# 존재하지 않는 태그 처리 테스트
# ---------------------------------------------------------------------------


def test_generate_unknown_citation_tag_excluded() -> None:
    """존재하지 않는 [cX] 태그는 citations 에서 제외된다."""
    generator = Generator()
    chunks = [_make_rc("c1"), _make_rc("c2")]
    # [c99] 는 chunks 에 없는 id
    mock_answer = "Some fact [c1]. Another fact [c99]."

    with patch.object(generator, "_generate_call", return_value=mock_answer):
        result = generator.generate("query", chunks)

    citation_ids = {c.chunk_id for c in result.citations}
    assert "c1" in citation_ids
    assert "c99" not in citation_ids


def test_generate_no_citations_in_answer() -> None:
    """답변에 인용 태그가 없으면 citations=[], used_chunk_ids=[] 이다."""
    generator = Generator()
    chunks = [_make_rc("c1"), _make_rc("c2")]
    mock_answer = "This answer has no citation tags at all."

    with patch.object(generator, "_generate_call", return_value=mock_answer):
        result = generator.generate("query", chunks)

    assert result.citations == []
    assert result.used_chunk_ids == []
    assert result.refused is False  # 거부 아님, 인용만 없는 것


# ---------------------------------------------------------------------------
# 중복 태그 처리 테스트
# ---------------------------------------------------------------------------


def test_generate_duplicate_citation_tags_deduped() -> None:
    """같은 [id] 가 여러 번 나와도 citation 은 한 번만 포함된다."""
    generator = Generator()
    chunks = [_make_rc("c1")]
    mock_answer = "First claim [c1]. Second claim [c1]. Third claim [c1]."

    with patch.object(generator, "_generate_call", return_value=mock_answer):
        result = generator.generate("query", chunks)

    assert len(result.citations) == 1
    assert result.citations[0].chunk_id == "c1"


# ---------------------------------------------------------------------------
# _generate_call 호출 확인 테스트
# ---------------------------------------------------------------------------


def test_generate_calls_generate_call_with_system_and_user() -> None:
    """generate 는 _generate_call(system, user) 를 정확히 1회 호출한다."""
    generator = Generator()
    chunks = [_make_rc("c1")]
    mock_answer = "Answer [c1]."

    with patch.object(generator, "_generate_call", return_value=mock_answer) as mock_call:
        generator.generate("test query", chunks)

    mock_call.assert_called_once()
    call_args = mock_call.call_args
    system_arg: str = call_args[0][0]
    user_arg: str = call_args[0][1]

    # system 에 strict citation 지시 포함
    assert "citation" in system_arg.lower()
    # user 에 컨텍스트와 질의 포함
    assert "c1" in user_arg
    assert "test query" in user_arg


# ---------------------------------------------------------------------------
# GenerationResult 필드 검증
# ---------------------------------------------------------------------------


def test_generation_result_fields_exist() -> None:
    """GenerationResult 는 answer_en, citations, refused, used_chunk_ids 필드를 가진다."""
    result = GenerationResult(answer_en="test", citations=[], refused=False, used_chunk_ids=[])
    assert hasattr(result, "answer_en")
    assert hasattr(result, "citations")
    assert hasattr(result, "refused")
    assert hasattr(result, "used_chunk_ids")


def test_generation_result_defaults() -> None:
    """GenerationResult 는 citations, refused, used_chunk_ids 기본값이 있다."""
    result = GenerationResult(answer_en="hello")
    assert result.citations == []
    assert result.refused is False
    assert result.used_chunk_ids == []


# ---------------------------------------------------------------------------
# doi::index 형태 chunk id (실제 프로덕션 id 패턴) 처리 테스트
# ---------------------------------------------------------------------------


def test_generate_doi_colon_chunk_id_parsed_correctly() -> None:
    """doi::index 형태의 실제 chunk id 도 정확히 파싱된다."""
    doi_chunk_id = "10.1234/exercise.2023::5"
    rc = RetrievedChunk(
        chunk=Chunk(
            id=doi_chunk_id,
            doi="10.1234/exercise.2023",
            section="Results",
            text="VO2max improved significantly after 8 weeks of training.",
            ctx_text="[ctx] vo2max",
            page=3,
            chunk_index=5,
            source="test",
        ),
        score=0.95,
    )
    generator = Generator()
    mock_answer = f"VO2max improved [{doi_chunk_id}]."

    with patch.object(generator, "_generate_call", return_value=mock_answer):
        result = generator.generate("VO2max training effect?", [rc])

    assert len(result.citations) == 1
    assert result.citations[0].chunk_id == doi_chunk_id
    assert result.citations[0].doi == "10.1234/exercise.2023"
    assert result.citations[0].page == 3
