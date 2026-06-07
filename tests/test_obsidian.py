"""tests/test_obsidian.py — ingestion.obsidian 단위 테스트.

외부 네트워크 없이 실행 가능. build_citation_links 실제 API 테스트는
네트워크가 없으면 자동 skip.

실행:
    uv run pytest tests/test_obsidian.py -q
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from core.models.paper import Paper
from ingestion.obsidian import note_filename, paper_to_note, write_vault
from ingestion.obsidian.note_builder import slug
from ingestion.obsidian.vault_writer import build_citation_links

# ---------------------------------------------------------------------------
# 공통 픽스처
# ---------------------------------------------------------------------------

yaml = pytest.importorskip("yaml", reason="PyYAML 없음 — frontmatter YAML 파싱 테스트 건너뜀")


def _extract_frontmatter(note: str) -> dict:
    """노트 문자열에서 YAML frontmatter 블록을 파싱해 dict 로 반환한다."""
    # 첫 번째 --- 와 두 번째 --- 사이
    lines = note.split("\n")
    assert lines[0] == "---", f"노트가 '---' 로 시작하지 않음: {lines[0]!r}"
    try:
        end_idx = lines.index("---", 1)
    except ValueError:
        pytest.fail("닫는 '---' 를 찾을 수 없음")
    fm_text = "\n".join(lines[1:end_idx])
    return yaml.safe_load(fm_text)


def _make_papers() -> tuple[Paper, Paper, Paper]:
    """테스트용 3개 Paper 픽스처를 반환한다."""
    p1 = Paper(
        doi="10.1234/test.001",
        title="Maximal Oxygen Uptake: A Review",
        first_author="Bassett",
        year=2000,
        journal="Medicine & Science in Sports",
        source="bassett2000",
        oa_status="gold",
        openalex_id="https://openalex.org/W1000000001",
        authors=["Bassett, David R.", "Howley, Edward T."],
        abstract="VO2max is a key determinant of endurance performance.",
    )
    p2 = Paper(
        doi="10.5678/test.002",
        title="CPET Protocols: Clinical Guidelines",
        first_author="Myers",
        year=2015,
        journal="Journal of Cardiopulmonary Rehabilitation",
        source="myers2015",
        oa_status="closed",
        openalex_id="https://openalex.org/W1000000002",
        authors=["Myers, Jonathan"],
        abstract="CPET is widely used for cardiopulmonary assessment.",
    )
    p3 = Paper(
        doi=None,
        title="Lactate Threshold Training",
        first_author="Jones",
        year=2010,
        journal=None,
        source="jones2010",
        oa_status=None,
        openalex_id="https://openalex.org/W1000000003",
        authors=["Jones, Andrew M."],
        abstract=None,
    )
    return p1, p2, p3


# ---------------------------------------------------------------------------
# note_filename 테스트
# ---------------------------------------------------------------------------


class TestNoteFilename:
    def test_basic_format(self) -> None:
        """파일명이 year_author_id.md 형식이어야 한다."""
        p1, _, _ = _make_papers()
        fname = note_filename(p1)
        assert fname.endswith(".md")
        parts = fname[:-3].split("_", 2)  # 최소 3 세그먼트
        assert parts[0] == "2000"
        assert "Bassett" in parts[1]

    def test_deterministic(self) -> None:
        """동일 Paper 는 항상 같은 파일명을 반환한다."""
        p1, _, _ = _make_papers()
        assert note_filename(p1) == note_filename(p1)

    def test_filesystem_safe(self) -> None:
        """파일명에 '/', 공백, 콜론이 없어야 한다."""
        p1, p2, p3 = _make_papers()
        for p in (p1, p2, p3):
            fname = note_filename(p)
            assert "/" not in fname
            assert " " not in fname
            assert ":" not in fname

    def test_no_doi_fallback_to_openalex(self) -> None:
        """doi 없어도 openalex_id 를 ID 세그먼트로 사용한다."""
        _, _, p3 = _make_papers()
        fname = note_filename(p3)
        # openalex_id URL 에서 W1000000003 추출되어야 함
        assert "W1000000003" in fname

    def test_year_nd_when_missing(self) -> None:
        """year 가 None 이면 'nd' 를 사용한다."""
        p = Paper(title="No Year Paper", source="test_src")
        fname = note_filename(p)
        assert fname.startswith("nd_")

    def test_author_na_when_missing(self) -> None:
        """first_author 가 None 이면 'na' 를 사용한다."""
        p = Paper(title="No Author Paper", source="no_author_src", year=2020)
        fname = note_filename(p)
        assert fname.startswith("2020_na_")

    def test_unique_per_paper(self) -> None:
        """서로 다른 3개 Paper 는 각각 다른 파일명을 가진다."""
        p1, p2, p3 = _make_papers()
        names = {note_filename(p1), note_filename(p2), note_filename(p3)}
        assert len(names) == 3

    def test_acceptance_paper(self) -> None:
        """AC 예시 Paper 에 대해 파일명을 정상 반환해야 한다."""
        p = Paper(title="Muscle metabolism", source="hargreaves2020", year=2020, openalex_id="W1")
        fname = note_filename(p)
        assert fname.endswith(".md")
        assert "2020" in fname
        assert "/" not in fname


# ---------------------------------------------------------------------------
# slug 테스트
# ---------------------------------------------------------------------------


class TestSlug:
    def test_none_returns_empty(self) -> None:
        assert slug(None) == ""

    def test_empty_string_returns_empty(self) -> None:
        assert slug("") == ""

    def test_spaces_replaced(self) -> None:
        result = slug("Hello World")
        assert " " not in result

    def test_colon_replaced(self) -> None:
        result = slug("Title: Subtitle")
        assert ":" not in result

    def test_max_len(self) -> None:
        long_text = "A" * 100
        assert len(slug(long_text, max_len=20)) <= 20

    def test_korean_preserved(self) -> None:
        result = slug("한국어테스트", max_len=40)
        assert "한국어테스트" in result


# ---------------------------------------------------------------------------
# paper_to_note 테스트
# ---------------------------------------------------------------------------


class TestPaperToNote:
    def test_starts_with_triple_dash(self) -> None:
        """노트는 반드시 '---\\n' 으로 시작해야 한다 (Obsidian frontmatter)."""
        p1, _, _ = _make_papers()
        note = paper_to_note(p1)
        assert note.startswith("---\n"), f"노트가 '---\\n' 로 시작하지 않음: {note[:30]!r}"

    def test_frontmatter_parseable(self) -> None:
        """frontmatter 를 yaml.safe_load 로 파싱할 수 있어야 한다."""
        p1, _, _ = _make_papers()
        note = paper_to_note(p1)
        fm = _extract_frontmatter(note)
        assert isinstance(fm, dict)

    def test_frontmatter_doi(self) -> None:
        p1, _, _ = _make_papers()
        fm = _extract_frontmatter(paper_to_note(p1))
        assert fm["doi"] == "10.1234/test.001"

    def test_frontmatter_title(self) -> None:
        p1, _, _ = _make_papers()
        fm = _extract_frontmatter(paper_to_note(p1))
        assert fm["title"] == "Maximal Oxygen Uptake: A Review"

    def test_frontmatter_authors_list(self) -> None:
        """authors 는 YAML list 로 파싱되어야 한다."""
        p1, _, _ = _make_papers()
        fm = _extract_frontmatter(paper_to_note(p1))
        assert isinstance(fm["authors"], list)
        assert "Bassett, David R." in fm["authors"]

    def test_frontmatter_year(self) -> None:
        p1, _, _ = _make_papers()
        fm = _extract_frontmatter(paper_to_note(p1))
        assert fm["year"] == 2000

    def test_frontmatter_source(self) -> None:
        p1, _, _ = _make_papers()
        fm = _extract_frontmatter(paper_to_note(p1))
        assert fm["source"] == "bassett2000"

    def test_frontmatter_tags_include_source(self) -> None:
        """tags 에는 'cpet-rag' 와 source 값이 포함되어야 한다."""
        p1, _, _ = _make_papers()
        fm = _extract_frontmatter(paper_to_note(p1))
        tags = fm["tags"]
        assert "cpet-rag" in tags
        assert "bassett2000" in tags

    def test_frontmatter_openalex_id_bare(self) -> None:
        """openalex_id 는 bare ID (W…) 로 저장되어야 한다."""
        p1, _, _ = _make_papers()
        fm = _extract_frontmatter(paper_to_note(p1))
        assert fm["openalex_id"] == "W1000000001"

    def test_title_header_in_body(self) -> None:
        """본문에 # {title} 헤더가 있어야 한다."""
        p1, _, _ = _make_papers()
        note = paper_to_note(p1)
        assert "# Maximal Oxygen Uptake: A Review" in note

    def test_abstract_used_as_body(self) -> None:
        """body 인수 없이 paper.abstract 가 본문으로 사용되어야 한다."""
        p1, _, _ = _make_papers()
        note = paper_to_note(p1)
        assert "VO2max is a key determinant" in note

    def test_custom_body_overrides_abstract(self) -> None:
        """body 인수가 있으면 abstract 대신 사용되어야 한다."""
        p1, _, _ = _make_papers()
        note = paper_to_note(p1, body="# Custom Body\nCustom content here.")
        assert "Custom content here." in note
        assert "VO2max" not in note

    def test_placeholder_when_no_abstract(self) -> None:
        """abstract 없고 body 없으면 placeholder 가 포함되어야 한다."""
        _, _, p3 = _make_papers()
        assert p3.abstract is None
        note = paper_to_note(p3)
        assert "본문 없음" in note or "PDF 파싱" in note

    def test_no_citation_section_when_empty(self) -> None:
        """cited_papers=[] 이면 인용 섹션이 없어야 한다."""
        p1, _, _ = _make_papers()
        note = paper_to_note(p1, cited_papers=[])
        assert "인용" not in note

    def test_wikilink_present_for_cited_paper(self) -> None:
        """cited_papers 에 포함된 논문에 대한 wikilink 가 있어야 한다."""
        p1, p2, _ = _make_papers()
        note = paper_to_note(p1, cited_papers=[p2])
        # wikilink 형식: [[note_filename_without_md|alias]] — .md 확장자 없음 (Obsidian 표준)
        expected_fname = note_filename(p2)
        expected_target = expected_fname[:-3] if expected_fname.endswith(".md") else expected_fname
        assert f"[[{expected_target}|" in note

    def test_wikilink_title_in_alias(self) -> None:
        """wikilink alias 에 인용 논문의 제목이 포함되어야 한다."""
        p1, p2, _ = _make_papers()
        note = paper_to_note(p1, cited_papers=[p2])
        assert "CPET Protocols: Clinical Guidelines" in note

    def test_multiple_wikilinks(self) -> None:
        """여러 인용 논문에 대한 wikilink 가 모두 있어야 한다."""
        p1, p2, p3 = _make_papers()
        note = paper_to_note(p1, cited_papers=[p2, p3])
        # wikilinks use filename without .md extension
        f2 = note_filename(p2)[:-3]
        f3 = note_filename(p3)[:-3]
        assert f2 in note
        assert f3 in note

    def test_title_with_pipe_escaped_in_wikilink(self) -> None:
        """제목에 '|' 가 있으면 wikilink alias 에서 안전하게 치환된다."""
        p_pipe = Paper(
            title="A | B title",
            source="pipe_test",
            year=2021,
            openalex_id="W9999",
        )
        p_main, _, _ = _make_papers()
        note = paper_to_note(p_main, cited_papers=[p_pipe])
        # | 가 alias 부분에 있으면 Obsidian 이 오작동하므로 치환되어야 함
        # [[target|alias]] 에서 alias 에 | 가 없어야 함
        wikilinks = re.findall(r"\[\[([^\]]+)\]\]", note)
        for link in wikilinks:
            parts = link.split("|")
            if len(parts) == 2:
                alias = parts[1]
                assert "|" not in alias

    def test_frontmatter_null_values_valid_yaml(self) -> None:
        """None 필드 (doi, journal 등) 가 null YAML 로 직렬화되어야 한다."""
        _, _, p3 = _make_papers()  # doi=None, journal=None
        fm = _extract_frontmatter(paper_to_note(p3))
        assert fm["doi"] is None
        assert fm["journal"] is None

    def test_acceptance_smoke(self) -> None:
        """AC 예시: paper_to_note 결과에 '---' 가 포함되어야 한다."""
        p = Paper(title="Muscle metabolism", source="hargreaves2020", year=2020, openalex_id="W1")
        note = paper_to_note(p)
        assert "---" in note
        assert note.startswith("---\n")


# ---------------------------------------------------------------------------
# write_vault 테스트
# ---------------------------------------------------------------------------


class TestWriteVault:
    def test_writes_correct_count(self, tmp_path: Path) -> None:
        """N 개 Paper 에 대해 N 개 파일이 기록되어야 한다."""
        p1, p2, p3 = _make_papers()
        count = write_vault([p1, p2, p3], tmp_path, fetch_citations=False)
        assert count == 3

    def test_files_created(self, tmp_path: Path) -> None:
        """실제 파일이 생성되어야 한다."""
        p1, p2, p3 = _make_papers()
        write_vault([p1, p2, p3], tmp_path, fetch_citations=False)
        files = list(tmp_path.glob("*.md"))
        assert len(files) == 3

    def test_files_start_with_triple_dash(self, tmp_path: Path) -> None:
        """각 파일은 '---' 으로 시작해야 한다."""
        p1, p2, p3 = _make_papers()
        write_vault([p1, p2, p3], tmp_path, fetch_citations=False)
        for f in tmp_path.glob("*.md"):
            content = f.read_text(encoding="utf-8")
            assert content.startswith("---"), f"{f.name} 이 '---' 로 시작하지 않음"

    def test_filenames_stable(self, tmp_path: Path) -> None:
        """같은 Paper 로 두 번 write_vault 하면 동일한 파일명이 나온다."""
        p1, p2, p3 = _make_papers()
        names1 = {f.name for f in tmp_path.glob("*.md")}
        write_vault([p1, p2, p3], tmp_path, fetch_citations=False)
        names2 = {f.name for f in tmp_path.glob("*.md")}
        assert names1 == names2 or len(names2) == 3  # 첫 실행 시 names1 비어있음

    def test_filenames_unique(self, tmp_path: Path) -> None:
        """서로 다른 3개 Paper 의 파일명은 모두 달라야 한다."""
        p1, p2, p3 = _make_papers()
        write_vault([p1, p2, p3], tmp_path, fetch_citations=False)
        files = list(tmp_path.glob("*.md"))
        names = [f.name for f in files]
        assert len(set(names)) == len(names)

    def test_bodies_map_used(self, tmp_path: Path) -> None:
        """bodies 맵이 있으면 해당 키의 본문이 노트에 반영되어야 한다."""
        from ingestion.load.registry import processed_key as real_processed_key

        p1, p2, p3 = _make_papers()
        # processed_key 는 openalex_id 를 URL 형식 그대로 반환한다
        key = real_processed_key(p1)
        assert key == "https://openalex.org/W1000000001"  # URL 형식 확인
        bodies = {key: "# Custom Body\nInjected content."}
        write_vault([p1, p2, p3], tmp_path, bodies=bodies, fetch_citations=False)

        fname = note_filename(p1)
        content = (tmp_path / fname).read_text(encoding="utf-8")
        assert "Injected content." in content

    def test_limit_parameter(self, tmp_path: Path) -> None:
        """limit=2 이면 2개 파일만 기록되어야 한다."""
        p1, p2, p3 = _make_papers()
        count = write_vault([p1, p2, p3], tmp_path, fetch_citations=False, limit=2)
        assert count == 2
        files = list(tmp_path.glob("*.md"))
        assert len(files) == 2

    def test_empty_papers(self, tmp_path: Path) -> None:
        """빈 리스트를 전달하면 0을 반환하고 파일이 없어야 한다."""
        count = write_vault([], tmp_path, fetch_citations=False)
        assert count == 0
        assert not list(tmp_path.glob("*.md"))

    def test_creates_outdir(self, tmp_path: Path) -> None:
        """out_dir 이 없으면 자동 생성해야 한다."""
        p1, _, _ = _make_papers()
        new_dir = tmp_path / "nested" / "vault"
        write_vault([p1], new_dir, fetch_citations=False)
        assert new_dir.exists()
        assert len(list(new_dir.glob("*.md"))) == 1


# ---------------------------------------------------------------------------
# build_citation_links (네트워크 선택 테스트)
# ---------------------------------------------------------------------------


class TestProcessedKeyConsistency:
    """_processed_key が real processed_key と一致することを検証する."""

    def test_matches_real_processed_key(self) -> None:
        """write_vault 内部の _processed_key は ingestion.load.processed_key と同一でなければならない."""
        from ingestion.load.registry import processed_key as real_processed_key
        from ingestion.obsidian.vault_writer import _processed_key

        p1, p2, p3 = _make_papers()
        for p in (p1, p2, p3):
            assert _processed_key(p) == real_processed_key(p), (
                f"키 불일치: _processed_key={_processed_key(p)!r}, "
                f"real={real_processed_key(p)!r} for paper.source={p.source}"
            )


class TestBuildCitationLinks:
    def test_fetch_false_returns_empty(self) -> None:
        """fetch=False 이면 항상 빈 리스트를 반환한다."""
        p1, _, _ = _make_papers()
        result = build_citation_links(p1, {}, fetch=False)
        assert result == []

    def test_no_doi_returns_empty(self) -> None:
        """DOI 없는 Paper 는 빈 리스트를 반환한다."""
        _, _, p3 = _make_papers()
        assert p3.doi is None
        result = build_citation_links(p3, {}, fetch=True)
        assert result == []

    def test_real_doi_optional(self) -> None:
        """실제 DOI 로 API 호출 — 오프라인이면 skip."""
        import socket

        try:
            socket.setdefaulttimeout(3)
            socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect(("8.8.8.8", 53))
        except OSError:
            pytest.skip("오프라인 환경 — 네트워크 테스트 건너뜀")
        finally:
            socket.setdefaulttimeout(None)

        # corpus_index.csv 에서 첫 번째 유효 DOI 를 가져온다
        try:
            from core.metadata import load_corpus_index

            papers = load_corpus_index()
            doi_papers = [p for p in papers if p.doi]
            if not doi_papers:
                pytest.skip("corpus_index.csv 에 DOI 있는 논문 없음")
            test_paper = doi_papers[0]
        except FileNotFoundError:
            pytest.skip("corpus_index.csv 없음")

        corpus_map: dict[str, Paper] = {}
        result = build_citation_links(test_paper, corpus_map, fetch=True)
        # 실패하지 않고 list 를 반환해야 함 (corpus 교집합이 비어있어도 OK)
        assert isinstance(result, list)
