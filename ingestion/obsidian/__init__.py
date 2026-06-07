"""ingestion.obsidian — Obsidian vault 생성 패키지.

코퍼스 Paper 목록으로부터 Obsidian 호환 마크다운 노트 vault 를 생성한다.
각 노트는 YAML frontmatter + 본문 + corpus 내 인용 wikilink 를 포함한다.

공개 API:
    note_filename  : Paper → 파일시스템 안전 노트 파일명 (.md)
    paper_to_note  : Paper → Obsidian 마크다운 문자열
    write_vault    : Paper 목록 → 디렉터리에 노트 파일 기록
"""

from ingestion.obsidian.note_builder import note_filename, paper_to_note
from ingestion.obsidian.vault_writer import write_vault

__all__ = ["note_filename", "paper_to_note", "write_vault"]
