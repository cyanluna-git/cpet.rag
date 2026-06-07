"""ingestion.load.registry — 증분 인입 추적 레지스트리.

어떤 논문이 이미 적재됐는지를 JSONL 파일로 추적한다.
동일 논문이라도 임베딩 모델/버전이 변경되면 재적재를 강제한다.

파일 위치 기본값:
    {lancedb_uri_parent}/processed.jsonl
    → settings.lancedb_uri = "./data/lancedb"  이면
      "./data/processed.jsonl" 에 저장.

S3 URI(s3://...) 인 경우에는 lancedb_uri 부모를 사용할 수 없으므로
명시적으로 path 를 전달하거나 로컬 "./data/processed.jsonl" 폴백을 사용한다.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from core.config.settings import Settings
from core.log import get_logger
from core.metadata.loader import normalize_doi
from core.models.paper import Paper

logger = get_logger(__name__)


def processed_key(paper: Paper) -> str:
    """논문의 안정적 레지스트리 키를 반환한다.

    우선순위:
    1. paper.openalex_id
    2. normalize_doi(paper.doi)
    3. paper.file
    4. paper.source (최후 폴백)
    """
    if paper.openalex_id:
        return paper.openalex_id
    doi = normalize_doi(paper.doi)
    if doi:
        return doi
    if paper.file:
        return paper.file
    return paper.source


def _default_registry_path() -> Path:
    """settings 에서 기본 processed.jsonl 경로를 계산한다."""
    settings = Settings()
    uri: str = settings.lancedb_uri or "./data/lancedb"
    if uri.startswith("s3://"):
        # S3 URI 는 로컬 폴백 사용
        logger.debug("lancedb_uri is S3 — falling back to local ./data/processed.jsonl")
        return Path("./data/processed.jsonl")
    return Path(uri).parent / "processed.jsonl"


class ProcessedRegistry:
    """적재 완료 논문을 JSONL 파일로 추적한다.

    각 행은 다음 구조를 가진다::

        {
            "key": "<openalex_id | doi | file | source>",
            "content_hash": "<sha256 or None>",
            "embed_model": "<model_id>",
            "embed_version": "<model@dim string or None>",
            "n_chunks": 42
        }

    동일 key 가 여러 번 기록될 경우 마지막 행이 유효하다 (last-write-wins).
    ``save()`` 는 중복 제거 후 전체를 재기록한다.

    Parameters
    ----------
    path:
        JSONL 파일 경로. None 이면 기본값(_default_registry_path)을 사용.
    """

    def __init__(self, path: str | Path | None = None) -> None:
        self._path: Path = Path(path) if path is not None else _default_registry_path()
        # key → record (last-write-wins)
        self._records: dict[str, dict[str, Any]] = {}
        if self._path.exists():
            self.load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def load(self) -> None:
        """JSONL 파일을 읽어 내부 레코드를 갱신한다 (last-write-wins)."""
        if not self._path.exists():
            logger.debug("ProcessedRegistry.load: 파일 없음 — 빈 레지스트리")
            return

        with self._path.open("r", encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record: dict[str, Any] = json.loads(line)
                    key = record.get("key")
                    if key:
                        self._records[key] = record
                except json.JSONDecodeError as exc:
                    logger.warning("ProcessedRegistry.load: 줄 %d 파싱 실패 — %s", lineno, exc)
        logger.debug("ProcessedRegistry.load: %d 항목 로드", len(self._records))

    def save(self) -> None:
        """현재 레코드 전체를 JSONL 파일로 재기록한다 (중복 제거)."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("w", encoding="utf-8") as fh:
            for record in self._records.values():
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        logger.debug("ProcessedRegistry.save: %d 항목 저장 → %s", len(self._records), self._path)

    def _append(self, record: dict[str, Any]) -> None:
        """레코드를 파일에 한 줄 추가 (append-only 운영 모드)."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    # ------------------------------------------------------------------
    # API
    # ------------------------------------------------------------------

    def is_processed(
        self,
        key: str,
        *,
        content_hash: str | None = None,
        embed_version: str | None = None,
    ) -> bool:
        """key 가 이미 적재됐는지 확인한다.

        Parameters
        ----------
        key:
            ``processed_key(paper)`` 로 얻은 안정 키.
        content_hash:
            파일 해시. 제공되면 저장된 값과 비교 — 다르면 False(재적재).
        embed_version:
            임베딩 모델 버전 문자열. 제공되면 저장된 값과 비교 — 다르면 False(재적재).

        Returns
        -------
        bool
            True: 동일 조건으로 이미 적재됨.
            False: 미적재 또는 hash/version 불일치 → 재적재 필요.
        """
        record = self._records.get(key)
        if record is None:
            return False

        if embed_version is not None and record.get("embed_version") != embed_version:
            logger.debug(
                "is_processed: key=%s embed_version 불일치 (stored=%s, wanted=%s)",
                key,
                record.get("embed_version"),
                embed_version,
            )
            return False

        if content_hash is not None and record.get("content_hash") != content_hash:
            logger.debug("is_processed: key=%s content_hash 불일치 — 재적재 필요", key)
            return False

        return True

    def mark_processed(
        self,
        key: str,
        *,
        content_hash: str | None = None,
        embed_version: str | None = None,
        n_chunks: int = 0,
    ) -> None:
        """key 를 적재 완료로 기록한다.

        파일에 즉시 한 줄 추가(append)하고 메모리 레코드도 갱신한다.
        """
        record: dict[str, Any] = {
            "key": key,
            "content_hash": content_hash,
            "embed_version": embed_version,
            "n_chunks": n_chunks,
        }
        self._records[key] = record
        self._append(record)
        logger.debug("mark_processed: key=%s n_chunks=%d", key, n_chunks)

    def all_processed(self) -> set[str]:
        """현재 레지스트리에 기록된 모든 키를 반환한다."""
        return set(self._records.keys())
