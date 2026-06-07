# ingestion — 인입 파이프라인

이벤트 기반·증분. PDF → 파싱 → 청킹 → 임베딩 → 벡터스토어 적재.

- `parse/` Docling (+ Gemini VLM 폴백, 복잡 표/수식)
- `chunk/` Late Chunking (구조 경계 + 메타 프리펜드)
- `embed/` Jina-embeddings-v3 (GPU)
- `load/`  LanceDB@S3 upsert + FTS
- `pipeline.py` LlamaIndex IngestionPipeline (processed dedup)

실행: 개발=Colab GPU, 운영=AWS Batch(g5). 신규 PDF만 처리.
