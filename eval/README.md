# eval — RAGAS 평가 프레임워크

Faithfulness · Context Precision/Recall · Retrieval hit@k 지표로 RAG 파이프라인을 정량 평가한다. 파이프라인 변경 시 회귀 측정을 위해 이 디렉터리를 사용한다.

---

## 디렉터리 구조

```
eval/
├── README.md            ← 이 파일
├── qa_set/
│   ├── schema.md        ← 평가 레코드 포맷 및 RAGAS 매핑 문서
│   ├── qa_set.jsonl     ← 스타터 QA 세트 (8–12개, source: "starter-auto")
│   └── loader.py        ← load_qa_set() + to_ragas_dataset() 유틸리티
└── ragas/               ← RAGAS 실행 스크립트 (카드 #3129에서 구현)
```

---

## 평가 레코드 포맷

각 레코드는 다음 필드를 포함한다 (상세 설명: `qa_set/schema.md`):

| 필드 | 설명 |
|------|------|
| `id` | 고유 식별자 (예: `q001`) |
| `question_ko` | 한국어 연구 질문 — RAG 실제 입력 언어 |
| `question_en` | 영어 번역 |
| `answer_gold` | 전문가 검증 정답 (ground truth) |
| `relevant_dois` | 정답에 필요한 DOI 목록 — `data/corpus_index.csv`에 존재해야 함 |
| `difficulty` | `easy` / `med` / `hard` |
| `tags` | 주제 태그 |
| `source` | `starter-auto` 또는 `expert` |

---

## 로더 사용법

```python
from eval.qa_set.loader import load_qa_set, to_ragas_dataset

# 1. QA 세트 로드 및 검증
items = load_qa_set()          # 기본: eval/qa_set/qa_set.jsonl
# items = load_qa_set("path/to/custom.jsonl")

# 2. RAGAS 입력 포맷으로 변환 (contexts/answer는 평가 실행 시 채워짐)
ragas_rows = to_ragas_dataset(items)
# ragas_rows[0] → {"question": "...", "ground_truth": "...", "contexts": [], "answer": None}
```

---

## 평가 실행 (카드 #3129)

> **미구현** — 이 섹션은 #3129 완료 후 업데이트된다.

```bash
# 예상 명령 (구현 후 확인 필요)
uv run python eval/ragas/run_eval.py \
  --qa-set eval/qa_set/qa_set.jsonl \
  --output eval/results/$(date +%Y%m%d).json
```

실행 시 파이프라인:
1. `load_qa_set()` → 레코드 로드
2. 각 `question_ko`를 RAG retriever에 전달 → `contexts` 채우기
3. LLM 호출 → `answer` 채우기
4. `ragas.evaluate()` → Faithfulness, Context Precision, Context Recall, Answer Relevancy 계산

> **RAGAS 버전 주의**: RAGAS ≥ 0.2는 컬럼명이 `user_input` / `reference` / `retrieved_contexts` / `response`로 변경됐다. `to_ragas_dataset()`는 구 형식 기준이므로 #3129에서 버전에 맞게 조정한다.

---

## 스타터 세트 현황 (source: "starter-auto")

현재 `qa_set.jsonl`에는 12개의 스타터 QA 레코드가 있다.

| 수 | 유형 | 기반 |
|----|------|------|
| 8 | abstract-grounded | OpenAlex에서 초록 확인 후 작성 |
| 4 | title-based | 논문 제목·저널·맥락 기반 작성 |

앵커 논문 (corpus_index.csv의 실제 DOI):
- `10.3390/ijerph18094963` — Regulation of Energy Substrate Metabolism in Endurance Exercise (Alghannam, 2021)
- `10.1007/s00421-022-04935-1` — Century of exercise physiology: glycogen metabolism (Katz, 2022)
- `10.3390/ijerph20010453` — FATmax vs. Aerobic Threshold (Ferri Marini, 2022)
- `10.3390/jcm13020535` — Blood lactate / Graded Exercise Test in cyclists (Zając, 2024)
- `10.1007/s00421-025-06022-7` — Maximal lactate accumulation rate cLamax (Quittmann, 2025)
- `10.3389/fphys.2020.585137` — Fat oxidation during endurance exercise in women (Isacco, 2020)
- `10.1016/j.heliyon.2022.e11091` — AMPK and skeletal muscle glucose metabolism (Esquejo, 2022)
- `10.1073/pnas.2204750120` — Exercise/mitochondrial dynamics/aging (Campos, 2023)
- `10.1161/circulationaha.116.018093` — Athlete's heart / cardiac remodeling (Wasfy, 2016)
- `10.1016/j.jcmg.2017.09.016` — Cardiac MRI in athletes (Tahir, 2018)

---

## 전문가 QA 세트 확장 프로세스 (20~30개 목표)

스타터 세트를 20~30개의 전문가 검증 세트로 확장하려면 아래 프로세스를 따른다.

### 단계별 프로세스

1. **교수(도메인 전문가) 검토**
   - 현재 `qa_set.jsonl`의 각 `answer_gold`가 해당 논문 내용과 실제로 일치하는지 검증
   - 오류가 있으면 수정하고 `source: "expert"`로 변경
   - 검증 불가한 레코드는 제거 또는 `difficulty: hard`로 재표시

2. **전문 지식 기반 신규 QA 추가**
   - 각 QA는 반드시 하나 이상의 `relevant_dois`를 가져야 하며, 해당 DOI는 `data/corpus_index.csv`에 존재해야 함
   - `answer_gold`는 논문 원문(full-text PDF 또는 초록)에서 직접 인용/요약해야 함 — 추측 금지
   - **전문가 QA에는 전문지 PDF가 필요하다** (초록만으로 충분하지 않은 경우 많음): S3 버킷에서 관련 PDF를 가져올 것

3. **난이도 분포 권장** (20~30개 기준)
   - easy: ~8개 (기본 개념·단일 논문 사실 질문)
   - med: ~14개 (비교·수치·다개념 종합)
   - hard: ~6개 (다논문 증거 종합·임상 판단)

4. **태그 일관성 유지**
   - 태그는 `schema.md`에 정의된 태그 풀에서 선택하거나 새로운 태그 추가 시 schema.md도 갱신

5. **검증 스크립트 실행**
   ```python
   # 모든 relevant_dois가 corpus에 존재하는지 확인
   from core.metadata.loader import load_corpus_index, normalize_doi
   import json
   from pathlib import Path

   corpus_dois = {normalize_doi(p.doi) for p in load_corpus_index() if p.doi}
   qa_path = Path("eval/qa_set/qa_set.jsonl")
   with qa_path.open(encoding="utf-8") as fh:
       for line in fh:
           if line.strip():
               rec = json.loads(line)
               for doi in rec["relevant_dois"]:
                   assert normalize_doi(doi) in corpus_dois, f"DOI not in corpus: {doi}"
   print("All DOIs verified!")
   ```

6. **source 필드 업데이트**
   - 교수가 검토한 레코드: `"source": "expert"`
   - 스타터 세트에서 미검증 상태로 유지: `"source": "starter-auto"`

### 우선 추가 권장 주제 (hargreaves2020/dausin2026 코퍼스 기반)

- CPET 프로토콜 표준화 및 VO2max 측정 방법
- 운동 유발 심실 부정맥 판별 기준
- 고강도 인터벌 훈련(HIIT) vs 지속적 유산소 훈련의 미토콘드리아 적응 비교
- 운동선수 심장의 ECG 변화 해석
- 젖산 역치(LT1/LT2) 기반 훈련 처방 방법론
