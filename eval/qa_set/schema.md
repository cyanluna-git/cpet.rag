# RAGAS 평가 레코드 포맷 (schema.md)

## 레코드 구조

각 평가 레코드는 다음 필드를 포함하는 JSON 객체이다.

```json
{
  "id": "q001",
  "question_ko": "지구성 운동 중 골격근의 주요 에너지 기질은 무엇인가?",
  "question_en": "What are the primary energy substrates in skeletal muscle during endurance exercise?",
  "answer_gold": "지구성 운동 중 골격근은 주로 탄수화물(근글리코겐, 혈중 포도당)과 지방(근육 내 중성지방, 혈중 유리지방산)을 에너지 기질로 사용한다. 운동 강도가 높아질수록 탄수화물 의존도가 증가하며, 낮은 강도에서는 지방 산화 비율이 높다.",
  "relevant_dois": ["10.3390/ijerph18094963"],
  "difficulty": "easy",
  "tags": ["energy metabolism", "skeletal muscle", "endurance exercise", "carbohydrate", "fat oxidation"],
  "source": "starter-auto"
}
```

## 필드 정의

| 필드 | 타입 | 필수 | 설명 |
|------|------|------|------|
| `id` | string | 필수 | 레코드 고유 식별자 (예: `q001`). 정렬 가능한 순번 접두사 권장. |
| `question_ko` | string | 필수 | 한국어 연구 질문. RAG 시스템이 실제로 수신하는 입력 언어. |
| `question_en` | string | 필수 | 영어 번역. 벡터 검색 쿼리 또는 LLM 프롬프트에 활용 가능. |
| `answer_gold` | string | 필수 | 전문가 검증 정답 (ground truth). 해당 논문 내용에 근거해야 함. |
| `relevant_dois` | list[string] | 필수 | 정답 도출에 필요한 논문 DOI 목록. `data/corpus_index.csv`에 존재하는 DOI만 허용. retrieval 평가의 기준이 됨. |
| `difficulty` | string | 필수 | `easy` / `med` / `hard` 중 하나. |
| `tags` | list[string] | 필수 | 주제 태그 (검색·필터링용). |
| `source` | string | 필수 | 출처 유형: `starter-auto` (본 파일의 초기 세트) 또는 `expert` (교수 검토 완료). |

### difficulty 기준

- **easy**: 단일 논문에서 직접 확인 가능한 사실 질문.
- **med**: 두 개 이상의 개념을 종합하거나 수치/비교가 필요한 질문.
- **hard**: 여러 논문 간 증거를 종합하거나 임상적 판단이 필요한 질문.

---

## RAGAS 매핑

RAGAS는 4개의 컬럼으로 평가 데이터셋을 구성한다.

| RAGAS 컬럼 | 이 스키마의 필드 | 메모 |
|------------|----------------|------|
| `question` | `question_ko` | RAG 시스템의 실제 입력. `question_en`으로 바꾸는 경우 #3129에서 결정. |
| `ground_truth` | `answer_gold` | 평가 시점에 고정값. |
| `contexts` | *(평가 실행 시 채워짐)* | RAG retriever가 반환한 청크 목록. 이 파일에는 `[]`로 저장. |
| `answer` | *(평가 실행 시 채워짐)* | LLM이 생성한 답변. 이 파일에는 `null`로 저장. |

> **RAGAS ≥ 0.2 변경 사항**: RAGAS 0.2+는 컬럼명을 `user_input` / `reference` / `retrieved_contexts` / `response`로 변경했다. `loader.py`의 `to_ragas_dataset()`는 이전 형식(`question`, `ground_truth`)을 기본으로 반환하며, #3129 구현 시 버전에 맞게 조정한다.

### 실행 시 데이터 흐름

```
qa_set.jsonl
    ↓ load_qa_set()
list[EvalItem]
    ↓ to_ragas_dataset()  ← question, ground_truth만 채워진 상태
    ↓ (평가 실행: retriever 호출 → contexts 채우기)
    ↓ (평가 실행: LLM 호출 → answer 채우기)
    ↓ ragas.evaluate()
RAGAS 메트릭 (Faithfulness, Context Precision, Context Recall, Answer Relevancy)
```

---

## 확장 프로세스 (20~30개 전문가 세트)

starter-auto 레코드를 교수 전문가 검토를 거쳐 `source: "expert"` 레코드로 승격하거나 신규 추가한다. 자세한 내용은 `eval/README.md` 참조.
