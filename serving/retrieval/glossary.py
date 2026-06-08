"""serving.retrieval.glossary — 운동생리/CPET/심장 도메인 KO↔EN 용어 보호 사전.

보호 사전은 번역 API 호출 전에 전문 용어를 placeholder 로 치환하고
번역 후 복원하는 "용어 보호" 메커니즘에서 사용된다.

ko2en_glossary: 한국어 → 영어 매핑 (KO 질의 번역 시 사용)
en2ko_glossary: 영어 → 한국어 매핑 (EN 답변 역번역 시 사용)
"""

# 한국어 → 영어 (ko2en 보호용)
# 긴 용어가 먼저 오도록 정렬되어야 함 — 짧은 서브스트링 우선 치환 방지.
# 실제 사용 시에는 _sorted_ko 를 통해 내림차순 길이 정렬하여 사용한다.
KO2EN_GLOSSARY: dict[str, str] = {
    # CPET / 심폐 기능
    "최대산소섭취량": "VO2max",
    "최고산소섭취량": "peak VO2",
    "산소섭취량": "oxygen uptake",
    "무산소성 역치": "anaerobic threshold",
    "무산소 역치": "anaerobic threshold",
    "젖산 역치": "lactate threshold",
    "환기 역치": "ventilatory threshold",
    "운동부하검사": "CPET",
    "심폐운동부하검사": "CPET",
    "호흡교환율": "respiratory exchange ratio",
    "환기당량": "ventilatory equivalent",
    "산소맥": "oxygen pulse",
    "호흡보상점": "respiratory compensation point",
    # 심혈관
    "심박출량": "cardiac output",
    "일회박출량": "stroke volume",
    "심박수": "heart rate",
    "최대심박수": "maximum heart rate",
    "심박예비량": "heart rate reserve",
    "동정맥산소차": "arteriovenous oxygen difference",
    # 에너지 대사
    "미토콘드리아": "mitochondria",
    "미토콘드리아 생합성": "mitochondrial biogenesis",
    "산화적 인산화": "oxidative phosphorylation",
    "해당과정": "glycolysis",
    "크렙스 회로": "Krebs cycle",
    "시트르산 회로": "citric acid cycle",
    "ATP": "ATP",
    # 근골격
    "골격근": "skeletal muscle",
    "근섬유": "muscle fiber",
    "속근 섬유": "fast-twitch fiber",
    "지근 섬유": "slow-twitch fiber",
    "모세혈관 밀도": "capillary density",
    # 호흡
    "분시환기량": "minute ventilation",
    "폐활량": "vital capacity",
    "노력성 폐활량": "forced vital capacity",
    "일초량": "FEV1",
    "최대수의환기량": "maximum voluntary ventilation",
    # 약어 — 혼용 보호 (알파벳 그대로 유지)
    "VO2max": "VO2max",
    "VO2peak": "VO2peak",
    "VT": "VT",
    "AT": "AT",
    "RER": "RER",
    "VE": "VE",
    "HR": "HR",
    "CO": "CO",
    "SV": "SV",
    "CPET": "CPET",
    "FVC": "FVC",
    "FEV1": "FEV1",
    "MVV": "MVV",
}

# 영어 → 한국어 (en2ko 보호용)
EN2KO_GLOSSARY: dict[str, str] = {en: ko for ko, en in KO2EN_GLOSSARY.items() if en != ko}

# 같은 EN 값이 여러 KO 키에 매핑될 경우 마지막 항목이 남는다 (충분히 허용 가능).
# 추가로 알파벳 약어 자기 보호 (영문→영문 항등)
_ACRONYMS: list[str] = [
    "VO2max",
    "VO2peak",
    "VT",
    "AT",
    "RER",
    "VE",
    "HR",
    "CO",
    "SV",
    "CPET",
    "FVC",
    "FEV1",
    "MVV",
    "ATP",
]
for _a in _ACRONYMS:
    EN2KO_GLOSSARY.setdefault(_a, _a)
