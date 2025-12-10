# Corporate Relationship Graph Builder (GNN + LLM)

한국 상장기업의 DART 사업보고서를 수집하여 사업 내용·표·LLM 추출 결과를 결합해  
GNN 학습용 기업 관계 그래프(NetworkX → PyG)를 생성하는 파이프라인입니다.

주요 기능
- DART 사업보고서 다운로드 및 파싱 (ZIP/XML → BeautifulSoup)
- 표(계열사 등) 자동 추출 및 정규화
- 규칙 기반 문장 분리 · 키워드 필터링
- LLM(GPT 계열)으로 기업 간 관계 추출 (엄격한 JSON 출력 규칙)
- NetworkX 그래프 생성 및 PyTorch Geometric(Data) 변환/저장

빠른 시작
1. Python 3.10+ 권장
2. 의존성 설치:
   ```
   pip install -r requirements.txt
   ```
   (requirements.txt에 openai, requests, beautifulsoup4, networkx, torch, torch-geometric, pandas, tqdm, xmltodict 등 명시)

환경변수 (필수)
- OPENAI_API_KEY: OpenAI API 키 (LLM 호출용)
- DART_API_KEY: DART(전자공시) API 키
- (선택) SAVE_DIR: 결과 저장 폴더 (기본: `output`)

입력 파일
- `theme_stock.csv` : 종목 목록(프로젝트 루트 기준)을 사용해 분석 대상 기업 필터링

실행 예시
1) 환경변수 설정 (bash 예시)
```
export OPENAI_API_KEY="sk-..."
export DART_API_KEY="YOUR_DART_KEY"
export SAVE_DIR="output"
```
2) 스크립트 실행
```
python build_graph.py
```

출력
- `{SAVE_DIR}/relationship_graph_llm.pkl` : NetworkX 그래프(피클)
- `{SAVE_DIR}/relationship_graph_pyg_llm.pt` : PyG Data 객체(torch.save)

결과 예시
1. `relationship_graph_llm.pkl` (NetworkX 그래프)
   - 노드: 기업 이름, 속성(corp_code, stock_code 등)
   - 엣지: 관계 유형(relation), 증거 문장(evidence) 등
   - 예:
     ```python
     import pickle
     with open("output/relationship_graph_llm.pkl", "rb") as f:
         G = pickle.load(f)
     print(G.nodes(data=True))  # 노드 정보 출력
     print(G.edges(data=True))  # 엣지 정보 출력
     ```

2. `relationship_graph_pyg_llm.pt` (PyG Data 객체)
   - 노드 임베딩: 임의의 128차원 벡터
   - 엣지: PyTorch Geometric의 `edge_index` 형식
   - 예:
     ```python
     import torch
     data = torch.load("output/relationship_graph_pyg_llm.pt")
     print(data.x)  # 노드 임베딩
     print(data.edge_index)  # 엣지 연결 정보
     ```

주의사항
- OpenAI 사용 시 비용 및 호출률 제한을 확인하세요. 대량 처리 시 비용이 발생합니다.  
- DART API ZIP/XML 인코딩(utf-8 / euc-kr / cp949)을 다루므로 파싱 오류가 발생할 수 있음.  
- LLM 출력은 엄격한 JSON 형식을 요구합니다. 실패 시 로그에 원문(부분)을 출력하니 확인하세요.  
- 종목 매칭은 corp_name 기준으로 단순 문자열 매칭을 사용합니다. 필요시 매칭 로직을 보완하세요.

참고
- DART API 가이드: https://opendart.fss.or.kr/guide/main.do  
- OpenAI 문서: https://platform.openai.com/docs

라이선스 및 문의
- 내부 사용·연구용으로 적합합니다. 외부 배포 전 라이선스·API 이용약관을 확인하세요.  
- 문제가 발생하면 스크립트 로그를 확인하고, 필요한 경우 코드 내 주석을 참고해 엔진·토큰 길이·인코딩을 조정하십시오.
