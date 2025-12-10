# NOTE:
# - 필수 환경변수: OPENAI_API_KEY, DART_API_KEY
# - 의존성은 requirements.txt에 명시하세요.
#   예: pip install -r requirements.txt
# - DART API 문서: https://opendart.fss.or.kr/guide/main.do

import os
import zipfile
import requests
from bs4 import BeautifulSoup
import re
import networkx as nx
import torch
from tqdm import tqdm
import pandas as pd
import xmltodict
import io
import json
import pickle  # Added

# --- Replace hardcoded API keys with environment variables ---
from openai import OpenAI

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DART_API_KEY = os.getenv("DART_API_KEY")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY 환경변수를 설정하세요.")
if not DART_API_KEY:
    raise RuntimeError("DART_API_KEY 환경변수를 설정하세요.")

client = OpenAI(api_key=OPENAI_API_KEY)

def load_corp_code(api_key):
    url = f"https://opendart.fss.or.kr/api/corpCode.xml?crtfc_key={api_key}"

    res = requests.get(url)
    print("Content-Type:", res.headers.get("Content-Type"))

    z = zipfile.ZipFile(io.BytesIO(res.content))
    xml_name = z.namelist()[0]
    xml_data = z.read(xml_name)

    with open("CORPCODE.xml", "wb") as f:
        f.write(xml_data)
    print("XML 저장 완료: CORPCODE.xml")

    data = xmltodict.parse(xml_data)

    # DataFrame 생성
    corp_df = pd.DataFrame(data["result"]["list"])

    # 🔥 stock_code NOT NULL → 상장사만 추출
    corp_df = corp_df[corp_df["stock_code"].notnull() & (corp_df["stock_code"] != "")]
    corp_df = corp_df.reset_index(drop=True)

    return corp_df

corp_list = load_corp_code(DART_API_KEY)
print("CORP list loaded:", corp_list.shape)

def get_latest_business_report(corp_code):
    # 변경: API_KEY -> DART_API_KEY
    url = (
        "https://opendart.fss.or.kr/api/list.json"
        f"?crtfc_key={DART_API_KEY}&corp_code={corp_code}&bgn_de=20200101&pblntf_detail_ty=A001"
    )
    res = requests.get(url).json()

    if res.get("status") != "013" and res.get("list"):
        return res["list"][0]["rcept_no"]
    return None

def extract_html_from_document_zip(rcept_no):
    # 변경: API_KEY -> DART_API_KEY
    url = f"https://opendart.fss.or.kr/api/document.xml?crtfc_key={DART_API_KEY}&rcept_no={rcept_no}"
    res = requests.get(url)

    content = res.content

    if content[:2] == b'PK':
        print("✔ document.xml: ZIP 파일 감지")

        try:
            z = zipfile.ZipFile(io.BytesIO(content))
        except:
            raise Exception("❌ ZIP 파일 열기 실패")

        print("ZIP 내부:", z.namelist())

        # 1) 사업보고서 본문 파일 우선 선택
        main_file = f"{rcept_no}.xml"
        if main_file in z.namelist():
            raw = z.read(main_file)
            text = None
            for enc in ["utf-8", "euc-kr", "cp949"]:
                try:
                    text = raw.decode(enc)
                    break
                except:
                    pass

            if text is None:
                print(f"❌ 디코딩 실패: {main_file}")

            soup = BeautifulSoup(text, "html.parser")
            print(f"✔ HTML 파싱 성공: {main_file}")
            return soup

        for name in z.namelist():
          raw = z.read(name) # 디코딩 시도
          text = None
          for enc in ["utf-8", "euc-kr", "cp949"]:
              try:
                text = raw.decode(enc)
                break
              except:
                  pass

          if text is None:
              print(f"❌ 디코딩 실패: {name}")
              continue # BeautifulSoup 로 HTML 파싱 시도
          try:
            soup = BeautifulSoup(text, "html.parser")
            print(f"✔ HTML 파싱 성공: {name}")
            return soup
          except Exception as e:
            print(f"❌ HTML 파싱 실패: {name}", e)
            continue
        return None

    else:
      # -------------------------------
      # ② ZIP이 아니라 단일 XML 문서일 때
      # -------------------------------
      print("✔ document.xml: 단일 XML 문서 감지")

      try:
          text = content.decode("utf-8")
      except:
          try:
              text = content.decode("euc-kr")
          except:
              try:
                  text = content.decode("cp949")
              except:
                  raise Exception("❌ 단일 XML 디코딩 실패")

      soup = BeautifulSoup(text, "html.parser")
      print("✔ 단일 XML HTML 파싱 성공")
      return soup

import re

SECTION_TITLES = [
    r'Ⅰ\.\s*회사[의]? 개요',   r'I\.\s*회사[의]? 개요',
    r'Ⅱ\.\s*사업[의]? 내용',   r'II\.\s*사업[의]? 내용',
    r'계열회사\s*현황',
    r'종속기업[의\s]*개황',
    r'관계회사\s*현황',
]

SECTION_PATTERN = "(" + "|".join(SECTION_TITLES) + ")"

def split_sections(text):
    parts = re.split(SECTION_PATTERN, text)
    sections = {}
    current = None

    for p in parts:
        if re.match(SECTION_PATTERN, p):
            current = p
            sections[current] = ""
        else:
            if current:
                sections[current] += p + "\n"
    return sections

def extract_key_sections(text):
    all_sections = split_sections(text)
    result = {"business": "", "subsidiaries": ""}

    for title, body in all_sections.items():
        if re.search("사업[의 ]*내용", title):
            result["business"] += body
        if re.search("(계열회사|종속기업|관계회사)", title):
            result["subsidiaries"] += body

    return result

RELATION_KEYWORDS = [
    # governance
    "종속", "자회사", "관계회사", "계열회사", "계열", "지분", "지배", "합작",
    "joint venture", "JV", "associate", "affiliate", "subsidiary",
    "지배구조", "출자",

    # supplier
    "공급", "납품", "원재료", "재료공급", "부품공급", "supplier",

    # customer
    "고객사", "매출처", "주요고객", "판매처", "수요기업", "customer",

    # competitor
    "경쟁사", "경쟁기업", "경쟁사들", "경쟁",

    # 기술/협력
    "협력", "파트너", "기술제휴", "라이선스", "license", "oem",
    "기술협력", "기술공동",

    # 금융/투자
    "투자", "지분참여", "출자", "펀드", "loan", "금융지원", "underwriter",

    # 유통
    "유통", "물류", "배송", "logistics", "distribution partner",
]

NEGATIVE_PATTERNS = [
    "설명", "기준", "작성", "참고", "목적", "요약",
    "재무제표", "회계", "감사", "법령", "관련 규정",
    "기준일", "공시", "보고서", "총괄", "개요", "일반사항"
]

def split_sentences(text):
    # 점(.) , 다. , ? , ! , \n 로 구분
    pattern = r'(?<=[\.!?])\s+|(?<=다\.)\s+|\n+'
    s_list = re.split(pattern, text)
    return [s.strip() for s in s_list if len(s.strip()) > 8]


def extract_relation_sentences(text):
    sentences = split_sentences(text)
    result = []

    for s in sentences:
        # 관계 키워드 포함 여부
        if not any(k in s for k in RELATION_KEYWORDS):
            continue

        # 불필요한 문장 제거
        if any(n in s for n in NEGATIVE_PATTERNS):
            continue

        # 너무 짧은 문장 제거
        if len(s) < 15:
            continue

        result.append(s.strip())

    return result

from bs4 import BeautifulSoup

def extract_subsidiary_table(soup):
    """
    DART OOXML 기반 HTML/XML에서 종속/계열회사 테이블을 robust하게 추출하는 최종본
    """

    # 표 후보 전체 수집
    tables = soup.find_all(["table", "TABLE"])
    subsidiaries = []

    if not tables:
        print("⚠️ 표를 찾지 못함")
        return []

    print(f"표 전체 개수: {len(tables)}")

    # 헤더 키워드
    HEADER_MAP = {
        "기업명": "name",
        "회사명": "name",
        "법인명": "name",
        "상호": "name",
        "Subsidiary": "name",

        "업종": "business",
        "사업": "business",
        "Industry": "business",

        "지역": "region",
        "국가": "region",
        "Country": "region",

        "지분율": "share_ratio",
        "지분율(%)": "share_ratio",
        "Ownership": "share_ratio",
    }

    for idx, tbl in enumerate(tables):
        rows = tbl.find_all(["tr", "TR"])
        if len(rows) < 2:
            continue

        # --------------------------------------------------------
        # 1) 헤더 탐색
        # --------------------------------------------------------
        header = None
        header_idx = None

        for i, r in enumerate(rows[:5]):  # 보통 상위 5줄 안에 header 존재
            cols = r.find_all(["th", "TH", "td", "TD"])
            texts = [c.get_text(" ", strip=True) for c in cols]
            joined = "".join(texts)

            # "기업명", "지분율" 등 핵심 헤더 포함 여부
            if any(key in joined for key in HEADER_MAP.keys()):
                header = texts
                header_idx = i
                break

        if header is None:
            continue

        # print(f"✔ 표 {idx}에서 header 발견: {header}")

        # --------------------------------------------------------
        # 2) header 이름을 정규화
        # --------------------------------------------------------
        normalized_headers = []
        for h in header:
            mapped = None
            for key, val in HEADER_MAP.items():
                if key in h:
                    mapped = val
                    break
            normalized_headers.append(mapped if mapped else h)

        # --------------------------------------------------------
        # 3) row 추출
        # --------------------------------------------------------
        for r in rows[header_idx + 1:]:
            cols = r.find_all(["td", "TD", "th", "TH"])
            if len(cols) != len(header):
                continue  # irregular row skip

            values = [c.get_text(" ", strip=True) for c in cols]
            row_dict = dict(zip(normalized_headers, values))

            # 회사명 없으면 데이터 아님
            if not row_dict.get("name"):
                continue

            subsidiaries.append({
                "name": row_dict.get("name"),
                "region": row_dict.get("region"),
                "business": row_dict.get("business"),
                "share_ratio": row_dict.get("share_ratio"),
                "raw": row_dict
            })

    print(f"✔ 최종 추출된 계열회사 수: {len(subsidiaries)}")
    return subsidiaries

def clean_json_output(text):
    # 코드블럭 제거
    text = text.strip()
    if text.startswith("```"):
        # 첫 번째 ```
        text = text.split("```", 1)[1]
        # 두 번째 ```
        text = text.split("```", 1)[0]

    # 혹시 "json" 같은 언어 태그 제거
    text = text.replace("json", "", 1).strip()

    return text

def extract_relations_llm(text, corp_name):
    prompt = f"""
당신은 금융·산업 분석 전문가입니다.
아래 텍스트는 한 기업의 사업보고서이며, 목적은 “기업 간 관계 그래프”를 구축하는 것입니다.
이 그래프는 GNN 기반 뉴스 영향 예측 모델에서 사용됩니다.

당신의 임무는 다음의 모든 관계 유형을 가능한 한 정확하게 추출하는 것입니다.

────────────────────────────────────────
[1] Supplier(공급업체)
- raw_supplier
- component_supplier
- manufacturing_outsource

[2] Customer(고객사)
- major_customer
- b2b_customer
- b2c_channel

[3] Competitor(경쟁사)
- direct_competitor
- market_competitor
- potential_competitor

[4] Product/기술 협력
- tech_partner
- license_in
- license_out
- oem_partner

[5] Governance(지배구조)
- subsidiary
- sub_subsidiary
- affiliate
- associate
- joint_venture
- parent
- major_shareholder (개인은 제외)

[6] 공공기관/규제
- regulator
- public_customer
- public_supplier

[7] 금융/투자
- loan_provider
- bond_underwriter
- investment_partner

[8] 물류·유통
- logistics_partner
- distribution_partner
- retail_channel
────────────────────────────────────────

엄격한 규칙:
- 개인 이름(예: 홍길동)은 절대 target에 포함하지 않습니다.
- target은 반드시 기업/법인/기관·단체만 가능합니다.
- 불확실한 관계는 생성하지 않습니다.
- source는 반드시 "{corp_name}" 입니다.
- JSON 외 어떤 텍스트도 출력하지 마십시오.
- 코드블록(```)도 금지합니다.

반드시 아래 JSON 형식을 그대로 따르십시오:

{{
  "relations": [
    {{
      "source": "{corp_name}",
      "target": "기업명",
      "type": "위 관계 타입 중 하나",
      "evidence": "원문 문장"
    }}
  ]
}}

아래는 사업보고서 본문입니다. 기업 간 관계만 추출하십시오.

텍스트:
{text}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )

    raw = response.choices[0].message.content

    # 코드블럭 제거
    cleaned = clean_json_output(raw)

    # JSON 파싱
    try:
        return json.loads(cleaned).get("relations", [])
    except Exception as e:
        print("JSON 파싱 실패:", cleaned[:500])
        return []

def normalize_name(name):
    return name.replace("(주)", "").replace("㈜","").strip()

def find_corp_info(name, corp_list):
    norm = normalize_name(name)

    # corp_name 정규화한 임시 컬럼이 있는 경우 활용
    if "norm_name" not in corp_list.columns:
        corp_list["norm_name"] = corp_list["corp_name"].apply(normalize_name)

    matches = corp_list[corp_list["norm_name"] == norm]

    if matches.empty:
        return None, None  # corp_code, stock_code
    row = matches.iloc[0]
    return row["corp_code"], row["stock_code"]

def extract_text_from_soup(soup):
    return soup.get_text(" ", strip=True)

def chunk_text(text, max_chars=8000):
    chunks = []
    for i in range(0, len(text), max_chars):
        chunks.append(text[i:i+max_chars])
    return chunks

import networkx as nx
from tqdm import tqdm

def build_graph_llm(corp_codes):
    G = nx.DiGraph()

    for corp_code in tqdm(corp_codes):
        print("="*80)
        print(f"📌 corp_code: {corp_code}")
        row = corp_list[corp_list["corp_code"] == corp_code]
        if row.empty:
            continue

        corp_name = row.iloc[0]["corp_name"]

        print(f"회사명: {corp_name}")
        rcept_no = get_latest_business_report(corp_code)

        print(f"📄 rcept_no: {rcept_no}")

        if not rcept_no:
            print("⚠️ 최신 사업보고서가 없음, 스킵:", corp_name)
            continue

        soup = extract_html_from_document_zip(rcept_no)
        if soup is None:
            print("⚠️ 문서 파싱 실패, 스킵:", corp_name)
            continue

        text = extract_text_from_soup(soup)

        sections = extract_key_sections(text)
        business_text = sections["business"]
        # subsidiary_text = sections["subsidiaries"]

        # 🔥 계열회사 table
        subsidiary_rows = extract_subsidiary_table(soup)

        print("사업의 내용 길이:", len(business_text))
        print("표에서 추출한 계열회사 수:", len(subsidiary_rows))

        # 🔥 관계 문장
        rel_sentences = extract_relation_sentences(business_text)

        llm_input = "\n".join(rel_sentences)
        chunks = chunk_text(llm_input, max_chars=8000)

        all_rel = []
        for i, chunk in enumerate(chunks):
            print(f"🤖 GPT 처리 중 (chunk {i+1}/{len(chunks)})...")
            r = extract_relations_llm(chunk, corp_name)
            if r:
                all_rel.extend(r)

        print("🔍 GPT가 추출한 총 관계 수:", len(all_rel))

        edge_set = set()

        # 🔥 1) GPT 관계 추가
        for r in all_rel:
            src = r["source"]
            tgt = r["target"]
            rtype = r["type"]

            src_n = normalize_name(src)
            tgt_n = normalize_name(tgt)
            key = (src_n, tgt_n, rtype)

            if key in edge_set: continue
            edge_set.add(key)

            src_corp, src_stock = find_corp_info(src, corp_list)
            tgt_corp, tgt_stock = find_corp_info(tgt, corp_list)

            if (tgt_corp is None) or (tgt_stock is None):
                continue

            G.add_node(src, corp_code=src_corp, stock_code=src_stock)
            G.add_node(tgt, corp_code=tgt_corp, stock_code=tgt_stock)
            G.add_edge(src, tgt, relation=rtype)

        # 🔥 2) 테이블 기반 계열회사 추가
        for row in subsidiary_rows:
            src = corp_name
            tgt = row["name"]

            key = (src, tgt, "subsidiary")
            if key in edge_set:
                continue
            edge_set.add(key)

            src_c, src_s = find_corp_info(src, corp_list)
            tgt_c, tgt_s = find_corp_info(tgt, corp_list)

            G.add_node(src, corp_code=src_c, stock_code=src_s)
            G.add_node(tgt, corp_code=tgt_c, stock_code=tgt_s, region=row["region"], business=row["business"])

            G.add_edge(src, tgt, relation="subsidiary", share=row["share_ratio"])


    return G

from torch_geometric.data import Data

def graph_to_pyg(G, embedding_dim=128):
    idx = {node: i for i, node in enumerate(G.nodes())}

    # 임시 임베딩 (GPU 메모리 절약 목적)
    x = torch.randn((len(G.nodes()), embedding_dim))

    edges = []
    for src, dst in G.edges():
        edges.append([idx[src], idx[dst]])
    if len(edges) == 0:
        # 빈 그래프 처리: 빈 edge_index
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edges).t().contiguous()

    data = Data(x=x, edge_index=edge_index)
    return data

import pandas as pd

# 1) CSV 로드
theme_df = pd.read_csv("theme_stock.csv")

def extract_stock_names(row):
    if pd.isnull(row):
        return []
    return [x.strip() for x in str(row).replace("\t", ",").replace(" ", ",").split(",") if x.strip()]

# 2) 종목 목록 전체 추출
stock_names = []
for items in theme_df["종목 목록"]:
    stock_names.extend(extract_stock_names(items))
stock_names = list(set(stock_names))
print("총 종목 수:", len(stock_names))
print(stock_names)

# 3) corp_list에서 corp_code 매칭
def get_corp_codes_from_names(stock_names, corp_list):
    codes = []
    for name in stock_names:
        match = corp_list[corp_list["corp_name"].str.contains(name, na=False)]
        if not match.empty:
            codes.extend(list(match["corp_code"]))
        else:
            print(f"⚠️ 매칭 실패: {name}")
    return list(set(codes))

filtered_corp_codes = get_corp_codes_from_names(stock_names, corp_list)

print("필터링된 corp_code 수:", len(filtered_corp_codes))
print(filtered_corp_codes)


# 4) 그래프 생성
G = build_graph_llm(filtered_corp_codes)

print("Nodes:", G.number_of_nodes())
print("Edges:", G.number_of_edges())

# 저장 디렉토리: 기본 로컬 output 또는 환경변수 SAVE_DIR 사용
save_dir = os.getenv("SAVE_DIR", "output")
os.makedirs(save_dir, exist_ok=True)
print("저장 폴더:", save_dir)

# 예시 저장 (실제 실행 시 주석 해제)
data = graph_to_pyg(G)
with open(f"{save_dir}/relationship_graph_llm.pkl", "wb") as f:
    pickle.dump(G, f)
torch.save(data, f"{save_dir}/relationship_graph_pyg_llm.pt")
