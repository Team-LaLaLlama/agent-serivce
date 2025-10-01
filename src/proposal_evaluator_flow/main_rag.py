# main_rag.py
# 제안서 평가를 위한 통합 RAG 시스템
# - 제안서, RFP, 사내 정보를 모두 임베딩하여 통합 벡터스토어 생성
# - CrewAI Agent들이 벡터스토어를 활용하여 제안서를 자동 평가

import os
import asyncio
import json
import glob
import torch
import chromadb
from dotenv import load_dotenv
from crewai import Agent, Task, Crew
from crewai.llm import LLM

# LangChain 관련 라이브러리 임포트
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

load_dotenv()

# =================================================================
# 1. RAG 파이프라인 설정 및 함수 정의
# =================================================================

# --- 경로 설정 ---
PROPOSAL_DIR = "./proposal"  # 제안서 디렉토리
RFP_PATH = "./RFP/수협_rfp.pdf"  # RFP 파일 경로
INTERNAL_DATA_DIR = "./internal_data"  # 사내 정보 디렉토리

# --- 전역 변수 ---
# 생성된 벡터스토어를 저장할 전역 변수
# Agent들이 공유해서 사용할 수 있도록 합니다.
unified_vectorstore = None

def initialize_rag_components():
    """RAG에 필요한 임베딩 모델과 ChromaDB 클라이언트를 초기화합니다."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"INFO: RAG 디바이스로 '{device}'를 사용합니다.")
    
    print("INFO: 임베딩 모델을 로딩합니다...")
    embedding_model = HuggingFaceEmbeddings(
        model_name="Qwen/Qwen3-Embedding-0.6B",
        model_kwargs={'device': device},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    chroma_client = chromadb.PersistentClient(path="./chroma_db_crewai")
    print("INFO: 임베딩 모델 및 ChromaDB 클라이언트 초기화 완료.")
    return embedding_model, chroma_client

def load_document(file_path, doc_type, proposal_name):
    """
    제안서/RFP 문서를 로드하는 함수

    Args:
        file_path (str): 로드할 파일의 경로
        doc_type (str): 문서 타입 ("RFP" 또는 "제안서")
        proposal_name (str): 제안서 파일명 (RFP의 경우 "RFP")

    Returns:
        list[Document]: 로드된 Document 객체 리스트
    """
    documents = []
    print(f"  - [{doc_type}] '{os.path.basename(file_path)}' 로딩 중...")

    # PDF 파일 로드 (PyPDFLoader 사용)
    if file_path.endswith('.pdf'):
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        # 각 Document에 메타데이터 추가 (문서 타입, 제안서명)
        for doc in docs:
            doc.metadata.update({"doc_type": doc_type, "proposal_name": proposal_name})
        documents.extend(docs)

    # 텍스트 파일 로드 (TextLoader 사용)
    elif file_path.endswith('.txt'):
        loader = TextLoader(file_path, encoding='utf-8')
        docs = loader.load()
        # 각 Document에 메타데이터 추가
        for doc in docs:
            doc.metadata.update({"doc_type": doc_type, "proposal_name": proposal_name})
        documents.extend(docs)

    # HTML 파일 로드 (추가)
    elif file_path.endswith('.html'):
        loader = TextLoader(file_path, encoding='utf-8')
        docs = loader.load()
        for doc in docs:
            doc.metadata.update({"doc_type": doc_type, "proposal_name": proposal_name})
        documents.extend(docs)

    print(f"    → {len(documents)}개 섹션 로드됨")
    return documents


# =================================================================
# 사내 정보 로더 함수들
# =================================================================

def load_internal_structured_data(file_path, doc_type_prefix):
    """
    정형화된 사내 정보 데이터를 로드하는 함수

    데이터 형식: '---'로 구분된 항목들
    각 항목은 key: value 형식의 메타데이터를 포함

    Args:
        file_path (str): 사내 정보 파일 경로
        doc_type_prefix (str): 문서 타입 접두사 (예: "사내_기술스택")

    Returns:
        list[Document]: 로드된 Document 객체 리스트
    """
    documents = []

    # 파일이 존재하지 않으면 빈 리스트 반환
    if not os.path.exists(file_path):
        print(f"  ⚠ 파일이 존재하지 않습니다: {file_path}")
        return documents

    print(f"  - [{doc_type_prefix}] '{os.path.basename(file_path)}' 로딩 중...")

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # '---'로 구분된 각 항목을 개별 Document로 변환
    entries = content.split('---')

    for entry in entries:
        entry = entry.strip()
        if not entry:  # 빈 항목은 스킵
            continue

        # 기본 메타데이터 설정
        metadata = {"doc_type": doc_type_prefix}

        # 각 줄을 파싱하여 메타데이터 추출
        # 형식: key: value
        lines = entry.split('\n')
        for line in lines:
            if ':' in line:
                key, value = line.split(':', 1)
                # 메타데이터에 키-값 저장 (공백 제거)
                metadata[key.strip()] = value.strip()

        # Document 생성 (전체 내용을 page_content로, 파싱된 정보는 metadata로)
        doc = Document(
            page_content=entry,  # 원본 텍스트 그대로 저장
            metadata=metadata
        )
        documents.append(doc)

    print(f"    → {len(documents)}개 항목 로드됨")
    return documents


def load_all_internal_data(internal_data_dir):
    """
    사내 정보 디렉토리의 모든 파일을 자동으로 로드하는 함수

    파일명 규칙:
    - tech_stacks*.txt → 사내_기술스택
    - contacts*.txt → 사내_담당자
    - migrations*.txt → 사내_마이그레이션
    - incidents*.txt → 사내_장애이력

    Args:
        internal_data_dir (str): 사내 정보 디렉토리 경로

    Returns:
        list[Document]: 모든 사내 정보 Document 리스트
    """
    all_internal_docs = []

    if not os.path.exists(internal_data_dir):
        print(f"  ⚠ 사내 정보 디렉토리가 존재하지 않습니다: {internal_data_dir}")
        return all_internal_docs

    print(f"\n[사내 정보 로드 시작: {internal_data_dir}]")

    # 디렉토리 내 모든 .txt 파일 검색
    internal_files = glob.glob(os.path.join(internal_data_dir, "*.txt"))

    for file_path in internal_files:
        filename = os.path.basename(file_path).lower()

        # 파일명에 따라 문서 타입 자동 분류
        if 'tech_stack' in filename or 'technology' in filename:
            docs = load_internal_structured_data(file_path, "사내_기술스택")
            all_internal_docs.extend(docs)

        elif 'contact' in filename or 'person' in filename:
            docs = load_internal_structured_data(file_path, "사내_담당자")
            all_internal_docs.extend(docs)

        elif 'migration' in filename:
            docs = load_internal_structured_data(file_path, "사내_마이그레이션")
            all_internal_docs.extend(docs)

        elif 'incident' in filename or 'failure' in filename:
            docs = load_internal_structured_data(file_path, "사내_장애이력")
            all_internal_docs.extend(docs)

        else:
            # 분류되지 않은 파일은 일반 사내 정보로 처리
            docs = load_internal_structured_data(file_path, "사내_기타")
            all_internal_docs.extend(docs)

    print(f"✅ 총 {len(all_internal_docs)}개의 사내 정보 항목 로드 완료\n")
    return all_internal_docs

def create_unified_vectorstore(
    proposal_files,
    rfp_path,
    internal_data_dir,
    embedding_model,
    chroma_client,
    collection_name="proposal_evaluation_store"
):
    """
    제안서, RFP, 사내 정보를 모두 포함하는 통합 벡터스토어를 생성하는 함수

    Args:
        proposal_files (list): 제안서 파일 경로 리스트
        rfp_path (str): RFP 파일 경로
        internal_data_dir (str): 사내 정보 디렉토리 경로
        embedding_model: HuggingFace 임베딩 모델
        chroma_client: ChromaDB 클라이언트
        collection_name (str): 벡터스토어 컬렉션 이름

    Returns:
        Chroma: 생성된 벡터스토어 객체
    """
    print(f"\n{'='*70}")
    print(f"  통합 벡터스토어 생성 시작 (Collection: {collection_name})")
    print(f"{'='*70}")

    all_documents = []

    # 1. RFP 문서 로드
    print("\n[1단계] RFP 문서 로드")
    if os.path.exists(rfp_path):
        all_documents.extend(load_document(rfp_path, "RFP", "RFP"))
    else:
        print(f"  ⚠ RFP 파일이 존재하지 않습니다: {rfp_path}")

    # 2. 제안서 문서 로드
    print("\n[2단계] 제안서 문서 로드")
    for proposal_path in proposal_files:
        proposal_name = os.path.basename(proposal_path)
        all_documents.extend(load_document(proposal_path, "제안서", proposal_name))

    # 3. 사내 정보 로드 (핵심 추가 부분!)
    print("\n[3단계] 사내 정보 로드")
    internal_docs = load_all_internal_data(internal_data_dir)
    all_documents.extend(internal_docs)

    # 전체 로드 완료 통계
    print(f"\n{'='*70}")
    print(f"  ✅ 총 {len(all_documents)}개 문서 섹션 로드 완료")
    print(f"{'='*70}")

    # 4. 텍스트 청크 분할
    # - 제안서/RFP: 긴 문서이므로 청크 분할 필요
    # - 사내 정보: 이미 작은 단위이지만 통일성을 위해 동일하게 처리
    print("\n[4단계] 텍스트 청크 분할")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,      # 청크 크기: 500자 (문맥 유지)
        chunk_overlap=50,    # 청크 간 오버랩: 50자 (경계 정보 유지)
        separators=["\n\n", "\n", ". ", " ", ""]  # 분할 우선순위
    )
    splits = text_splitter.split_documents(all_documents)
    print(f"  ✅ {len(splits)}개 청크로 분할 완료")

    # 5. 기존 컬렉션 삭제 (있을 경우)
    print("\n[5단계] 기존 컬렉션 확인 및 삭제")
    try:
        chroma_client.delete_collection(name=collection_name)
        print(f"  ✅ 기존 '{collection_name}' 컬렉션 삭제 완료")
    except Exception:
        print(f"  ℹ️ 기존 컬렉션 없음 (신규 생성)")

    # 6. 벡터스토어 생성 (임베딩 + 인덱싱)
    print("\n[6단계] 벡터 임베딩 및 인덱싱")
    print(f"  ⏳ {len(splits)}개 청크를 임베딩 중... (수 분 소요)")
    vectorstore = Chroma.from_documents(
        documents=splits,              # 청크 분할된 Document 리스트
        embedding=embedding_model,     # 임베딩 모델
        collection_name=collection_name,  # 컬렉션 이름
        client=chroma_client          # ChromaDB 클라이언트
    )

    print(f"\n{'='*70}")
    print(f"  🎉 통합 벡터스토어 생성 완료!")
    print(f"  - 총 청크 수: {len(splits)}개")
    print(f"  - 컬렉션명: {collection_name}")
    print(f"{'='*70}\n")

    return vectorstore

def get_context_for_topic(proposal_file, topic):
    """
    제안서에서 특정 토픽과 관련된 내용을 벡터스토어에서 검색하는 함수

    Args:
        proposal_file (str): 제안서 파일명 (예: "A사_제안서.txt")
        topic (str): 검색할 토픽 (예: "시스템 아키텍처")

    Returns:
        str: 검색된 관련 내용 (여러 청크를 합친 문자열)
    """
    global unified_vectorstore

    # 벡터스토어가 초기화되지 않은 경우 에러 메시지 반환
    if unified_vectorstore is None:
        return "오류: 벡터스토어가 초기화되지 않았습니다."

    print(f"  🔍 RAG 검색 실행 -> 제안서: '{proposal_file}', 토픽: '{topic}'")

    # 벡터스토어에서 유사도 검색 수행
    # - query: 검색 쿼리 (토픽)
    # - k: 반환할 결과 개수 (상위 2개)
    # - filter: 메타데이터 필터링 (특정 제안서만 검색)
    results = unified_vectorstore.similarity_search(
        query=topic,
        k=2,  # 관련성이 높은 상위 2개 청크만 가져옴
        filter={"proposal_name": proposal_file}  # 특정 제안서로 필터링
    )

    # 검색 결과가 없는 경우
    if not results:
        return "관련 내용을 찾을 수 없습니다."

    # 검색된 여러 청크를 하나의 문자열로 합침
    context = "\n\n---\n\n".join([doc.page_content for doc in results])
    return context


def search_internal_info(query, doc_type=None, k=3):
    """
    사내 정보를 벡터스토어에서 검색하는 함수
    (사용자 질문: "타 계열사에서 Kafka 쓰는 곳 있나?" 등에 사용)

    Args:
        query (str): 검색 쿼리 (예: "Kafka 장애 이력")
        doc_type (str, optional): 문서 타입 필터 (예: "사내_기술스택")
        k (int): 반환할 결과 개수 (기본값: 3)

    Returns:
        list[Document]: 검색된 Document 리스트
    """
    global unified_vectorstore

    if unified_vectorstore is None:
        print("⚠ 벡터스토어가 초기화되지 않았습니다.")
        return []

    print(f"  🔍 사내 정보 검색 -> 쿼리: '{query}', 타입: '{doc_type or '전체'}'")

    # 메타데이터 필터 생성
    filter_dict = {}
    if doc_type:
        filter_dict["doc_type"] = doc_type

    # 벡터스토어에서 검색
    results = unified_vectorstore.similarity_search(
        query=query,
        k=k,
        filter=filter_dict if filter_dict else None
    )

    print(f"  ✅ {len(results)}개 결과 발견")
    return results


# =================================================================
# 사내 정보 검색 헬퍼 함수들 (선택적 사용)
# =================================================================

def search_tech_stack(technology_name, k=3):
    """특정 기술 스택 정보 검색"""
    return search_internal_info(technology_name, doc_type="사내_기술스택", k=k)


def search_contact(query, k=3):
    """담당자 정보 검색"""
    return search_internal_info(query, doc_type="사내_담당자", k=k)


def search_migration_case(technology_name, k=2):
    """마이그레이션 사례 검색"""
    return search_internal_info(f"{technology_name} 마이그레이션", doc_type="사내_마이그레이션", k=k)


def search_incident(technology_name, k=2):
    """장애 이력 검색"""
    return search_internal_info(f"{technology_name} 장애", doc_type="사내_장애이력", k=k)


#==============================================================


# 2. CrewAI Agent 및 프로세스 정의
# =================================================================

async def main():
    """
    메인 함수: 제안서 자동 평가 프로세스 실행

    전체 흐름:
    1. RAG 파이프라인 초기화 (임베딩 모델, 벡터스토어)
    2. Phase 1: 심사 항목 자동 분류 (Dispatcher Agent)
    3. Phase 2: 대분류별 전문가 Agent가 병렬 평가
    4. Phase 3: 최종 보고서 작성 (Reporting Agent)
    """
    print("\n" + "="*70)
    print("  동적 Agent 생성 및 평가 프로세스를 시작합니다.")
    print("="*70)

    # --- [전제] RAG 파이프라인 초기화 ---
    # 메인 프로세스 시작 전, 벡터스토어를 먼저 준비합니다.
    global unified_vectorstore

    # 제안서 파일 검색 (.txt, .html 등)
    proposal_files = glob.glob(os.path.join(PROPOSAL_DIR, "*.txt"))
    proposal_files.extend(glob.glob(os.path.join(PROPOSAL_DIR, "*.html")))

    # 파일 존재 여부 확인
    if not proposal_files:
        print("❌ 오류: 제안서 파일이 없습니다. 경로를 확인하세요.")
        print(f"   - 제안서 디렉토리: {PROPOSAL_DIR}")
        return

    if not os.path.exists(RFP_PATH):
        print("❌ 오류: RFP 파일이 없습니다. 경로를 확인하세요.")
        print(f"   - RFP 경로: {RFP_PATH}")
        return

    print(f"\n✅ 제안서 파일 {len(proposal_files)}개 발견:")
    for pf in proposal_files:
        print(f"   - {os.path.basename(pf)}")

    # RAG 컴포넌트 초기화 (임베딩 모델, ChromaDB)
    embedding_model, chroma_client = initialize_rag_components()

    # 통합 벡터스토어 생성 (제안서 + RFP + 사내정보)
    unified_vectorstore = create_unified_vectorstore(
        proposal_files=proposal_files,
        rfp_path=RFP_PATH,
        internal_data_dir=INTERNAL_DATA_DIR,  # 사내 정보 디렉토리 추가!
        embedding_model=embedding_model,
        chroma_client=chroma_client
    )
    # --- RAG 초기화 완료 ---

    # 전체 심사 항목 리스트 (비정형 데이터)
    # 실제로는 심사기준표 파일에서 로드할 수 있음
    unstructured_evaluation_items = [
        {"대분류": "기술", "topic": "시스템 아키텍처", "criteria": "MSA 기반의 유연하고 확장 가능한 아키텍처인가?"},
        {"대분류": "관리", "topic": "프로젝트 관리 방안", "criteria": "WBS 기반의 상세하고 실현 가능한 일정을 제시하였는가?"},
        {"대분류": "기술", "topic": "데이터베이스 암호화", "criteria": "개인정보보호 및 데이터 암호화 방안이 명시되었는가?"},
        {"대분류": "관리", "topic": "투입 인력 계획", "criteria": "투입 인력의 역할과 경력이 적절한가?"},
        {"대분류": "가격", "topic": "비용 산정 내역", "criteria": "제시된 비용이 합리적이고 구체적인 근거를 포함하는가?"},
    ]

    # --- LLM 정의 ---
    # Ollama 로컬 서버 사용 (llama3.2 모델)
    # 또는 HuggingFace Inference API 사용 가능
    llm = LLM(model="ollama/llama3.2", base_url="http://localhost:11434")

    # =================================================================
    # Phase 1: Dispatcher가 대분류를 스스로 찾아내고 항목 분류
    # =================================================================
    # 목적: 비정형 심사 항목 리스트를 '대분류' 기준으로 자동 그룹화
    # 예: {"기술": [...], "관리": [...], "가격": [...]}
    print("\n" + "="*70)
    print("  [Phase 1] 심사 항목 자동 분류")
    print("="*70)
    
    dispatcher_agent = Agent(
        role="평가 항목 자동 분류 및 그룹화 전문가",
        goal="주어진 심사 항목 리스트에서 '대분류'를 기준으로 모든 항목을 그룹화하여 JSON으로 반환",
        backstory="당신은 복잡한 목록을 받아서 주요 카테고리별로 깔끔하게 정리하고 구조화하는 데 매우 뛰어난 능력을 가졌습니다.",
        llm=llm,
        verbose=True
    )

    items_as_string = json.dumps(unstructured_evaluation_items, ensure_ascii=False)
    
    dispatcher_task = Task(
        description=f"""아래 심사 항목 리스트를 '대분류' 키 값을 기준으로 그룹화해주세요.
        [전체 심사 항목 리스트]: {items_as_string}
        결과 JSON의 key는 리스트에 존재하는 '대분류'의 이름이어야 합니다.
        각 항목의 '대분류', 'topic', 'criteria' 키와 값을 모두 그대로 유지해야 합니다.
        """,
        expected_output="JSON 객체. 각 key는 심사 항목 리스트에 있던 '대분류'이며, value는 해당 대분류에 속하는 항목 객체들의 리스트입니다. 각 객체는 원본의 모든 키-값을 포함해야 합니다.",
        agent=dispatcher_agent
    )

    dispatcher_crew = Crew(agents=[dispatcher_agent], tasks=[dispatcher_task], verbose=False)
    categorization_result = dispatcher_crew.kickoff()
    
    try:
        # LLM이 생성한 결과물에서 JSON 부분만 추출
        json_string = categorization_result.raw[categorization_result.raw.find('{'):categorization_result.raw.rfind('}')+1]
        categorized_items = json.loads(json_string)
        print("✅ 항목 분류 완료. 발견된 대분류:")
        for category, items in categorized_items.items():
            print(f"  - {category}: {len(items)}개 항목")
    except (json.JSONDecodeError, IndexError):
        print("❌ 항목 분류 실패!")
        print(f"   - 원본 결과: {categorization_result.raw}")
        categorized_items = {}

    # =================================================================
    # Phase 2: 대분류 개수만큼 동적으로 Agent를 생성하고 병렬 평가
    # =================================================================
    # 목적:
    # - 각 대분류(기술, 관리, 가격 등)별로 전문가 Agent를 동적 생성
    # - 각 제안서를 순회하며 모든 심사 항목을 병렬 평가
    # - RAG를 통해 제안서에서 관련 내용을 자동 추출하여 평가 근거로 활용
    print("\n" + "="*70)
    print("  [Phase 2] 발견된 대분류별로 전문가 Agent를 동적으로 생성하여 병렬 평가합니다")
    print("="*70)

    # 모든 제안서 파일에 대해 평가를 반복합니다.
    for proposal_path in proposal_files:
        proposal_name = os.path.basename(proposal_path)
        print(f"\n\n{'='*20} [{proposal_name}] 평가 시작 {'='*20}")

        # 제안서별로 Agent와 Task 리스트 초기화
        specialist_agents = []  # 전문가 Agent 리스트
        evaluation_tasks = []   # 평가 Task 리스트

        # 대분류별로 전문가 Agent를 동적 생성
        for category, items in categorized_items.items():
            # 대분류별 전문가 Agent 생성 (예: "기술 부문 전문 평가관")
            specialist_agent = Agent(
                role=f"'{category}' 부문 전문 평가관",
                goal=f"'{proposal_name}' 제안서의 '{category}' 부문을 전문적으로 평가",
                backstory=f"당신은 '{category}' 분야 최고의 전문가로서, 주어진 관련 내용을 바탕으로 심사 기준에 따라 제안서를 냉철하게 분석하고 평가 보고서를 작성해야 합니다.",
                llm=llm,
                verbose=True
            )
            specialist_agents.append(specialist_agent)

            # 해당 대분류의 모든 심사 항목에 대한 Task 생성
            for item in items:
                # RAG를 통해 제안서에서 관련 내용 검색
                # - 벡터스토어에서 토픽과 유사한 내용을 자동으로 찾아옴
                context = get_context_for_topic(proposal_name, item['topic'])

                # 평가 Task 생성
                task = Task(
                    description=f"""'{proposal_name}' 제안서의 '{category}' 부문 중 '{item['topic']}' 항목을 평가하시오.

- 심사 기준: {item['criteria']}

- 제안서 관련 내용:
---
{context}
---

위 내용을 근거로 평가를 수행하고 보고서를 작성하시오.""",
                    expected_output=f"'{item['topic']}' 항목에 대한 평가 보고서. 반드시 [평가 점수(1-100)], [평가 요약], [판단 근거] 세 가지 항목을 포함해야 합니다.",
                    agent=specialist_agent  # 해당 대분류 전문가에게 할당
                )
                evaluation_tasks.append(task)

        # Task가 없으면 다음 제안서로
        if not evaluation_tasks:
            print("⚠ 평가할 작업이 없습니다.")
            continue

        # Crew 구성 및 병렬 평가 실행
        # - 여러 전문가 Agent가 각자의 Task를 동시에 수행
        print(f"\n⏳ {len(evaluation_tasks)}개 평가 항목을 처리 중...")
        evaluation_crew = Crew(
            agents=specialist_agents,  # 동적으로 생성된 전문가 Agent 리스트
            tasks=evaluation_tasks,    # 동적으로 생성된 Task 리스트
            verbose=True
        )
        final_results = await evaluation_crew.kickoff_async()  # 비동기 병렬 실행

        # =================================================================
        # Phase 3: 최종 보고서 작성 (Reporting Agent)
        # =================================================================
        # 목적:
        # - 모든 개별 평가 보고서를 종합하여 하나의 최종 보고서 작성
        # - 제안서 전체에 대한 종합 평가, 강점/약점 분석, 최종 점수 제시
        print(f"\n{'='*70}")
        print(f"  [Phase 3] [{proposal_name}] 최종 보고서 작성")
        print(f"{'='*70}")

        # 개별 평가 보고서들을 하나의 문자열로 합침
        individual_reports = "\n\n".join([str(result) for result in final_results])

        # 최종 보고서 작성 Agent 생성
        reporting_agent = Agent(
            role="수석 평가 분석가",
            goal="여러 개의 개별 평가 보고서를 종합하여, 경영진이 의사결정을 내릴 수 있도록 하나의 완성된 최종 보고서를 작성",
            backstory="당신은 여러 부서의 보고를 취합하여 핵심만 요약하고, 전체적인 관점에서 강점과 약점을 분석하여 최종 보고서를 작성하는 데 매우 능숙합니다.",
            llm=llm,
            verbose=True
        )

        # 최종 보고서 작성 Task 생성
        reporting_task = Task(
            description=f"""아래는 '{proposal_name}' 제안서에 대한 각 분야 전문가들의 개별 평가 보고서입니다.

[개별 평가 보고서 목록]
{individual_reports}

위 보고서들을 모두 종합하여, 제안서 전체에 대한 최종 평가 보고서를 작성해주세요.
보고서는 다음 항목을 포함해야 합니다:
- [총평]: 제안서 전체에 대한 종합 평가
- [주요 강점]: 제안서의 뛰어난 점 (상위 3개)
- [주요 약점]: 제안서의 부족한 점 (상위 3개)
- [최종 추천 점수(1-100)]: 종합 점수 및 선정 추천 여부""",
            expected_output="하나의 완성된 최종 평가 보고서.",
            agent=reporting_agent
        )

        # 최종 보고서 생성 Crew 실행
        reporting_crew = Crew(agents=[reporting_agent], tasks=[reporting_task], verbose=False)
        final_comprehensive_report = reporting_crew.kickoff()

        # 최종 보고서 출력
        print(f"\n\n{'='*70}")
        print(f"  🎉 [{proposal_name}] 최종 종합 평가 보고서")
        print(f"{'='*70}")
        print(final_comprehensive_report.raw)
        print(f"{'='*70}\n")


# =================================================================
# 프로그램 실행
# =================================================================

if __name__ == '__main__':
    """
    사용 방법:
    1. 디렉토리 구조 준비:
       - ./proposal/        : 제안서 파일들 (.txt, .html)
       - ./RFP/            : RFP 파일 (수협_rfp.pdf)
       - ./internal_data/  : 사내 정보 파일들 (.txt)

    2. Ollama 서버 실행 (별도 터미널):
       $ ollama serve
       $ ollama run llama3.2

    3. 프로그램 실행:
       $ python main_rag.py

    4. 결과 확인:
       - 콘솔에 최종 평가 보고서 출력
       - ChromaDB에 벡터스토어 저장 (./chroma_db_crewai/)
    """
    asyncio.run(main())