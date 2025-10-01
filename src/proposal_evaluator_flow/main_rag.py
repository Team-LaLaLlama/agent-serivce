# main.py

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
PROPOSAL_DIR = "./proposal"
RFP_PATH = "./RFP/수협_rfp.pdf"

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
    """문서 로드 (JSON/PDF/TXT 지원)"""
    documents = []
    print(f"  - [{doc_type}] '{os.path.basename(file_path)}' 로딩 중...")
    
    # ... (RAG 노트북의 load_document 함수 내용과 동일) ...
    if file_path.endswith('.pdf'):
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        for doc in docs:
            doc.metadata.update({"doc_type": doc_type, "proposal_name": proposal_name})
        documents.extend(docs)
    elif file_path.endswith('.txt'):
        loader = TextLoader(file_path, encoding='utf-8')
        docs = loader.load()
        for doc in docs:
            doc.metadata.update({"doc_type": doc_type, "proposal_name": proposal_name})
        documents.extend(docs)
    # JSON 로더가 필요한 경우 여기에 추가
    
    print(f"    → {len(documents)}개 섹션 로드됨")
    return documents

def create_unified_vectorstore(proposal_files, rfp_path, embedding_model, chroma_client, collection_name="proposal_evaluation_store"):
    """모든 제안서와 RFP 문서를 단일 벡터 스토어로 변환"""
    print(f"\n--- [RAG Setup] 통합 벡터스토어 생성을 시작합니다 (Collection: {collection_name}) ---")
    
    all_documents = []
    all_documents.extend(load_document(rfp_path, "RFP", "RFP"))
    for proposal_path in proposal_files:
        proposal_name = os.path.basename(proposal_path)
        all_documents.extend(load_document(proposal_path, "제안서", proposal_name))
        
    print(f"\n  ✓ 총 {len(all_documents)}개 문서 섹션 로드 완료")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = text_splitter.split_documents(all_documents)
    print(f"  ✓ {len(splits)}개 청크로 분할 완료")

    try:
        chroma_client.delete_collection(name=collection_name)
        print(f"  ✓ 기존 '{collection_name}' 컬렉션 삭제 완료")
    except Exception:
        pass

    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embedding_model,
        collection_name=collection_name,
        client=chroma_client
    )
    print("  ✓ 통합 벡터스토어 생성 완료!")
    return vectorstore

def get_context_for_topic(proposal_file, topic):
    """
    (수정된 함수) 실제 벡터스토어에서 관련 내용을 검색합니다.
    """
    global unified_vectorstore
    if unified_vectorstore is None:
        return "오류: 벡터스토어가 초기화되지 않았습니다."

    print(f"INFO: RAG 검색 실행 -> 제안서: '{proposal_file}', 토픽: '{topic}'")
    
    # proposal_file 이름을 메타데이터 필터로 사용하여 검색
    results = unified_vectorstore.similarity_search(
        query=topic,
        k=2, # 관련성이 높은 2개의 청크를 가져옴
        filter={"proposal_name": proposal_file}
    )
    
    if not results:
        return "관련 내용을 찾을 수 없습니다."
        
    # 검색된 결과들을 하나의 문자열로 합침
    context = "\n\n---\n\n".join([doc.page_content for doc in results])
    return context
#==============================================================


# 2. CrewAI Agent 및 프로세스 정의
# =================================================================

async def main():
    print("## 동적 Agent 생성 및 평가 프로세스를 시작합니다.")
    
    # --- [전제] RAG 파이프라인 초기화 ---
    # 메인 프로세스 시작 전, 벡터스토어를 먼저 준비합니다.
    global unified_vectorstore
    proposal_files = glob.glob(os.path.join(PROPOSAL_DIR, "*.txt")) # .txt 제안서만 대상으로 함
    
    if not proposal_files or not os.path.exists(RFP_PATH):
        print("오류: 제안서 또는 RFP 파일이 없습니다. 경로를 확인하세요.")
        return
        
    embedding_model, chroma_client = initialize_rag_components()
    unified_vectorstore = create_unified_vectorstore(
        proposal_files, RFP_PATH, embedding_model, chroma_client
    )
    # --- RAG 초기화 완료 ---

    # 전체 심사 항목 리스트
    unstructured_evaluation_items = [
        {"대분류": "기술", "topic": "시스템 아키텍처", "criteria": "MSA 기반의 유연하고 확장 가능한 아키텍처인가?"},
        {"대분류": "관리", "topic": "프로젝트 관리 방안", "criteria": "WBS 기반의 상세하고 실현 가능한 일정을 제시하였는가?"},
        {"대분류": "기술", "topic": "데이터베이스 암호화", "criteria": "개인정보보호 및 데이터 암호화 방안이 명시되었는가?"},
        {"대분류": "관리", "topic": "투입 인력 계획", "criteria": "투입 인력의 역할과 경력이 적절한가?"},
        {"대분류": "가격", "topic": "비용 산정 내역", "criteria": "제시된 비용이 합리적이고 구체적인 근거를 포함하는가?"},
    ]

    # --- LLM 정의 ---
    llm = LLM(model="ollama/llama3.2", base_url="http://localhost:11434")

    # =================================================================
    # Phase 1: Dispatcher가 대분류를 스스로 찾아내고 항목 분류
    # =================================================================
    print("\n--- [Phase 1] Dispatcher Agent가 대분류를 식별하고 항목을 분류합니다 ---")
    
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
    print("\n--- [Phase 2] 발견된 대분류별로 전문가 Agent를 동적으로 생성하여 병렬 평가합니다 ---")
    
    # 모든 제안서 파일에 대해 평가를 반복합니다.
    for proposal_path in proposal_files:
        proposal_name = os.path.basename(proposal_path)
        print(f"\n\n{'='*20} [{proposal_name}] 평가 시작 {'='*20}")

        specialist_agents = []
        evaluation_tasks = []

        for category, items in categorized_items.items():
            specialist_agent = Agent(
                role=f"'{category}' 부문 전문 평가관",
                goal=f"'{proposal_name}' 제안서의 '{category}' 부문을 전문적으로 평가",
                backstory=f"당신은 '{category}' 분야 최고의 전문가로서, 주어진 관련 내용을 바탕으로 심사 기준에 따라 제안서를 냉철하게 분석하고 평가 보고서를 작성해야 합니다.",
                llm=llm,
                verbose=True
            )
            specialist_agents.append(specialist_agent)

            for item in items:
                # 수정된 RAG 함수 호출
                context = get_context_for_topic(proposal_name, item['topic'])
                
                task = Task(
                    description=f"'{proposal_name}' 제안서의 '{category}' 부문 중 '{item['topic']}' 항목을 평가하시오.\n\n- 심사 기준: {item['criteria']}\n\n- 제안서 관련 내용:\n---\n{context}\n---\n\n위 내용을 근거로 평가를 수행하고 보고서를 작성하시오.",
                    expected_output=f"'{item['topic']}' 항목에 대한 평가 보고서. 반드시 [평가 점수(1-100)], [평가 요약], [판단 근거] 세 가지 항목을 포함해야 합니다.",
                    agent=specialist_agent
                )
                evaluation_tasks.append(task)
        
        if not evaluation_tasks:
            print("평가할 작업이 없습니다.")
            continue

        evaluation_crew = Crew(
            agents=specialist_agents,
            tasks=evaluation_tasks,
            verbose=True
        )
        final_results = await evaluation_crew.kickoff_async()

        print(f"\n--- [Phase 3] [{proposal_name}] 최종 보고서를 작성합니다 ---")
        individual_reports = "\n\n".join([str(result) for result in final_results])

        reporting_agent = Agent(
            role="수석 평가 분석가",
            goal="여러 개의 개별 평가 보고서를 종합하여, 경영진이 의사결정을 내릴 수 있도록 하나의 완성된 최종 보고서를 작성",
            backstory="당신은 여러 부서의 보고를 취합하여 핵심만 요약하고, 전체적인 관점에서 강점과 약점을 분석하여 최종 보고서를 작성하는 데 매우 능숙합니다.",
            llm=llm, verbose=True
        )
        reporting_task = Task(
            description=f"아래는 '{proposal_name}' 제안서에 대한 각 분야 전문가들의 개별 평가 보고서입니다.\n\n[개별 평가 보고서 목록]\n{individual_reports}\n\n위 보고서들을 모두 종합하여, 제안서 전체에 대한 최종 평가 보고서를 작성해주세요. 보고서는 [총평], [주요 강점], [주요 약점], [최종 추천 점수(1-100)] 항목을 포함해야 합니다.",
            expected_output="하나의 완성된 최종 평가 보고서.",
            agent=reporting_agent
        )
        reporting_crew = Crew(agents=[reporting_agent], tasks=[reporting_task], verbose=False)
        final_comprehensive_report = reporting_crew.kickoff()

        print(f"\n\n🚀 [{proposal_name}] 최종 종합 평가 보고서\n==========================================")
        print(final_comprehensive_report.raw)
        print("==========================================\n")


if __name__ == '__main__':
    asyncio.run(main())