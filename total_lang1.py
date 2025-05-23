import streamlit as st
import os
import json
import numpy as np
import urllib.request
import urllib.parse
import re
import pandas as pd
from datetime import datetime
import asyncio
import aiohttp
from typing import TypedDict, List, Dict, Optional, Literal, Annotated
from bs4 import BeautifulSoup
import time
import threading
from concurrent.futures import ThreadPoolExecutor
import operator

from supabase import create_client
from openai import OpenAI
from sentence_transformers import SentenceTransformer

# LangGraph imports
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# LangChain imports for tracing
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# LangSmith 추적 설정
try:
    from langsmith import Client
    from langchain_core.tracers.context import tracing_v2_enabled
    LANGSMITH_AVAILABLE = True
except ImportError:
    LANGSMITH_AVAILABLE = False
    
# 페이지 구성
st.set_page_config(page_title="스마트 쇼핑 파인더 (LangGraph Enhanced)", layout="wide")

# ========== API 키 및 클라이언트 초기화 ==========
def init_clients():
    """클라이언트 초기화"""
    try:
        # Streamlit Cloud 환경에서는 st.secrets 사용
        if hasattr(st, 'secrets') and st.secrets:
            supabase_url = st.secrets["SUPABASE_URL"]
            supabase_key = st.secrets["SUPABASE_KEY"]
            openai_api_key = st.secrets["OPENAI_API_KEY"]
            NAVER_CLIENT_ID = st.secrets["NAVER_CLIENT_ID"]
            NAVER_CLIENT_SECRET = st.secrets["NAVER_CLIENT_SECRET"]
            LANGCHAIN_API_KEY = st.secrets.get("LANGCHAIN_API_KEY")
            LANGCHAIN_PROJECT = st.secrets.get("LANGCHAIN_PROJECT", "smart-shopping-finder")
        else:
            # 로컬 환경에서는 환경 변수 사용
            supabase_url = os.environ.get("SUPABASE_URL")
            supabase_key = os.environ.get("SUPABASE_KEY")
            openai_api_key = os.environ.get("OPENAI_API_KEY")
            NAVER_CLIENT_ID = os.environ.get("NAVER_CLIENT_ID")
            NAVER_CLIENT_SECRET = os.environ.get("NAVER_CLIENT_SECRET")
            LANGCHAIN_API_KEY = os.environ.get("LANGCHAIN_API_KEY")
            LANGCHAIN_PROJECT = os.environ.get("LANGCHAIN_PROJECT", "smart-shopping-finder")

        # API 키 확인
        if not all([supabase_url, supabase_key, openai_api_key, NAVER_CLIENT_ID, NAVER_CLIENT_SECRET]):
            st.error("필요한 API 키가 설정되지 않았습니다.")
            st.stop()

        # LangSmith 설정
        if LANGCHAIN_API_KEY:
            os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY
            os.environ["LANGCHAIN_PROJECT"] = LANGCHAIN_PROJECT
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
            st.sidebar.success(f"✅ LangSmith 추적 활성화됨: {LANGCHAIN_PROJECT}")
        else:
            st.sidebar.warning("⚠️ LangSmith API 키가 없어 추적이 비활성화됨")
            os.environ["LANGCHAIN_TRACING_V2"] = "false"

        # 클라이언트 초기화
        supabase = create_client(supabase_url, supabase_key)
        openai_client = OpenAI(api_key=openai_api_key)
        
        # LangChain ChatOpenAI 초기화 (추적을 위해)
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.3,
            api_key=openai_api_key
        )
        
        return supabase, openai_client, llm, NAVER_CLIENT_ID, NAVER_CLIENT_SECRET
        
    except Exception as e:
        st.error(f"클라이언트 초기화 중 오류가 발생했습니다: {str(e)}")
        st.stop()

# 클라이언트 초기화
supabase, openai_client, llm, NAVER_CLIENT_ID, NAVER_CLIENT_SECRET = init_clients()

# ========== 상태 정의 ==========
class SearchState(TypedDict):
    query: str
    source_type: str
    search_mode: str
    naver_results: List[Dict]
    full_articles: List[Dict]
    embeddings: List[List[float]]
    semantic_results: List[Dict]
    final_answer: str
    error: Optional[str]
    retry_count: int
    quality_score: float
    status_messages: Annotated[List[str], operator.add]

# ========== 유틸리티 함수 ==========
@st.cache_resource
def load_embedding_model():
    """한국어 임베딩 모델 로딩"""
    try:
        model = SentenceTransformer('jhgan/ko-sroberta-multitask')
        return model
    except:
        try:
            model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            return model
        except Exception as e:
            st.error(f"임베딩 모델 로딩 실패: {str(e)}")
            return None

embedding_model = load_embedding_model()

def generate_embedding(text):
    """텍스트 임베딩 생성"""
    if not embedding_model:
        return None
        
    try:
        if not text or len(text.strip()) < 10:
            return None
        
        cleaned_text = re.sub(r'\s+', ' ', text.strip())
        cleaned_text = re.sub(r'[^\w\s가-힣\.]', ' ', cleaned_text)
        
        if len(cleaned_text) > 512:
            cleaned_text = cleaned_text[:512]
        
        embedding = embedding_model.encode(cleaned_text, convert_to_tensor=False)
        embedding_list = embedding.tolist() if hasattr(embedding, 'tolist') else embedding
        
        # 768차원을 1536차원으로 패딩
        if len(embedding_list) == 768:
            return embedding_list + [0.0] * (1536 - 768)
        elif len(embedding_list) == 1536:
            return embedding_list
        else:
            if len(embedding_list) < 1536:
                return embedding_list + [0.0] * (1536 - len(embedding_list))
            else:
                return embedding_list[:1536]
    except Exception as e:
        st.error(f"임베딩 생성 오류: {str(e)}")
        return None

# ========== LangGraph 노드 함수들 ==========
def search_naver_node(state: SearchState) -> SearchState:
    """네이버 API 검색 노드"""
    try:
        # API 엔드포인트 설정
        api_endpoint = {
            "블로그": "blog",
            "뉴스": "news", 
            "쇼핑": "shop"
        }.get(state["source_type"], "blog")
        
        encoded_query = urllib.parse.quote(state["query"])
        url = f"https://openapi.naver.com/v1/search/{api_endpoint}?query={encoded_query}&display=20&sort=sim"
        
        request = urllib.request.Request(url)
        request.add_header("X-Naver-Client-Id", NAVER_CLIENT_ID)
        request.add_header("X-Naver-Client-Secret", NAVER_CLIENT_SECRET)
        
        with urllib.request.urlopen(request, timeout=15) as response:
            if response.getcode() == 200:
                response_data = json.loads(response.read().decode('utf-8'))
                items = response_data.get('items', [])
                state["naver_results"] = items
                state["status_messages"] = [f"✅ {len(items)}개의 검색 결과를 찾았습니다."]
            else:
                state["error"] = "네이버 API 오류"
                
    except Exception as e:
        state["error"] = str(e)
        
    return state

def fetch_full_articles_node(state: SearchState) -> SearchState:
    """전체 기사 가져오기 노드"""
    if state["source_type"] != "뉴스" or state["search_mode"] != "새 데이터 수집 및 저장":
        state["full_articles"] = state["naver_results"]
        return state
    
    def fetch_article(item):
        url = item.get('link', '')
        if url:
            try:
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
                request = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(request, timeout=10) as response:
                    html = response.read().decode('utf-8')
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    # 네이버 뉴스 본문 추출
                    article_body = soup.find('article', {'id': 'dic_area'})
                    if article_body:
                        item_copy = item.copy()
                        item_copy['full_content'] = article_body.get_text(strip=True)
                        return item_copy
            except:
                pass
        return item
    
    # ThreadPoolExecutor를 사용하여 병렬 처리
    with ThreadPoolExecutor(max_workers=5) as executor:
        full_articles = list(executor.map(fetch_article, state["naver_results"][:10]))
    
    # 나머지 항목들도 추가
    full_articles.extend(state["naver_results"][10:])
    
    state["full_articles"] = full_articles
    full_count = sum(1 for a in full_articles if 'full_content' in a)
    if full_count > 0:
        state["status_messages"] = [f"✅ {full_count}개의 전체 기사를 수집했습니다."]
    
    return state

def save_to_database_node(state: SearchState) -> SearchState:
    """데이터베이스 저장 노드"""
    if state["search_mode"] != "새 데이터 수집 및 저장":
        return state
        
    saved_count = 0
    articles = state.get("full_articles", state["naver_results"])
    
    for item in articles:
        try:
            # HTML 태그 제거
            title = re.sub('<[^<]+?>', '', item.get('title', ''))
            
            # 전체 내용이 있으면 사용, 없으면 description 사용
            if 'full_content' in item and item['full_content']:
                content = item['full_content']
            else:
                content = re.sub('<[^<]+?>', '', item.get('description', ''))
            
            # 메타데이터 구성
            if state["source_type"] == "뉴스":
                metadata = {
                    'title': title,
                    'url': item.get('link', ''),
                    'publisher': item.get('originallink', '').replace('https://', '').replace('http://', '').split('/')[0] if item.get('originallink') else '',
                    'date': item.get('pubDate', ''),
                    'collection': state["source_type"]
                }
                full_text = f"뉴스 제목: {title}\n뉴스 내용: {content}\n언론사: {metadata.get('publisher', '')}"
            elif state["source_type"] == "블로그":
                metadata = {
                    'title': title,
                    'url': item.get('link', ''),
                    'bloggername': item.get('bloggername', ''),
                    'date': item.get('postdate', ''),
                    'collection': state["source_type"]
                }
                full_text = f"제목: {title}\n내용: {content}\n블로거: {metadata.get('bloggername', '')}"
            else:  # 쇼핑
                metadata = {
                    'title': title,
                    'url': item.get('link', ''),
                    'lprice': item.get('lprice', ''),
                    'mallname': item.get('mallName', ''),
                    'brand': item.get('brand', ''),
                    'collection': state["source_type"]
                }
                full_text = f"상품명: {title}\n설명: {content}\n브랜드: {metadata.get('brand', '')}"
            
            # 임베딩 생성
            embedding = generate_embedding(full_text)
            if embedding:
                # 중복 체크
                existing = supabase.table('documents').select('id').eq('metadata->>url', metadata.get('url', '')).execute()
                
                if not existing.data:
                    data = {
                        'content': full_text,
                        'embedding': embedding,
                        'metadata': metadata
                    }
                    supabase.table('documents').insert(data).execute()
                    saved_count += 1
                    
        except Exception as e:
            continue
    
    if saved_count > 0:
        state["status_messages"] = [f"✅ {saved_count}개의 문서를 저장했습니다."]
    
    return state

def semantic_search_node(state: SearchState) -> SearchState:
    """시맨틱 검색 노드"""
    try:
        # 쿼리 전처리
        if state["source_type"] == "뉴스":
            processed_query = f"뉴스 검색: {state['query']} 뉴스 기사 언론사 보도"
        elif state["source_type"] == "쇼핑":
            processed_query = f"상품 검색: {state['query']} 쇼핑 상품 가격"
        else:
            processed_query = f"블로그 검색: {state['query']} 블로그 포스팅"
        
        # 임베딩 생성
        query_embedding = generate_embedding(processed_query)
        
        if query_embedding is None:
            state["semantic_results"] = []
            return state
        
        # 벡터 검색
        response = supabase.rpc(
            'match_documents',
            {
                'query_embedding': query_embedding,
                'match_threshold': 0.3,
                'match_count': 50
            }
        ).execute()
        
        # 소스 타입 필터링
        filtered_results = []
        if response.data:
            for item in response.data:
                metadata = item.get('metadata', {})
                if isinstance(metadata, str):
                    metadata = json.loads(metadata)
                
                if metadata.get('collection') == state["source_type"]:
                    filtered_results.append(item)
        
        # 유사도 순으로 정렬
        filtered_results.sort(key=lambda x: x.get('similarity', 0), reverse=True)
        
        state["semantic_results"] = filtered_results[:10]
        state["status_messages"] = [f"✅ {len(state['semantic_results'])}개의 관련 문서를 찾았습니다."]
        
    except Exception as e:
        state["error"] = f"시맨틱 검색 오류: {str(e)}"
        state["semantic_results"] = []
        
    return state

def generate_answer_node(state: SearchState) -> SearchState:
    """AI 답변 생성 노드 (LangChain 사용)"""
    if not state["semantic_results"]:
        state["final_answer"] = f"'{state['query']}'에 대한 {state['source_type']} 검색 결과를 찾을 수 없습니다."
        return state
    
    # 컨텍스트 구성
    contexts = []
    for i, result in enumerate(state["semantic_results"][:5]):
        content = result['content']
        metadata = result.get('metadata', {})
        if isinstance(metadata, str):
            metadata = json.loads(metadata)
        
        title = metadata.get('title', '제목 없음')
        similarity = result.get('similarity', 0) * 100
        
        contexts.append(f"문서 {i+1} - {title} (유사도: {similarity:.1f}%):\n{content}\n")
    
    context_text = "\n".join(contexts)
    
    # 시스템 프롬프트
    system_prompts = {
        "블로그": """당신은 네이버 블로그 데이터를 기반으로 정확하고 유용한 정보를 제공하는 도우미입니다.
사용자의 질문에 대해 검색된 블로그 내용을 바탕으로 친근하고 실용적인 답변을 제공해주세요.
답변은 한국어로 작성하고, 구체적인 팁이나 경험담을 포함해 주세요.""",
        
        "뉴스": """당신은 네이버 뉴스 데이터를 기반으로 정확하고 객관적인 정보를 제공하는 도우미입니다.
사용자의 질문에 대해 검색된 뉴스 내용을 바탕으로 사실적이고 균형잡힌 답변을 제공해주세요.
답변은 한국어로 작성하고, 최신 동향과 다양한 관점을 포함해 주세요.""",
        
        "쇼핑": """당신은 네이버 쇼핑 데이터를 기반으로 정확하고 유용한 정보를 제공하는 도우미입니다.
사용자의 질문에 대해 검색된 상품 정보를 바탕으로 실용적인 쇼핑 조언을 제공해주세요.
답변은 한국어로 작성하고, 가격 비교나 제품 특징을 포함해 주세요."""
    }
    
    # LangChain 프롬프트 템플릿 사용
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompts[state["source_type"]]),
        ("human", """다음은 네이버 {source_type}에서 수집한 데이터입니다:

{context}

질문: {query}

위 내용을 바탕으로 질문에 대한 유용한 답변을 작성해주세요.""")
    ])
    
    try:
        # LangChain 체인 실행 (이렇게 하면 자동으로 추적됨)
        chain = prompt | llm | StrOutputParser()
        
        answer = chain.invoke({
            "source_type": state["source_type"],
            "context": context_text,
            "query": state["query"]
        })
        
        state["final_answer"] = answer
        state["status_messages"] = ["✅ AI 답변 생성 완료!"]
        
    except Exception as e:
        state["error"] = f"답변 생성 중 오류가 발생했습니다: {str(e)}"
        state["final_answer"] = "답변 생성에 실패했습니다."
        
    return state

def check_error(state: SearchState) -> str:
    """에러 체크 노드"""
    if state.get("error"):
        return "error"
    return "continue"

# ========== LangGraph 워크플로우 구성 ==========
def create_search_workflow():
    """검색 워크플로우 생성"""
    workflow = StateGraph(SearchState)
    
    # 노드 추가
    workflow.add_node("search_naver", search_naver_node)
    workflow.add_node("fetch_articles", fetch_full_articles_node)
    workflow.add_node("save_database", save_to_database_node)
    workflow.add_node("semantic_search", semantic_search_node)
    workflow.add_node("generate_answer", generate_answer_node)
    
    # 엣지 추가
    workflow.set_entry_point("search_naver")
    
    # 조건부 엣지
    workflow.add_conditional_edges(
        "search_naver",
        check_error,
        {
            "continue": "fetch_articles",
            "error": END
        }
    )
    
    workflow.add_edge("fetch_articles", "save_database")
    workflow.add_edge("save_database", "semantic_search")
    workflow.add_edge("semantic_search", "generate_answer")
    workflow.add_edge("generate_answer", END)
    
    # 체크포인터 추가
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    
    return app

# 워크플로우 생성
search_app = create_search_workflow()

# ========== Streamlit UI ==========
st.title("🛍️ 스마트 쇼핑 파인더: LangGraph Enhanced")
st.write("AI 에이전트가 네이버 검색 결과를 지능적으로 분석하여 최적의 답변을 제공합니다.")

# 사이드바 설정
with st.sidebar:
    st.title("🔧 설정")
    
    search_mode = st.radio(
        "검색 모드 선택",
        options=["시맨틱 검색 (저장된 데이터)", "새 데이터 수집 및 저장"],
        index=0,
        help="저장된 데이터에서 검색하거나 새로운 데이터를 수집합니다."
    )
    
    st.markdown("---")
    
    # 데이터베이스 상태
    st.markdown("### 📊 데이터베이스 상태")
    try:
        result = supabase.table('documents').select('id', count='exact').execute()
        doc_count = result.count if hasattr(result, 'count') else len(result.data)
        st.metric("총 문서 수", f"{doc_count:,}개")
    except Exception as e:
        st.error("DB 연결 오류")

# 메인 콘텐츠
col1, col2 = st.columns([3, 1])

with col1:
    # 검색 소스 선택
    source_options = ["쇼핑", "블로그", "뉴스"]
    selected_source = st.radio(
        "검색 소스 선택",
        options=source_options,
        horizontal=True
    )
    
    # 질문 입력
    query = st.text_input(
        "질문을 입력하세요",
        placeholder=f"{selected_source} 관련 질문을 입력하세요 (예: 아이폰 15 장점과 단점)",
        help=f"{selected_source}에서 검색할 내용을 자세히 입력해주세요."
    )

with col2:
    st.markdown("### 🎯 검색 팁")
    st.markdown("""
    - 구체적인 키워드 사용
    - 브랜드명이나 모델명 포함
    - 가격대나 조건 명시
    """)

# 검색 실행
if st.button(f"🔍 {selected_source}에서 검색하기", type="primary", use_container_width=True):
    if query.strip():
        # 진행 상황 표시
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # 초기 상태 설정
            initial_state = {
                "query": query,
                "source_type": selected_source,
                "search_mode": search_mode,
                "naver_results": [],
                "full_articles": [],
                "embeddings": [],
                "semantic_results": [],
                "final_answer": "",
                "error": None,
                "retry_count": 0,
                "quality_score": 0.0,
                "status_messages": []
            }
            
            # LangGraph 워크플로우 실행
            config = {"configurable": {"thread_id": f"{query}_{selected_source}_{datetime.now().isoformat()}"}}
            
            # 스트리밍으로 실행하여 각 단계별로 업데이트 받기
            step_count = 0
            total_steps = 5
            
            for event in search_app.stream(initial_state, config):
                step_count += 1
                progress = int((step_count / total_steps) * 100)
                progress_bar.progress(progress)
                
                # 각 노드의 결과에서 상태 메시지 추출
                for node_name, node_state in event.items():
                    if node_state.get("status_messages"):
                        for msg in node_state["status_messages"]:
                            status_text.info(msg)
                            time.sleep(0.5)  # 사용자가 메시지를 볼 수 있도록
                    
                    # 최종 상태 업데이트
                    if "final_answer" in node_state:
                        final_state = node_state
            
            # 에러 체크
            if final_state.get("error"):
                st.error(f"검색 중 오류가 발생했습니다: {final_state['error']}")
                st.stop()
            
            # 결과 표시
            st.markdown("---")
            st.markdown("## 🤖 AI 답변")
            st.markdown(final_state.get("final_answer", "답변을 생성할 수 없습니다."))
            
            # 상세 결과 표시
            if final_state.get("semantic_results"):
                with st.expander("📊 검색 결과 상세 보기", expanded=False):
                    st.markdown(f"### {len(final_state['semantic_results'])}개의 관련 문서를 찾았습니다")
                    
                    for i, item in enumerate(final_state["semantic_results"]):
                        metadata = item.get('metadata', {})
                        if isinstance(metadata, str):
                            metadata = json.loads(metadata)
                        
                        similarity = item.get('similarity', 0) * 100
                        title = metadata.get('title', '제목 없음')
                        
                        with st.container():
                            st.markdown(f"**{i+1}. {title}** (유사도: {similarity:.1f}%)")
                            
                            # 메타데이터 표시
                            col1, col2 = st.columns(2)
                            with col1:
                                if metadata.get('url'):
                                    st.markdown(f"🔗 [원본 보기]({metadata['url']})")
                            with col2:
                                if selected_source == "뉴스" and metadata.get('publisher'):
                                    st.markdown(f"📰 {metadata['publisher']}")
                                elif selected_source == "블로그" and metadata.get('bloggername'):
                                    st.markdown(f"✍️ {metadata['bloggername']}")
                                elif selected_source == "쇼핑" and metadata.get('brand'):
                                    st.markdown(f"🏷️ {metadata['brand']}")
                            
                            # 내용 미리보기
                            content_preview = item['content'][:200] + "..." if len(item['content']) > 200 else item['content']
                            st.markdown(f"> {content_preview}")
                            st.markdown("---")
            
        except Exception as e:
            st.error(f"검색 중 오류가 발생했습니다: {str(e)}")
        
        finally:
            # 진행 상황 표시 제거
            progress_bar.empty()
            status_text.empty()
    else:
        st.warning("검색어를 입력해주세요!")

# 하단 정보
st.markdown("---")
with st.expander("🚀 LangGraph 개선사항", expanded=False):
    st.markdown("""
    **새로운 기능:**
    - 🔄 **지능적 워크플로우**: LangGraph로 구조화된 검색 프로세스
    - 📊 **LangSmith 추적**: 모든 단계가 자동으로 추적되어 디버깅 용이
    - ⚡ **병렬 처리**: 여러 기사를 동시에 가져와 속도 향상
    - 🧠 **상태 관리**: 검색 프로세스의 모든 단계를 추적
    - 🔁 **에러 핸들링**: 각 단계별 에러 처리 및 복구
    - 💾 **캐싱 최적화**: 임베딩 모델과 검색 결과 캐싱으로 성능 향상
    
    **LangSmith에서 확인 가능한 정보:**
    - 🎯 각 노드의 실행 시간과 성능
    - 📝 입력/출력 데이터 추적
    - 🔍 프롬프트와 LLM 응답 내용
    - ⚠️ 에러 발생 시 상세 로그
    - 📈 전체 워크플로우 시각화
    
    **사용 방법:**
    1. LangSmith에서 프로젝트명 'smart-shopping-finder' 확인
    2. 각 검색 요청마다 새로운 thread_id로 추적
    3. 워크플로우의 각 단계별 실행 결과 확인 가능
    """)
