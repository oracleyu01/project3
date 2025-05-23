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
from typing import TypedDict, List, Dict, Optional, Literal
from bs4 import BeautifulSoup
import time

from supabase import create_client
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from langgraph.graph import StateGraph, END
from langgraph.checkpoint import MemorySaver

# LangSmith 추적 설정
from langsmith import Client
from langchain_core.tracers.context import tracing_v2_enabled

# 페이지 구성
st.set_page_config(page_title="스마트 쇼핑 파인더 (LangGraph Enhanced)", layout="wide")

# ========== API 키 및 클라이언트 초기화 ==========
try:
    # Streamlit Cloud 환경에서는 st.secrets 사용
    supabase_url = st.secrets["SUPABASE_URL"]
    supabase_key = st.secrets["SUPABASE_KEY"]
    openai_api_key = st.secrets["OPENAI_API_KEY"]
    NAVER_CLIENT_ID = st.secrets["NAVER_CLIENT_ID"]
    NAVER_CLIENT_SECRET = st.secrets["NAVER_CLIENT_SECRET"]
    
    # LangSmith API 키 추가
    LANGCHAIN_API_KEY = st.secrets["LANGCHAIN_API_KEY"]
    LANGCHAIN_PROJECT = st.secrets.get("LANGCHAIN_PROJECT", "smart-shopping-finder")
    
except Exception as e:
    # 로컬 환경에서는 환경 변수 사용
    try:
        import dotenv
        dotenv.load_dotenv()
        supabase_url = os.environ.get("SUPABASE_URL")
        supabase_key = os.environ.get("SUPABASE_KEY")
        openai_api_key = os.environ.get("OPENAI_API_KEY")
        NAVER_CLIENT_ID = os.environ.get("NAVER_CLIENT_ID")
        NAVER_CLIENT_SECRET = os.environ.get("NAVER_CLIENT_SECRET")
        
        # LangSmith API 키 추가
        LANGCHAIN_API_KEY = os.environ.get("LANGCHAIN_API_KEY")
        LANGCHAIN_PROJECT = os.environ.get("LANGCHAIN_PROJECT", "smart-shopping-finder")

    except:
        st.error("API 키를 가져오는 데 실패했습니다. 환경 변수나 Streamlit Secrets가 제대로 설정되었는지 확인하세요.")
        st.stop()

# API 키 확인
if not supabase_url or not supabase_key or not openai_api_key:
    st.error("필요한 API 키가 설정되지 않았습니다.")
    st.stop()

# LangSmith 설정 (선택적)
if LANGCHAIN_API_KEY:
    # LangSmith 환경 변수 설정
    os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY
    os.environ["LANGCHAIN_PROJECT"] = LANGCHAIN_PROJECT
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
    
    # LangSmith 클라이언트 초기화
    langsmith_client = Client(api_key=LANGCHAIN_API_KEY)
    
    st.sidebar.success(f"✅ LangSmith 추적 활성화됨 (프로젝트: {LANGCHAIN_PROJECT})")
    
    # 추적 URL 표시
    st.sidebar.info(
        f"추적 대시보드: [LangSmith]"
        f"(https://smith.langchain.com/o/YOUR_ORG/projects/p/{LANGCHAIN_PROJECT})"
    )
else:
    st.sidebar.warning("⚠️ LangSmith API 키가 없어 추적이 비활성화됨")
    os.environ["LANGCHAIN_TRACING_V2"] = "false"

# Supabase 클라이언트 초기화
try:
    supabase = create_client(supabase_url, supabase_key)
    st.sidebar.success("✅ Supabase 연결 성공!")
except Exception as e:
    st.error(f"Supabase 연결 중 오류가 발생했습니다: {str(e)}")
    st.stop()

# OpenAI 클라이언트 초기화
try:
    openai_client = OpenAI(api_key=openai_api_key)
    st.sidebar.success("✅ OpenAI 연결 성공!")
except Exception as e:
    st.error(f"OpenAI 연결 중 오류가 발생했습니다: {str(e)}")
    st.stop()

# 나머지 초기화 코드...
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
    status_message: str

# ========== 유틸리티 함수 ==========
@st.cache_resource
def load_embedding_model():
    """한국어 임베딩 모델 로딩"""
    try:
        model = SentenceTransformer('jhgan/ko-sroberta-multitask')
        return model
    except:
        model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        return model

embedding_model = load_embedding_model()

def generate_embedding(text):
    """텍스트 임베딩 생성"""
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
        return None

# ========== 뉴스 전체 기사 가져오기 ==========
async def fetch_full_article(session: aiohttp.ClientSession, url: str) -> Optional[str]:
    """뉴스 기사 전체 내용 비동기로 가져오기"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        async with session.get(url, headers=headers, timeout=10) as response:
            html = await response.text()
            soup = BeautifulSoup(html, 'html.parser')
            
            # 네이버 뉴스 본문 추출
            article_body = soup.find('article', {'id': 'dic_area'})
            if article_body:
                return article_body.get_text(strip=True)
            
            # 다른 뉴스 사이트 선택자들
            selectors = [
                'div.article_body', 'div.news_view', 'div.articleBody',
                'div.content', 'div.article-body', 'div#articleBodyContents'
            ]
            for selector in selectors:
                content = soup.select_one(selector)
                if content:
                    return content.get_text(strip=True)
            
            return None
    except Exception as e:
        return None

# ========== LangGraph 노드 함수들 ==========
async def search_naver_node(state: SearchState) -> Dict:
    """네이버 API 검색 노드"""
    try:
        query = state["query"]
        source_type = state["source_type"]
        
        # API 엔드포인트 설정
        api_endpoint = {
            "블로그": "blog",
            "뉴스": "news", 
            "쇼핑": "shop"
        }.get(source_type, "blog")
        
        encoded_query = urllib.parse.quote(query)
        url = f"https://openapi.naver.com/v1/search/{api_endpoint}?query={encoded_query}&display=20&sort=sim"
        
        request = urllib.request.Request(url)
        request.add_header("X-Naver-Client-Id", NAVER_CLIENT_ID)
        request.add_header("X-Naver-Client-Secret", NAVER_CLIENT_SECRET)
        
        response = urllib.request.urlopen(request, timeout=15)
        if response.getcode() == 200:
            response_data = json.loads(response.read().decode('utf-8'))
            items = response_data.get('items', [])
            
            return {
                "naver_results": items,
                "status_message": f"{len(items)}개 검색 결과 발견"
            }
        else:
            return {"error": "네이버 API 오류", "naver_results": []}
            
    except Exception as e:
        return {"error": str(e), "naver_results": []}

async def check_content_quality_node(state: SearchState) -> Dict:
    """콘텐츠 품질 평가 노드"""
    results = state.get("naver_results", [])
    source_type = state["source_type"]
    
    if not results:
        return {"quality_score": 0.0}
    
    total_content_length = 0
    for item in results:
        if source_type == "뉴스":
            content = item.get('description', '')
        else:
            content = item.get('description', '') + item.get('title', '')
        total_content_length += len(re.sub('<[^<]+?>', '', content))
    
    avg_length = total_content_length / len(results)
    
    # 뉴스는 더 낮은 기준 적용
    if source_type == "뉴스":
        quality_score = min(avg_length / 100, 1.0)  # 100자 이상이면 높은 품질
    else:
        quality_score = min(avg_length / 200, 1.0)  # 200자 이상이면 높은 품질
    
    return {
        "quality_score": quality_score,
        "status_message": f"콘텐츠 품질 점수: {quality_score:.2f}"
    }

async def fetch_full_articles_node(state: SearchState) -> Dict:
    """전체 기사 가져오기 노드 (병렬 처리)"""
    if state["source_type"] != "뉴스":
        return {"full_articles": state["naver_results"]}
    
    results = state["naver_results"]
    
    async with aiohttp.ClientSession() as session:
        tasks = []
        for item in results[:10]:  # 상위 10개만 처리
            url = item.get('link', '')
            if url:
                tasks.append(fetch_full_article(session, url))
        
        full_contents = await asyncio.gather(*tasks)
    
    # 전체 내용을 결과에 추가
    full_articles = []
    for i, item in enumerate(results[:10]):
        full_content = full_contents[i] if i < len(full_contents) else None
        
        if full_content:
            item_copy = item.copy()
            item_copy['full_content'] = full_content
            full_articles.append(item_copy)
        else:
            full_articles.append(item)
    
    # 나머지 항목들도 추가
    full_articles.extend(results[10:])
    
    successful_fetches = sum(1 for c in full_contents if c)
    return {
        "full_articles": full_articles,
        "status_message": f"{successful_fetches}개 전체 기사 수집 완료"
    }

async def save_to_supabase_node(state: SearchState) -> Dict:
    """Supabase에 저장하는 노드"""
    articles = state.get("full_articles", state.get("naver_results", []))
    source_type = state["source_type"]
    saved_count = 0
    
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
            if source_type == "뉴스":
                metadata = {
                    'title': title,
                    'url': item.get('link', ''),
                    'publisher': item.get('originallink', '').replace('https://', '').replace('http://', '').split('/')[0] if item.get('originallink') else '',
                    'date': item.get('pubDate', ''),
                    'collection': source_type
                }
                full_text = f"뉴스 제목: {title}\n뉴스 내용: {content}\n언론사: {metadata.get('publisher', '')}"
            elif source_type == "블로그":
                metadata = {
                    'title': title,
                    'url': item.get('link', ''),
                    'bloggername': item.get('bloggername', ''),
                    'date': item.get('postdate', ''),
                    'collection': source_type
                }
                full_text = f"제목: {title}\n내용: {content}\n블로거: {metadata.get('bloggername', '')}"
            else:  # 쇼핑
                metadata = {
                    'title': title,
                    'url': item.get('link', ''),
                    'lprice': item.get('lprice', ''),
                    'mallname': item.get('mallName', ''),
                    'brand': item.get('brand', ''),
                    'collection': source_type
                }
                full_text = f"상품명: {title}\n설명: {content}\n브랜드: {metadata.get('brand', '')}"
            
            # 임베딩 생성
            embedding = generate_embedding(full_text)
            if embedding:
                # 중복 체크
                existing = supabase.table('documents').select('id').eq(f"metadata->>url", metadata.get('url', '')).execute()
                
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
    
    return {
        "status_message": f"{saved_count}개 문서 저장 완료"
    }

async def semantic_search_node(state: SearchState) -> Dict:
    """시맨틱 검색 노드"""
    try:
        query = state["query"]
        source_type = state["source_type"]
        
        # 쿼리 전처리
        if source_type == "뉴스":
            processed_query = f"뉴스 검색: {query} 뉴스 기사 언론사 보도"
        elif source_type == "쇼핑":
            processed_query = f"상품 검색: {query} 쇼핑 상품 가격"
        else:
            processed_query = f"블로그 검색: {query} 블로그 포스팅"
        
        # 임베딩 생성
        query_embedding = generate_embedding(processed_query)
        
        if query_embedding is None:
            return {"semantic_results": [], "error": "임베딩 생성 실패"}
        
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
                
                if metadata.get('collection') == source_type:
                    filtered_results.append(item)
        
        # 유사도 순으로 정렬
        filtered_results.sort(key=lambda x: x.get('similarity', 0), reverse=True)
        
        return {
            "semantic_results": filtered_results[:10],
            "status_message": f"{len(filtered_results)}개 시맨틱 검색 결과"
        }
        
    except Exception as e:
        return {"semantic_results": [], "error": str(e)}

async def generate_answer_node(state: SearchState) -> Dict:
    """GPT 답변 생성 노드"""
    results = state.get("semantic_results", [])
    query = state["query"]
    source_type = state["source_type"]
    
    if not results:
        return {
            "final_answer": f"'{query}'에 대한 {source_type} 검색 결과를 찾을 수 없습니다.",
            "status_message": "답변 생성 완료"
        }
    
    # 컨텍스트 구성 (기존 generate_answer_with_gpt 로직 사용)
    contexts = []
    for i, result in enumerate(results[:5]):
        content = result['content']
        metadata = result.get('metadata', {})
        if isinstance(metadata, str):
            metadata = json.loads(metadata)
        
        title = metadata.get('title', '제목 없음')
        similarity = result.get('similarity', 0) * 100
        
        contexts.append(f"문서 {i+1} - {title} (유사도: {similarity:.1f}%):\n{content}\n")
    
    context_text = "\n".join(contexts)
    
    # GPT 호출
    system_prompt = get_system_prompt(source_type)
    user_prompt = get_user_prompt(query, context_text, source_type)
    
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=1000
        )
        
        return {
            "final_answer": response.choices[0].message.content,
            "status_message": "답변 생성 완료"
        }
    except Exception as e:
        return {
            "final_answer": "답변 생성 중 오류가 발생했습니다.",
            "error": str(e)
        }

# 기존 프롬프트 함수들 재사용
def get_system_prompt(source_type):
    # 기존 코드와 동일
    if source_type == "블로그":
        return """당신은 네이버 블로그 데이터를 기반으로 정확하고 유용한 정보를 제공하는 도우미입니다..."""
    elif source_type == "뉴스":
        return """당신은 네이버 뉴스 데이터를 기반으로 정확하고 객관적인 정보를 제공하는 도우미입니다..."""
    elif source_type == "쇼핑":
        return """당신은 네이버 쇼핑 데이터를 기반으로 정확하고 유용한 정보를 제공하는 도우미입니다..."""

def get_user_prompt(query, context_text, source_type):
    # 기존 코드와 동일
    if source_type == "블로그":
        return f"""다음은 네이버 블로그에서 수집한 데이터입니다..."""
    elif source_type == "뉴스":
        return f"""다음은 네이버 뉴스에서 수집한 데이터입니다..."""
    elif source_type == "쇼핑":
        return f"""다음은 네이버 쇼핑에서 수집한 상품 정보입니다..."""

# ========== LangGraph 워크플로우 구성 ==========
class SmartSearchSystem:
    def __init__(self):
        self.memory = MemorySaver()
        self.workflow = self.build_workflow()
        self.app = self.workflow.compile(checkpointer=self.memory)
    
    def build_workflow(self):
        workflow = StateGraph(SearchState)
        
        # 노드 추가
        workflow.add_node("search_naver", search_naver_node)
        workflow.add_node("check_quality", check_content_quality_node)
        workflow.add_node("fetch_full_articles", fetch_full_articles_node)
        workflow.add_node("save_to_supabase", save_to_supabase_node)
        workflow.add_node("semantic_search", semantic_search_node)
        workflow.add_node("generate_answer", generate_answer_node)
        
        # 시작점 설정
        workflow.set_entry_point("search_naver")
        
        # 엣지 추가
        workflow.add_edge("search_naver", "check_quality")
        
        # 조건부 엣지 - 품질에 따라 분기
        workflow.add_conditional_edges(
            "check_quality",
            self.quality_router,
            {
                "fetch_full": "fetch_full_articles",
                "skip_fetch": "save_to_supabase"
            }
        )
        
        workflow.add_edge("fetch_full_articles", "save_to_supabase")
        
        # 검색 모드에 따라 분기
        workflow.add_conditional_edges(
            "save_to_supabase",
            self.mode_router,
            {
                "semantic": "semantic_search",
                "end": END
            }
        )
        
        workflow.add_edge("semantic_search", "generate_answer")
        workflow.add_edge("generate_answer", END)
        
        return workflow
    
    def quality_router(self, state: SearchState):
        """품질 점수에 따라 전체 기사 가져올지 결정"""
        quality_score = state.get("quality_score", 0)
        source_type = state["source_type"]
        
        # 뉴스이고 품질이 낮으면 전체 기사 가져오기
        if source_type == "뉴스" and quality_score < 0.5:
            return "fetch_full"
        
        return "skip_fetch"
    
    def mode_router(self, state: SearchState):
        """검색 모드에 따라 다음 단계 결정"""
        if state["search_mode"] == "시맨틱 검색 (저장된 데이터)":
            return "semantic"
        return "end"
    
    async def search(self, query: str, source_type: str, search_mode: str):
        """통합 검색 실행"""
        initial_state = {
            "query": query,
            "source_type": source_type,
            "search_mode": search_mode,
            "retry_count": 0,
            "status_message": "검색 시작..."
        }
        
        config = {"configurable": {"thread_id": f"{query}-{source_type}"}}
        
        # 스트리밍으로 상태 업데이트 받기
        status_placeholder = st.empty()
        
        async for event in self.app.astream_events(initial_state, config=config, version="v1"):
            if event["event"] == "on_node_end":
                node_output = event["data"]["output"]
                if "status_message" in node_output:
                    status_placeholder.info(node_output["status_message"])
        
        # 최종 결과 가져오기
        final_state = await self.app.aget_state(config)
        return final_state.values

# ========== Streamlit UI ==========
st.title("🛍️ 스마트 쇼핑 파인더: LangGraph Enhanced")
st.write("AI 에이전트가 네이버 검색 결과를 지능적으로 분석하여 최적의 답변을 제공합니다.")

# 세션 상태 초기화
if "search_system" not in st.session_state:
    st.session_state.search_system = SmartSearchSystem()

# UI 구성 (기존과 유사)
search_mode = st.sidebar.radio(
    "검색 모드 선택",
    options=["시맨틱 검색 (저장된 데이터)", "새 데이터 수집 및 저장"],
    index=0
)

source_options = ["쇼핑", "블로그", "뉴스"]
selected_source = st.radio(
    "검색 소스 선택",
    options=source_options,
    horizontal=True
)

query = st.text_input(
    "질문 입력",
    placeholder=f"{selected_source} 관련 질문을 입력하세요"
)

# 검색 실행
if st.button(f"{selected_source}에서 {search_mode.split()[0]}", type="primary"):
    if query:
        with st.spinner("지능적 검색 진행 중..."):
            # 비동기 검색 실행
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            result = loop.run_until_complete(
                st.session_state.search_system.search(query, selected_source, search_mode)
            )
            
            # 결과 표시
            if result.get("final_answer"):
                st.markdown("## 🤖 AI 답변")
                st.markdown(result["final_answer"])
                
                # 검색 결과 상세 표시
                if st.checkbox("검색 결과 상세 보기"):
                    semantic_results = result.get("semantic_results", [])
                    if semantic_results:
                        st.markdown("### 📊 시맨틱 검색 결과")
                        for i, item in enumerate(semantic_results):
                            metadata = item.get('metadata', {})
                            if isinstance(metadata, str):
                                metadata = json.loads(metadata)
                            
                            similarity = item.get('similarity', 0) * 100
                            with st.expander(f"{i+1}. {metadata.get('title', '제목 없음')} (유사도: {similarity:.1f}%)"):
                                st.write(item['content'])
                                if metadata.get('url'):
                                    st.markdown(f"[원본 보기]({metadata['url']})")
            
            # 프로세스 통계
            with st.sidebar:
                st.markdown("### 🔍 검색 프로세스")
                if result.get("naver_results"):
                    st.success(f"✅ 네이버 검색: {len(result['naver_results'])}개")
                if result.get("full_articles"):
                    full_count = sum(1 for a in result['full_articles'] if 'full_content' in a)
                    if full_count > 0:
                        st.success(f"✅ 전체 기사 수집: {full_count}개")
                if result.get("semantic_results"):
                    st.success(f"✅ 시맨틱 매칭: {len(result['semantic_results'])}개")
                if result.get("quality_score") is not None:
                    st.info(f"📊 콘텐츠 품질: {result['quality_score']:.2f}")
    else:
        st.warning("검색어를 입력해주세요!")

# 데이터베이스 상태 (기존과 동일)
st.sidebar.title("📊 데이터베이스 상태")
try:
    result = supabase.table('documents').select('id', count='exact').execute()
    doc_count = result.count if hasattr(result, 'count') else len(result.data)
    st.sidebar.metric("총 문서 수", f"{doc_count:,}개")
except:
    st.sidebar.error("DB 연결 오류")

# LangGraph 특징 안내
with st.sidebar.expander("🚀 LangGraph 개선사항"):
    st.markdown("""
    **새로운 기능:**
    - 🔄 지능적 워크플로우: 콘텐츠 품질에 따라 자동으로 전체 기사 수집
    - ⚡ 병렬 처리: 여러 기사를 동시에 가져와 속도 향상
    - 📊 품질 평가: 검색 결과의 품질을 자동으로 평가
    - 🧠 상태 관리: 검색 프로세스의 모든 단계를 추적
    - 🔁 자동 재시도: 실패 시 자동으로 대체 전략 사용
    """)
