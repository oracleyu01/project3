import streamlit as st
import os
import json
import numpy as np
import urllib.request
import urllib.parse
import re
import pandas as pd
from datetime import datetime
from supabase import create_client
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import torch
import time

# 페이지 구성
st.set_page_config(page_title="네이버 통합 검색", layout="wide")

# 네이버 API 클라이언트 ID와 시크릿
NAVER_CLIENT_ID = "9XhhxLV1IzDpTZagoBr1"
NAVER_CLIENT_SECRET = "J14HFxv3B6"

# Streamlit에서 실행 중인지 확인하고 secrets 가져오기
try:
    # Streamlit Cloud 환경에서는 st.secrets 사용
    supabase_url = st.secrets["SUPABASE_URL"]
    supabase_key = st.secrets["SUPABASE_KEY"]
    openai_api_key = st.secrets["OPENAI_API_KEY"]
except Exception as e:
    # 로컬 환경에서는 환경 변수 사용
    try:
        import dotenv
        dotenv.load_dotenv()
        supabase_url = os.environ.get("SUPABASE_URL")
        supabase_key = os.environ.get("SUPABASE_KEY")
        openai_api_key = os.environ.get("OPENAI_API_KEY")
    except:
        st.error("API 키를 가져오는 데 실패했습니다. 환경 변수나 Streamlit Secrets가 제대로 설정되었는지 확인하세요.")
        st.stop()

# API 키 확인
if not supabase_url or not supabase_key or not openai_api_key:
    st.error("필요한 API 키가 설정되지 않았습니다. (OpenAI는 답변 생성용으로만 사용됩니다)")
    st.stop()

# Supabase 클라이언트 초기화
try:
    supabase = create_client(supabase_url, supabase_key)
    st.sidebar.success("Supabase 연결 성공!")
except Exception as e:
    st.error(f"Supabase 연결 중 오류가 발생했습니다: {str(e)}")
    st.stop()

# OpenAI 클라이언트 초기화 (GPT 답변 생성용)
try:
    openai_client = OpenAI(api_key=openai_api_key)
    st.sidebar.success("OpenAI 연결 성공!")
except Exception as e:
    st.error(f"OpenAI 연결 중 오류가 발생했습니다: {str(e)}")
    st.stop()

# 무료 임베딩 모델 초기화
@st.cache_resource
def load_embedding_model():
    """한국어 임베딩 모델 로딩 (캐시됨)"""
    try:
        # 한국어 성능이 좋은 무료 모델
        model = SentenceTransformer('jhgan/ko-sroberta-multitask')
        st.sidebar.success("임베딩 모델 로딩 성공!")
        return model
    except Exception as e:
        st.sidebar.error(f"임베딩 모델 로딩 실패: {str(e)}")
        # 백업 모델 사용
        try:
            model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            st.sidebar.warning("백업 임베딩 모델 사용 중")
            return model
        except Exception as e2:
            st.error(f"백업 모델도 로딩 실패: {str(e2)}")
            st.stop()

# 임베딩 모델 로딩
embedding_model = load_embedding_model()

def test_naver_api():
    """네이버 API 연결 테스트"""
    try:
        test_query = "테스트"
        encoded_query = urllib.parse.quote(test_query)
        url = f"https://openapi.naver.com/v1/search/blog?query={encoded_query}&display=1"
        
        request = urllib.request.Request(url)
        request.add_header("X-Naver-Client-Id", NAVER_CLIENT_ID)
        request.add_header("X-Naver-Client-Secret", NAVER_CLIENT_SECRET)
        request.add_header("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
        
        response = urllib.request.urlopen(request, timeout=10)
        return response.getcode() == 200
    except Exception as e:
        st.sidebar.error(f"네이버 API 테스트 실패: {str(e)}")
        return False

# 네이버 API 상태 확인
if test_naver_api():
    st.sidebar.success("네이버 API 연결 성공!")
else:
    st.sidebar.error("네이버 API 연결 실패!")

def generate_embedding(text):
    """텍스트에서 무료 임베딩 생성 - 개선된 버전"""
    try:
        # 텍스트 전처리 추가
        if not text or len(text.strip()) < 10:  # 너무 짧은 텍스트 제외
            return None
        
        # 텍스트 정규화 강화
        cleaned_text = re.sub(r'\s+', ' ', text.strip())  # 공백 정규화
        cleaned_text = re.sub(r'[^\w\s가-힣\.]', ' ', cleaned_text)  # 특수문자 제거 (마침표는 유지)
        cleaned_text = ' '.join(cleaned_text.split())  # 중복 공백 제거
        
        # 너무 긴 텍스트는 잘라내기 (모델 제한 고려)
        if len(cleaned_text) > 512:  # sentence-transformers 일반적 제한
            cleaned_text = cleaned_text[:512]
        
        # 무료 임베딩 모델로 임베딩 생성
        embedding = embedding_model.encode(cleaned_text, convert_to_tensor=False)
        
        # numpy array를 list로 변환
        if hasattr(embedding, 'tolist'):
            embedding_list = embedding.tolist()
        else:
            embedding_list = embedding
        
        # 768차원을 1536차원으로 패딩 (0으로 채움)
        if len(embedding_list) == 768:
            # 0으로 패딩하여 1536차원으로 만들기
            padded_embedding = embedding_list + [0.0] * (1536 - 768)
            return padded_embedding
        elif len(embedding_list) == 1536:
            return embedding_list
        else:
            st.warning(f"예상치 못한 임베딩 차원: {len(embedding_list)}")
            # 1536차원으로 맞추기
            if len(embedding_list) < 1536:
                return embedding_list + [0.0] * (1536 - len(embedding_list))
            else:
                return embedding_list[:1536]
            
    except Exception as e:
        st.error(f"임베딩 생성 중 오류 발생: {str(e)}")
        raise

def search_naver_api(query, source_type, count=20):
    """네이버 API를 사용하여 검색하고 결과를 Supabase에 저장 - 개선된 버전"""
    try:
        # 소스 타입에 따른 API 엔드포인트 설정
        if source_type == "블로그":
            api_endpoint = "blog"
        elif source_type == "뉴스":
            api_endpoint = "news"
        elif source_type == "쇼핑":
            api_endpoint = "shop"
        else:
            api_endpoint = "blog"  # 기본값
        
        # 쿼리 인코딩
        encoded_query = urllib.parse.quote(query)
        url = f"https://openapi.naver.com/v1/search/{api_endpoint}?query={encoded_query}&display={count}&sort=sim"
        
        # 요청 헤더 설정 (개선)
        request = urllib.request.Request(url)
        request.add_header("X-Naver-Client-Id", NAVER_CLIENT_ID)
        request.add_header("X-Naver-Client-Secret", NAVER_CLIENT_SECRET)
        request.add_header("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
        
        # API 요청 및 응답 처리 (개선된 예외 처리)
        try:
            response = urllib.request.urlopen(request, timeout=15)
            response_code = response.getcode()
            
            if response_code == 200:
                # 응답 읽기
                response_body = response.read()
                
                # 응답이 비어있는지 확인
                if not response_body:
                    st.error("네이버 API에서 빈 응답을 받았습니다.")
                    return [], 0, 0
                
                # 디코딩 및 JSON 파싱 (개선된 오류 처리)
                try:
                    response_text = response_body.decode('utf-8')
                    
                    # 디버깅: 응답 내용의 시작 부분 확인
                    if not response_text.strip().startswith('{'):
                        st.error(f"유효하지 않은 JSON 응답: {response_text[:200]}...")
                        return [], 0, 0
                    
                    response_data = json.loads(response_text)
                    
                except json.JSONDecodeError as e:
                    st.error(f"JSON 파싱 오류: {str(e)}")
                    st.error(f"응답 내용 미리보기: {response_text[:200] if 'response_text' in locals() else '디코딩 실패'}...")
                    return [], 0, 0
                except UnicodeDecodeError as e:
                    st.error(f"응답 디코딩 오류: {str(e)}")
                    return [], 0, 0
                
                # 응답 데이터 확인
                if 'items' not in response_data:
                    st.warning("검색 결과가 없거나 응답 형식이 올바르지 않습니다.")
                    return [], 0, 0
                
                # 결과 처리 및 Supabase에 저장
                saved_count = 0
                items = response_data.get('items', [])
                
                for i, item in enumerate(items):
                    try:
                        # HTML 태그 제거
                        title = re.sub('<[^<]+?>', '', item.get('title', '')) if item.get('title') else '제목 없음'
                        
                        # 소스 타입에 따른 내용 필드 추출 및 개선된 텍스트 구성
                        if source_type == "블로그":
                            content = re.sub('<[^<]+?>', '', item.get('description', '')) if item.get('description') else ''
                            metadata = {
                                'title': title,
                                'url': item.get('link', ''),
                                'bloggername': item.get('bloggername', ''),
                                'date': item.get('postdate', ''),
                                'collection': source_type
                            }
                            # 개선된 전체 텍스트 구성
                            full_text = f"제목: {title}\n내용: {content}\n블로거: {metadata.get('bloggername', '')}\n카테고리: 블로그"
                            
                        elif source_type == "뉴스":
                            content = re.sub('<[^<]+?>', '', item.get('description', '')) if item.get('description') else ''
                            # 간단한 날짜 처리
                            pub_date = item.get('pubDate', '')
                            
                            metadata = {
                                'title': title,
                                'url': item.get('link', ''),
                                'publisher': item.get('originallink', '').replace('https://', '').replace('http://', '').split('/')[0] if item.get('originallink') else '',
                                'date': pub_date,
                                'collection': source_type
                            }
                            # 개선된 전체 텍스트 구성 - 뉴스에 맞게 수정
                            full_text = f"뉴스 제목: {title}\n뉴스 내용: {content}\n언론사: {metadata.get('publisher', '')}\n날짜: {pub_date}\n분류: 뉴스 기사"
                            
                        elif source_type == "쇼핑":
                            content = f"{title}. " + re.sub('<[^<]+?>', '', item.get('category3', '')) if item.get('category3') else title
                            metadata = {
                                'title': title,
                                'url': item.get('link', ''),
                                'lprice': item.get('lprice', ''),
                                'hprice': item.get('hprice', ''),
                                'mallname': item.get('mallName', ''),
                                'maker': item.get('maker', ''),
                                'brand': item.get('brand', ''),
                                'collection': source_type
                            }
                            # 개선된 전체 텍스트 구성
                            full_text = f"상품명: {title}\n설명: {content}\n브랜드: {metadata.get('brand', '')}\n제조사: {metadata.get('maker', '')}\n판매처: {metadata.get('mallname', '')}\n카테고리: 쇼핑"
                        
                        # 빈 텍스트 건너뛰기
                        if not full_text.strip() or len(full_text.strip()) < 20:
                            continue
                        
                        # 뉴스 데이터 디버깅 (임시)
                        if source_type == "뉴스" and i < 3:  # 처음 3개만 디버깅
                            st.sidebar.write(f"**뉴스 저장 디버깅 {i+1}:**")
                            st.sidebar.write(f"- 제목: {title[:50]}...")
                            st.sidebar.write(f"- 언론사: {metadata.get('publisher', 'N/A')}")
                            st.sidebar.write(f"- 텍스트 길이: {len(full_text)}")
                        
                        try:
                            # 임베딩 생성
                            embedding = generate_embedding(full_text)
                            
                            if embedding is None:  # 임베딩 생성 실패 시 건너뛰기
                                continue
                            
                            # Supabase에 데이터 삽입
                            data = {
                                'content': full_text,
                                'embedding': embedding,
                                'metadata': metadata
                            }
                            
                            # 중복 체크 개선 (제목과 URL 기반)
                            check_url = metadata.get('url', '')
                            
                            try:
                                # URL 기반 중복 체크
                                if check_url:
                                    existing = supabase.table('documents').select('id').eq(f"metadata->>url", check_url).execute()
                                    
                                    if not existing.data:  # 중복이 없을 경우에만 삽입
                                        result = supabase.table('documents').insert(data).execute()
                                        saved_count += 1
                                        if source_type == "뉴스":
                                            st.sidebar.success(f"뉴스 저장 성공: {title[:30]}...")
                                else:
                                    # URL이 없으면 그냥 저장
                                    result = supabase.table('documents').insert(data).execute()
                                    saved_count += 1
                            
                            except Exception as e:
                                st.warning(f"항목 {i+1} 저장 중 상세 오류: {str(e)}")
                                continue
                            
                        except Exception as e:
                            st.warning(f"항목 {i+1} 저장 중 오류: {str(e)}")
                            continue
                    
                    except Exception as e:
                        st.warning(f"항목 {i+1} 처리 중 오류: {str(e)}")
                        continue
                
                return items, response_data.get('total', 0), saved_count
            
            else:
                st.error(f"네이버 API HTTP 오류: {response_code}")
                return [], 0, 0
        
        except urllib.error.HTTPError as e:
            st.error(f"네이버 API HTTP 오류: {e.code} - {e.reason}")
            if e.code == 400:
                st.error("잘못된 요청입니다. 검색어를 확인해주세요.")
            elif e.code == 401:
                st.error("인증 오류입니다. API 키를 확인해주세요.")
            elif e.code == 403:
                st.error("접근 거부되었습니다. API 사용 권한을 확인해주세요.")
            elif e.code == 429:
                st.error("API 호출 한도를 초과했습니다. 잠시 후 다시 시도해주세요.")
            return [], 0, 0
            
        except urllib.error.URLError as e:
            st.error(f"네트워크 연결 오류: {str(e)}")
            return [], 0, 0
            
        except Exception as e:
            st.error(f"예상치 못한 오류: {str(e)}")
            return [], 0, 0
            
    except Exception as e:
        st.error(f"네이버 검색 중 전체 오류 발생: {str(e)}")
        return [], 0, 0

def semantic_search(query_text, source_type="블로그", limit=10, match_threshold=0.5):
    """시맨틱 검색 수행 - 개선된 버전"""
    try:
        # 쿼리 전처리를 소스 타입별로 다르게
        if source_type == "뉴스":
            processed_query = f"뉴스 검색: {query_text} 뉴스 기사 언론사 보도"
        elif source_type == "쇼핑":
            processed_query = f"상품 검색: {query_text} 쇼핑 상품 가격"
        else:
            processed_query = f"블로그 검색: {query_text} 블로그 포스팅"
        
        # 쿼리 텍스트에 대한 임베딩 생성
        query_embedding = generate_embedding(processed_query)
        
        if query_embedding is None:
            st.error("쿼리 임베딩 생성에 실패했습니다.")
            return []
        
        # 디버깅: 쿼리 정보 출력
        if source_type == "뉴스":
            st.sidebar.write(f"뉴스 검색 쿼리: {processed_query[:50]}...")
            st.sidebar.write(f"임베딩 차원: {len(query_embedding)}")
        
        # match_documents 함수를 사용한 벡터 검색 - 더 관대한 설정
        try:
            # 뉴스의 경우 더 낮은 임계값 사용
            if source_type == "뉴스":
                adjusted_threshold = max(0.1, match_threshold - 0.3)
            else:
                adjusted_threshold = max(0.2, match_threshold - 0.2)
            
            response = supabase.rpc(
                'match_documents', 
                {
                    'query_embedding': query_embedding,
                    'match_threshold': adjusted_threshold,
                    'match_count': limit * 5  # 필터링 후 충분한 결과를 위해 더 많이 가져옴
                }
            ).execute()
            
            if response.data and len(response.data) > 0:
                # 클라이언트 측에서 소스 타입에 따라 필터링
                filtered_results = []
                for item in response.data:
                    # metadata 확인
                    metadata = item.get('metadata', {})
                    if isinstance(metadata, str):
                        try:
                            metadata = json.loads(metadata)
                        except:
                            metadata = {}
                    
                    item_source_type = metadata.get('collection', '')
                    
                    # 소스 타입이 일치하는 경우에만 추가
                    if item_source_type == source_type:
                        filtered_results.append(item)
                        # 뉴스 디버깅
                        if source_type == "뉴스" and len(filtered_results) <= 3:
                            st.sidebar.write(f"뉴스 매칭 발견: {metadata.get('title', '')[:30]}... (유사도: {item.get('similarity', 0):.3f})")
                
                # 유사도 점수로 재정렬
                filtered_results.sort(key=lambda x: x.get('similarity', 0), reverse=True)
                
                # 최대 limit 개수만큼 결과 반환
                return filtered_results[:limit]
            else:
                st.info(f"'{query_text}'에 대한 {source_type} 검색 결과가 없습니다.")
                # 뉴스 디버깅
                if source_type == "뉴스":
                    st.sidebar.warning(f"뉴스 검색 결과 없음. 임계값: {adjusted_threshold}")
                return []
                
        except Exception as e:
            st.sidebar.warning(f"시맨틱 검색 실패: {str(e)}")
            return []
        
    except Exception as e:
        st.error(f"시맨틱 검색 중 오류 발생: {str(e)}")
        raise

def get_system_prompt(source_type):
    """소스 타입에 따른 시스템 프롬프트 생성"""
    if source_type == "블로그":
        return """당신은 네이버 블로그 데이터를 기반으로 정확하고 유용한 정보를 제공하는 도우미입니다.
블로그 글은 개인의 경험과 의견을 담고 있으므로, 주관적인 내용이 포함될 수 있음을 인지하세요.
여러 블로그의 정보를 종합하여 균형 잡힌 시각을 제공하되, 정보의 출처가 개인 블로그임을 명시하세요.
특히 레시피, DIY 방법, 여행 경험 등 실용적인 정보에 집중하되, 의학적 조언이나 전문적인 내용은 참고 정보로만 안내하세요."""

    elif source_type == "뉴스":
        return """당신은 네이버 뉴스 데이터를 기반으로 정확하고 객관적인 정보를 제공하는 도우미입니다.
뉴스 기사의 사실과 정보를 전달할 때는 편향되지 않게 중립적인 입장을 유지하세요.
여러 언론사의 기사를 비교하여 다양한 관점을 제시하고, 정보의 출처와 발행 날짜를 명확히 하세요.
특히 시사 문제, 최신 이슈, 사회 현상에 대해 설명할 때는 다양한 의견이 있을 수 있음을 인지하세요."""

    elif source_type == "쇼핑":
        return """당신은 네이버 쇼핑 데이터를 기반으로 정확하고 유용한 정보를 제공하는 도우미입니다.
상품 정보, 가격, 기능, 특징 등을 객관적으로 설명하고 비교하세요.
다양한 상품 옵션과 가격대를 안내하되, 특정 브랜드나 제품을 지나치게 홍보하지 마세요.
사용자의 요구에 맞는 상품 추천이나 구매 팁을 제공할 때는 실용적인 관점에서 접근하세요."""

    else:
        return """당신은 네이버 검색 데이터를 기반으로 정확하고 유용한 정보를 제공하는 도우미입니다.
주어진 문서들의 내용만 사용하여 사용자 질문에 맞는 최적의 답변을 제공하세요.
문서에 없는 내용은 추가하지 말고 정확한 사실만 전달하세요."""

def get_user_prompt(query, context_text, source_type):
    """소스 타입에 따른 사용자 프롬프트 생성"""
    if source_type == "블로그":
        return f"""다음은 네이버 블로그에서 수집한 데이터입니다:

{context_text}

위 블로그 글들을 바탕으로 다음 질문에 상세히 답변해주세요: 
"{query}"

답변 작성 규칙:
1. 한국어로 자연스럽게 답변해주세요.
2. 블로그 글은 개인의 경험과 의견을 담고 있으므로, 정보의 주관성을 고려해주세요.
3. 여러 블로그의 공통된 내용에 중점을 두고, 개인적 경험이나 팁은 "블로거의 경험에 따르면..."과 같이 맥락을 제공해주세요.
4. 블로그 글들 간에 상충되는 정보가 있다면 "일부 블로거는 A를 추천하는 반면, 다른 블로거는 B를 선호합니다"와 같이 다양한 의견을 제시해주세요.
5. 레시피, DIY 방법, 여행 경험 등 실용적인 정보에 집중해주세요.
6. 출처를 명시할 때는 "문서 2의 블로거에 따르면..."과 같이 표현해주세요.
7. 제공된 문서 내용만 사용하고, 문서에 없는 내용은 추측하거나 답변하지 마세요."""

    elif source_type == "뉴스":
        return f"""다음은 네이버 뉴스에서 수집한, 신뢰할 수 있는 언론사의 기사입니다:

{context_text}

위 뉴스 기사들을 바탕으로 다음 질문에 상세히 답변해주세요: 
"{query}"

답변 작성 규칙:
1. 한국어로 자연스럽게 답변해주세요.
2. 뉴스 기사의 사실과 정보를 전달할 때는 편향되지 않게 중립적인 입장을 유지하세요.
3. 기사의 발행 날짜를 고려하여 정보의 시의성을 명시하세요. (예: "2023년 5월 보도에 따르면...")
4. 여러 언론사의 기사를 인용할 때는 "문서 1의 OO일보에 따르면..."와 같이 출처를 명확히 하세요.
5. 기사들 간에 상충되는 정보가 있다면 이를 언급하고 각 관점을 공정하게 제시하세요.
6. 제공된 기사 내용만 사용하고, 기사에 없는 내용은 추측하거나 답변하지 마세요."""

    elif source_type == "쇼핑":
        return f"""다음은 네이버 쇼핑에서 수집한 상품 정보입니다:

{context_text}

위 쇼핑 데이터를 바탕으로 다음 질문에 상세히 답변해주세요: 
"{query}"

답변 작성 규칙:
1. 한국어로 자연스럽게 답변해주세요.
2. 상품의 가격, 기능, 특징 등을 객관적으로 설명하고 비교해주세요.
3. 가격은 범위로 표현하고 정확한 가격이 있다면 언급해주세요
. (예: "이 제품은 30,000원에서 50,000원 사이의 가격대를 형성하고 있습니다")
4. 다양한 브랜드와 제품을 균형 있게 소개하고, 특정 상품을 지나치게 홍보하지 마세요.
5. 상품의 특징을 비교할 때는 "A 제품은 X 기능이 있지만, B 제품은 Y 기능이 강조됩니다"와 같이 객관적으로 설명해주세요.
6. 제공된 상품 정보만 사용하고, 문서에 없는 내용은 추측하거나 답변하지 마세요."""

   else:
       return f"""다음은 네이버 검색에서 수집한 데이터입니다:

{context_text}

위 내용을 바탕으로 다음 질문에 상세히 답변해주세요: 
"{query}"

답변 작성 규칙:
1. 한국어로 자연스럽게 답변해주세요.
2. 제공된 문서 내용만 사용하여 사실에 기반한 답변을 작성해주세요.
3. 문서에 없는 내용은 추측하거나 답변하지 마세요.
4. 여러 문서 간에 상충되는 정보가 있다면 이를 언급해주세요.
5. 답변에 적절한 정보가 부족하다면 솔직하게 말씀해주세요.
6. 답변은 논리적인 구조로 정리하여 사용자가 이해하기 쉽게 작성해주세요.
7. 필요한 경우 정보의 출처를 언급해주세요(예: "문서 2에 따르면...")."""

def generate_answer_with_gpt(query, search_results, source_type):
   """GPT-4o-mini를 사용하여 검색 결과에 기반한 답변 생성"""
   try:
       # 검색 결과가 없는 경우
       if not search_results:
           return f"죄송합니다. 입력하신 '{query}'에 대한 {source_type} 검색 결과를 찾을 수 없습니다. 다른 검색어나 다른 소스 타입으로 시도해보세요."
           
       # 검색 결과를 컨텍스트로 정리
       contexts = []
       for i, result in enumerate(search_results[:5]):  # 상위 5개 결과만 사용
           content = result['content']
           
           # metadata 확인 (JSON 문자열일 경우 파싱)
           metadata = result.get('metadata', {})
           if isinstance(metadata, str):
               try:
                   metadata = json.loads(metadata)
               except:
                   metadata = {}
                   
           title = metadata.get('title', '제목 없음')
           date = metadata.get('date', '')  # 날짜 정보가 있으면 추가
           
           # 날짜 정보가 있으면 포함
           date_info = f" (작성일: {date})" if date else ""
           
           # 소스 타입에 맞는 추가 정보
           if source_type == "블로그" and 'bloggername' in metadata:
               source_info = f" - 블로거: {metadata['bloggername']}"
           elif source_type == "뉴스" and 'publisher' in metadata:
               source_info = f" - 출처: {metadata['publisher']}"
           elif source_type == "쇼핑" and 'mallname' in metadata:
               price_info = f", 가격: {metadata.get('lprice', '정보 없음')}원" if 'lprice' in metadata else ""
               source_info = f" - 판매처: {metadata['mallname']}{price_info}"
           else:
               source_info = ""
           
           # 유사도 점수 추가
           similarity = result.get('similarity', 0) * 100
           similarity_info = f" (유사도: {similarity:.1f}%)"
           
           # 출처 타입과 함께 컨텍스트 추가
           contexts.append(f"문서 {i+1} - [{source_type}] {title}{date_info}{source_info}{similarity_info}:\n{content}\n")
       
       context_text = "\n".join(contexts)
       
       # 소스 타입에 맞는 프롬프트 생성
       system_prompt = get_system_prompt(source_type)
       user_prompt = get_user_prompt(query, context_text, source_type)

       # GPT-4o-mini로 답변 생성
       response = openai_client.chat.completions.create(
           model="gpt-4o-mini",
           messages=[
               {"role": "system", "content": system_prompt},
               {"role": "user", "content": user_prompt}
           ],
           temperature=0.3,  # 일관성 있는 답변을 위해 낮은 온도 설정
           max_tokens=1000   # 충분한 답변 길이
       )
       
       return response.choices[0].message.content
       
   except Exception as e:
       st.error(f"GPT 답변 생성 중 오류 발생: {str(e)}")
       return "답변 생성 중 오류가 발생했습니다."

# 메인 UI
st.title("네이버 통합 검색 & 질의응답")
st.write("시맨틱 검색 기술을 이용하여 네이버 데이터를 검색하고 질문에 답변합니다.")

# 검색 모드 선택
search_mode = st.sidebar.radio(
   "검색 모드 선택", 
   options=["시맨틱 검색 (저장된 데이터)", "새 데이터 수집 및 저장"], 
   index=0
)

# 검색 소스 선택 (라디오 버튼) - 가로로 배치
source_type = st.radio(
   "검색 소스 선택", 
   options=["블로그", "뉴스", "쇼핑"], 
   index=0,
   horizontal=True  # 가로로 배치
)

# 검색 입력 - 소스 타입에 따라 다른 예시 질문 제공
if source_type == "블로그":
   default_query = "안성탕면 맛있게 끓이는 방법이 뭐지?"
   help_text = "블로그 데이터에서 레시피, 리뷰, 여행 경험 등을 검색해보세요"
elif source_type == "뉴스":
   default_query = "최근 경제 이슈는 무엇인가요?"
   help_text = "뉴스 데이터에서 시사, 경제, 사회 이슈 등을 검색해보세요"
elif source_type == "쇼핑":
   default_query = "삼성노트북 최신 노트북이 뭐지?"
   help_text = "쇼핑 데이터에서 상품 정보, 가격 비교, 구매 팁 등을 검색해보세요"

# 검색 입력
query = st.text_input("질문 입력", value=default_query, help=help_text)

# 원본 검색 결과 표시 옵션
show_raw_results = st.sidebar.checkbox("원본 검색 결과 표시", value=True)

# 검색 결과 수 및 유사도 설정
if search_mode == "시맨틱 검색 (저장된 데이터)":
   col1, col2 = st.sidebar.columns(2)
   with col1:
       result_count = st.slider("검색 결과 수", min_value=3, max_value=20, value=10)
   with col2:
       similarity_threshold = st.slider("유사도 임계값", min_value=0.0, max_value=1.0, value=0.4, step=0.05)  # 기본값을 0.4로 낮춤
else:
   result_count = st.sidebar.slider("검색 결과 수", min_value=5, max_value=50, value=20)

# 검색 버튼
search_button_text = "시맨틱 검색" if search_mode == "시맨틱 검색 (저장된 데이터)" else "데이터 수집 및 저장"
if st.button(f"{source_type} {search_button_text}", key="search_button"):
   if query:
       if search_mode == "시맨틱 검색 (저장된 데이터)":
           # 시맨틱 검색 모드
           with st.spinner(f"{source_type} 시맨틱 검색 중..."):
               try:
                   # 시맨틱 검색 수행
                   results = semantic_search(query, source_type=source_type, limit=result_count, match_threshold=similarity_threshold)
                   
                   if results:
                       st.success(f"{len(results)}개의 {source_type} 결과를 찾았습니다.")
                       
                       # GPT로 답변 생성
                       with st.spinner("AI 에이전트 답변 생성 중..."):
                           gpt_answer = generate_answer_with_gpt(query, results, source_type)
                           
                           # 답변 표시
                           st.markdown(f"## AI 답변 ({source_type} 데이터 기반)")
                           st.markdown(gpt_answer)
                           
                           # 구분선
                           st.markdown("---")
                       
                       # 원본 검색 결과 표시 옵션
                       if show_raw_results:
                           st.markdown(f"## {source_type} 검색 결과 원본")
                           for i, result in enumerate(results):
                               similarity = result['similarity'] * 100  # 백분율로 변환
                               
                               # 메타데이터 확인 (JSON 문자열일 경우 파싱)
                               metadata = result.get('metadata', {})
                               if isinstance(metadata, str):
                                   try:
                                       metadata = json.loads(metadata)
                                   except:
                                       metadata = {}
                               
                               title = metadata.get('title', '제목 없음')
                               
                               # URL 추출
                               url = metadata.get('url', None)
                               
                               # 결과 표시
                               with st.expander(f"{i+1}. {title} (유사도: {similarity:.2f}%)"):
                                   st.write(f"**내용:** {result['content']}")
                                   
                                   # 메타데이터 정보 표시
                                   meta_col1, meta_col2 = st.columns(2)
                                   
                                   with meta_col1:
                                       if source_type == "블로그" and 'bloggername' in metadata:
                                           st.write(f"**블로거:** {metadata['bloggername']}")
                                       elif source_type == "뉴스" and 'publisher' in metadata:
                                           st.write(f"**언론사:** {metadata['publisher']}")
                                       elif source_type == "쇼핑" and 'maker' in metadata:
                                           st.write(f"**제조사:** {metadata['maker']}")
                                       elif source_type == "쇼핑" and 'brand' in metadata:
                                           st.write(f"**브랜드:** {metadata['brand']}")
                                           
                                       if 'date' in metadata:
                                           st.write(f"**날짜:** {metadata['date']}")
                                   
                                   with meta_col2:
                                       if url:
                                           st.markdown(f"**링크:** [원본 보기]({url})")
                                       if source_type == "쇼핑":
                                           if 'lprice' in metadata:
                                               st.write(f"**최저가:** {metadata['lprice']}원")
                                           if 'mallname' in metadata:
                                               st.write(f"**판매처:** {metadata['mallname']}")
                   else:
                       st.warning(f"{source_type}에서 검색 결과가 없습니다. 새 데이터를 수집하거나 다른 검색어를 시도해보세요.")
                       st.info("💡 팁: 유사도 임계값을 더 낮추거나, 다른 검색어로 시도해보세요.")
               
               except Exception as e:
                   st.error(f"검색 중 오류가 발생했습니다: {str(e)}")
       
       else:
           # 네이버 API 검색 및 저장 모드
           with st.spinner(f"네이버 {source_type} API 검색 및 데이터 저장 중..."):
               try:
                   # 네이버 API 검색 수행 및 Supabase에 저장
                   items, total_count, saved_count = search_naver_api(query, source_type, result_count)
                   
                   if items:
                       st.success(f"네이버 {source_type}에서 총 {total_count}개 중 {len(items)}개의 결과를 찾았고, {saved_count}개를 새로 저장했습니다.")
                       
                       # 저장 후 즉시 시맨틱 검색 수행
                       with st.spinner("저장된 데이터로 시맨틱 검색 중..."):
                           # 잠시 대기 (데이터베이스 저장 완료 대기) - 시간 증가
                           time.sleep(5)  # 5초 대기
                           
                           # 시맨틱 검색 수행 - 더 관대한 설정
                           results = semantic_search(query, source_type=source_type, limit=result_count, match_threshold=0.3)
                           
                           if results:
                               # GPT로 답변 생성
                               with st.spinner("AI 에이전트 답변 생성 중..."):
                                   gpt_answer = generate_answer_with_gpt(query, results, source_type)
                                   
                                   # 답변 표시
                                   st.markdown(f"## AI 답변 ({source_type} 데이터 기반)")
                                   st.markdown(gpt_answer)
                                   
                                   # 구분선
                                   st.markdown("---")
                           else:
                               st.warning("데이터는 저장되었지만 시맨틱 검색에서 관련 결과를 찾지 못했습니다. 잠시 후 다시 시도해 보세요.")
                               st.info("💡 새로 저장된 데이터의 임베딩 처리가 완료될 때까지 몇 분 정도 소요될 수 있습니다.")
                       
                       # 네이버 API 결과 표시
                       if show_raw_results:
                           st.markdown(f"## 네이버 {source_type} 검색 결과")
                           
                           # 데이터프레임으로 표시할 데이터 준비
                           df_data = []
                           for i, item in enumerate(items):
                               try:
                                   # HTML 태그 제거
                                   title = re.sub('<[^<]+?>', '', item.get('title', '')) if item.get('title') else '제목 없음'
                                   
                                   # 소스 타입별 표시 항목
                                   if source_type == "블로그":
                                       description = re.sub('<[^<]+?>', '', item.get('description', '')) if item.get('description') else ''
                                       df_data.append({
                                           '제목': title,
                                           '내용 미리보기': description[:100] + "..." if len(description) > 100 else description,
                                           '블로거': item.get('bloggername', ''),
                                           '날짜': item.get('postdate', ''),
                                           '링크': item.get('link', '')
                                       })
                                   elif source_type == "뉴스":
                                       description = re.sub('<[^<]+?>', '', item.get('description', '')) if item.get('description') else ''
                                       df_data.append({
                                           '제목': title,
                                           '내용 미리보기': description[:100] + "..." if len(description) > 100 else description,
                                           '언론사': item.get('originallink', '').replace('https://', '').replace('http://', '').split('/')[0] if item.get('originallink') else '',
                                           '날짜': item.get('pubDate', ''),
                                           '링크': item.get('link', '')
                                       })
                                   elif source_type == "쇼핑":
                                       price_display = f"{item.get('lprice', '')}원" if item.get('lprice') else '가격 정보 없음'
                                       df_data.append({
                                           '제품명': title,
                                           '가격': price_display,
                                           '판매처': item.get('mallName', ''),
                                           '제조사': item.get('maker', ''),
                                           '링크': item.get('link', '')
                                       })
                               except Exception as e:
                                   st.warning(f"항목 {i+1} 처리 중 오류: {str(e)}")
                                   continue
                           
                           if df_data:
                               # 데이터프레임 생성 및 표시
                               df = pd.DataFrame(df_data)
                               st.dataframe(df, use_container_width=True)
                               
                               # 각 결과 상세 내용 표시
                               for i, item in enumerate(items):
                                   try:
                                       title = re.sub('<[^<]+?>', '', item.get('title', '')) if item.get('title') else '제목 없음'
                                       
                                       with st.expander(f"{i+1}. {title}"):
                                           if source_type in ["블로그", "뉴스"]:
                                               description = re.sub('<[^<]+?>', '', item.get('description', '')) if item.get('description') else ''
                                               if description:
                                                   st.write(f"**내용:** {description}")
                                           
                                           # 메타데이터 정보 표시
                                           meta_col1, meta_col2 = st.columns(2)
                                           
                                           with meta_col1:
                                               if source_type == "블로그":
                                                   if item.get('bloggername'):
                                                       st.write(f"**블로거:** {item.get('bloggername')}")
                                                   if item.get('postdate'):
                                                       st.write(f"**날짜:** {item.get('postdate')}")
                                               elif source_type == "뉴스":
                                                   if item.get('originallink'):
                                                       publisher = item.get('originallink', '').replace('https://', '').replace('http://', '').split('/')[0]
                                                       st.write(f"**언론사:** {publisher}")
                                                   if item.get('pubDate'):
                                                       st.write(f"**날짜:** {item.get('pubDate')}")
                                               elif source_type == "쇼핑":
                                                   if item.get('maker'):
                                                       st.write(f"**제조사:** {item.get('maker')}")
                                                   if item.get('brand'):
                                                       st.write(f"**브랜드:** {item.get('brand')}")
                                           
                                           with meta_col2:
                                               if item.get('link'):
                                                   st.markdown(f"**링크:** [원본 보기]({item.get('link')})")
                                               if source_type == "쇼핑":
                                                   if item.get('lprice'):
                                                       st.write(f"**최저가:** {item.get('lprice')}원")
                                                   if item.get('mallName'):
                                                       st.write(f"**판매처:** {item.get('mallName')}")
                                   except Exception as e:
                                       st.warning(f"항목 {i+1} 표시 중 오류: {str(e)}")
                                       continue
                           else:
                               st.warning("표시할 수 있는 검색 결과가 없습니다.")
                   else:
                       st.warning(f"네이버 {source_type}에서 검색 결과가 없습니다. 다른 검색어나 다른 소스 타입으로 시도해보세요.")
               
               except Exception as e:
                   st.error(f"검색 중 오류가 발생했습니다: {str(e)}")
                   import traceback
                   st.error(f"상세 오류: {traceback.format_exc()}")
   else:
       st.warning("질문을 입력하세요.")

# 데이터베이스 상태
st.sidebar.title("데이터베이스 상태")
try:
   # 전체 문서 수 가져오기
   result = supabase.table('documents').select('id', count='exact').execute()
   doc_count = result.count if hasattr(result, 'count') else len(result.data)
   st.sidebar.info(f"저장된 총 문서 수: {doc_count}개")
   
   # 각 소스 타입별 문서 수 표시 시도
   try:
       collections = {}
       collection_query = supabase.table('documents').select('metadata').execute()
       for item in collection_query.data:
           metadata = item.get('metadata', {})
           if isinstance(metadata, str):
               try:
                   metadata = json.loads(metadata)
               except:
                   continue
           
           collection = metadata.get('collection', '기타')
           if collection in collections:
               collections[collection] += 1
           else:
               collections[collection] = 1
       
       # 소스 타입별 문서 수 표시
       for collection, count in collections.items():
           st.sidebar.info(f"{collection} 문서 수: {count}개")
   except Exception as e:
       st.sidebar.warning(f"소스별 통계 조회 실패: {str(e)}")
       
except Exception as e:
   st.sidebar.error(f"데이터베이스 상태를 확인할 수 없습니다: {str(e)}")

# 뉴스 데이터 샘플 확인 버튼 추가
if st.sidebar.button("뉴스 데이터 샘플 확인"):
   try:
       with st.spinner("뉴스 데이터 조회 중..."):
           news_sample = supabase.table('documents').select('*').eq('metadata->>collection', '뉴스').limit(5).execute()
           
           if news_sample.data:
               st.sidebar.write("### 저장된 뉴스 데이터 샘플")
               for i, item in enumerate(news_sample.data):
                   st.sidebar.write(f"**샘플 {i+1}:**")
                   st.sidebar.write(f"내용: {item['content'][:100]}...")
                   
                   metadata = item.get('metadata', {})
                   if isinstance(metadata, str):
                       try:
                           metadata = json.loads(metadata)
                       except:
                           metadata = {}
                   
                   st.sidebar.write(f"메타데이터: {metadata}")
                   st.sidebar.write("---")
           else:
               st.sidebar.warning("저장된 뉴스 데이터가 없습니다.")
   except Exception as e:
       st.sidebar.error(f"뉴스 데이터 조회 실패: {str(e)}")

# 사용 안내
st.sidebar.title("사용 안내")
st.sidebar.info(f"""
**검색 모드:**
1. **시맨틱 검색 (저장된 데이터)**: 이미 저장된 데이터를 의미 기반으로 검색합니다.
2. **새 데이터 수집 및 저장**: 네이버 API에서 새 데이터를 가져와 저장하고 검색합니다.

**검색 소스 선택:** 블로그, 뉴스, 쇼핑 중에서 검색할 소스를 선택하세요.

**유사도 임계값:** 시맨틱 검색에서 얼마나 유사한 결과를 포함할지 결정합니다. 
- 높음 (0.7~1.0): 매우 관련성 높은 결과만 표시
- 중간 (0.4~0.6): 균형잡힌 관련성 (권장)
- 낮음 (0.1~0.3): 더 많은 결과를 포함하지만 관련성이 낮을 수 있음

💡 **개선 사항:**
- 뉴스 데이터 저장 형식 개선
- 뉴스 전용 낮은 유사도 임계값 적용
- 언론사 정보 추출 로직 개선
- 뉴스 검색 쿼리 최적화

💡 팁: 각 소스 타입에 적합한 질문을 입력하세요:
- 블로그: 레시피, 여행 경험, 리뷰, DIY 방법 등
- 뉴스: 시사 이슈, 사회 현상, 경제 동향 등
- 쇼핑: 상품 정보, 가격 비교, 구매 팁 등
""")

# 네이버 API 정보 및 문제해결
st.sidebar.title("네이버 API 정보")
st.sidebar.info("""
**API 상태:**
- Client ID: 9XhhxLV1IzDpTZagoBr1
- 데이터 출처: 네이버 검색 API

**문제해결:**
- API 오류 시 잠시 후 다시 시도
- 검색어를 단순하게 변경해보세요
- 다른 소스 타입으로 시도해보세요
""")

# 추가 디버깅 정보 (개발용)
st.sidebar.title("디버깅 정보")
if st.sidebar.checkbox("디버깅 모드", value=False):
   st.sidebar.write(f"현재 검색 모드: {search_mode}")
   st.sidebar.write(f"선택된 소스: {source_type}")
   st.sidebar.write(f"현재 쿼리: {query}")
   st.sidebar.write(f"사용 중인 임베딩 모델: jhgan/ko-sroberta-multitask")
   
   # API 키 상태 확인
   st.sidebar.write("**API 키 상태:**")
   st.sidebar.write(f"- Supabase URL: {'✅' if supabase_url else '❌'}")
   st.sidebar.write(f"- Supabase Key: {'✅' if supabase_key else '❌'}")
   st.sidebar.write(f"- OpenAI Key: {'✅' if openai_api_key else '❌'}")
   st.sidebar.write(f"- Naver Client ID: {'✅' if NAVER_CLIENT_ID else '❌'}")
   st.sidebar.write(f"- Naver Client Secret: {'✅' if NAVER_CLIENT_SECRET else '❌'}")
