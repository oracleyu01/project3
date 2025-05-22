import streamlit as st
import json
import os
import tempfile
from datetime import datetime
from supabase import create_client
import numpy as np
from openai import OpenAI
import dotenv
import re

# 환경 변수 로드
dotenv.load_dotenv()

# Supabase 클라이언트 초기화
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_KEY")
supabase = create_client(supabase_url, supabase_key)

# OpenAI 클라이언트 초기화 (벡터 임베딩용)
openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def generate_embedding(text):
    """텍스트에서 OpenAI 임베딩 생성"""
    response = openai_client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

def clean_html_tags(text):
    """HTML 태그 제거"""
    if not text:
        return ""
    return re.sub(r'<.*?>', '', text)

def detect_naver_api_type(data):
    """네이버 API 응답 타입 감지 (블로그, 쇼핑, 뉴스)"""
    if not isinstance(data, dict) or 'items' not in data:
        return "unknown"
    
    # 샘플 아이템 확인
    if not data['items']:
        return "unknown"
    
    sample_item = data['items'][0]
    
    # 타입 감지 로직
    if 'bloggername' in sample_item:
        return "블로그"
    elif 'productType' in sample_item or 'maker' in sample_item or 'mallName' in sample_item:
        return "쇼핑"
    elif 'pubDate' in sample_item and ('articleId' in sample_item or 'originallink' in sample_item):
        return "뉴스"
    else:
        return "unknown"

def process_json_file(file_path, collection_name=None, source_type=None):
    """JSON 파일 처리 및 Supabase에 저장"""
    # JSON 파일 로드
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 네이버 API 응답 구조 확인
    if isinstance(data, dict) and 'items' in data:
        # 네이버 API 응답 형식인 경우
        items = data['items']
        
        # 소스 타입이 지정되지 않은 경우 자동 감지
        if not source_type:
            source_type = detect_naver_api_type(data)
            st.info(f"데이터 형식이 '{source_type}'으로 감지되었습니다.")
    else:
        # 직접 JSON 배열인 경우
        items = data
    
    # 컬렉션 이름 생성
    if not collection_name:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        collection_name = f'{source_type}_{timestamp}'
    
    # 처리된 문서 수 카운트
    doc_count = 0
    
    # 각 항목 처리
    for i, item in enumerate(items):
        # 소스 타입별로 다른 필드 처리
        if source_type == "블로그":
            title = clean_html_tags(item.get('title', ''))
            content = clean_html_tags(item.get('description', ''))
            full_content = title + " " + content
            
            metadata = {
                "title": title,
                "collection": source_type,
                "collected_at": datetime.now().isoformat(),
                "url": item.get('link', ''),
                "date": item.get('postdate', ''),
                "bloggername": item.get('bloggername', ''),
                "bloggerlink": item.get('bloggerlink', '')
            }
            
        elif source_type == "쇼핑":
            title = clean_html_tags(item.get('title', ''))
            content = clean_html_tags(item.get('description', item.get('category3', '')))
            full_content = title + " " + content
            
            # 가격 정보 숫자로 변환
            price = item.get('lprice', '')
            try:
                price = int(price)
            except (ValueError, TypeError):
                price = None
                
            metadata = {
                "title": title,
                "collection": source_type,
                "collected_at": datetime.now().isoformat(),
                "url": item.get('link', ''),
                "price": price,
                "maker": item.get('maker', ''),
                "brand": item.get('brand', ''),
                "mallName": item.get('mallName', ''),
                "productId": item.get('productId', ''),
                "productType": item.get('productType', '')
            }
            
        elif source_type == "뉴스":
            title = clean_html_tags(item.get('title', ''))
            content = clean_html_tags(item.get('description', ''))
            full_content = title + " " + content
            
            metadata = {
                "title": title,
                "collection": source_type,
                "collected_at": datetime.now().isoformat(),
                "url": item.get('link', item.get('originallink', '')),
                "date": item.get('pubDate', ''),
                "publisher": item.get('publisher', '')
            }
            
        else:
            # 기본 처리 (타입이 불분명한 경우)
            title = clean_html_tags(item.get('title', ''))
            content = clean_html_tags(item.get('description', item.get('content', '')))
            full_content = title + " " + content
            
            metadata = {
                "title": title,
                "collection": source_type if source_type else "general",
                "collected_at": datetime.now().isoformat()
            }
            
            # 공통 필드 추가
            if 'link' in item:
                metadata['url'] = item['link']
            
        # 임베딩 생성
        embedding = generate_embedding(full_content)
        
        # Supabase에 데이터 삽입
        data = {
            'content': full_content,
            'embedding': embedding,
            'metadata': metadata
        }
        
        supabase.table('documents').insert(data).execute()
        doc_count += 1
    
    return collection_name, doc_count, source_type

# Streamlit 앱 UI
st.title("네이버 JSON 파일을 Supabase에 저장하기")

uploaded_file = st.file_uploader("JSON 파일 업로드", type=['json'])

if uploaded_file is not None:
    # 파일 저장
    with tempfile.NamedTemporaryFile(delete=False, suffix='.json') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name
    
    # 타입 선택
    source_type = st.radio(
        "데이터 소스 타입 선택 (자동 감지하려면 '자동 감지' 선택)",
        ['자동 감지', '블로그', '쇼핑', '뉴스']
    )
    
    # 자동 감지인 경우 None으로 설정
    if source_type == '자동 감지':
        source_type = None
    
    # 컬렉션 이름 입력
    collection_name = st.text_input("컬렉션 이름 (입력하지 않으면 자동 생성됩니다)")
    
    if st.button("Supabase에 저장"):
        with st.spinner("데이터 처리 중..."):
            try:
                collection_name, doc_count, detected_type = process_json_file(
                    tmp_file_path, 
                    collection_name, 
                    source_type
                )
                
                st.success(f"성공적으로 {doc_count}개의 문서가 저장되었습니다!")
                st.write(f"컬렉션 이름: {collection_name}")
                st.write(f"데이터 타입: {detected_type}")
                
                # 데이터베이스 상태 표시
                try:
                    result = supabase.table('documents').select('id', count='exact').execute()
                    doc_count_total = result.count if hasattr(result, 'count') else len(result.data)
                    st.write(f"데이터베이스 총 문서 수: {doc_count_total}개")
                except Exception as e:
                    st.warning(f"데이터베이스 상태 확인 중 오류: {str(e)}")
                
            except Exception as e:
                st.error(f"오류 발생: {str(e)}")
            finally:
                # 임시 파일 삭제
                os.unlink(tmp_file_path)
