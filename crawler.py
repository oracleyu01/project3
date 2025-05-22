import requests
import os
import json
from datetime import datetime
from supabase import create_client
from openai import OpenAI
import re
import time

# 환경 변수 로드
NAVER_CLIENT_ID = os.environ.get("NAVER_CLIENT_ID")
NAVER_CLIENT_SECRET = os.environ.get("NAVER_CLIENT_SECRET")
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# 클라이언트 초기화
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# 검색할 키워드 리스트
SEARCH_KEYWORDS = [
    "아이폰 15",
    "갤럭시 S24", 
    "맥북 프로",
    "에어팟 프로"
]

def generate_embedding(text):
    """텍스트에서 OpenAI 임베딩 생성"""
    try:
        response = openai_client.embeddings.create(
            input=text,
            model="text-embedding-3-small"
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"임베딩 생성 오류: {e}")
        return None

def clean_html_tags(text):
    """HTML 태그 제거"""
    if not text:
        return ""
    return re.sub(r'<.*?>', '', text)

def search_naver_shopping(keyword, display=50, sort="sim"):
    """네이버 쇼핑 API 검색"""
    url = "https://openapi.naver.com/v1/search/shop.json"
    
    headers = {
        "X-Naver-Client-Id": NAVER_CLIENT_ID,
        "X-Naver-Client-Secret": NAVER_CLIENT_SECRET
    }
    
    params = {
        "query": keyword,
        "display": display,
        "sort": sort
    }
    
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"API 요청 오류 ({keyword}): {e}")
        return None

def check_duplicate_by_url(url):
    """URL로 중복 상품 확인"""
    try:
        # metadata에서 url 필드로 중복 확인
        result = supabase.table('documents').select('id').eq('metadata->>url', url).execute()
        return len(result.data) > 0
    except Exception as e:
        print(f"중복 확인 오류: {e}")
        return False

def save_to_supabase(items, keyword):
    """Supabase에 상품 데이터 저장"""
    saved_count = 0
    skipped_count = 0
    
    for item in items:
        # 기본 정보 추출
        title = clean_html_tags(item.get('title', ''))
        description = clean_html_tags(item.get('description', ''))
        product_url = item.get('link', '')
        
        # 중복 확인 (URL 기준)
        if check_duplicate_by_url(product_url):
            skipped_count += 1
            continue
        
        # content 필드 생성 (제목 + 설명)
        content = f"{title} {description}".strip()
        
        # 가격 정보 처리
        lprice = item.get('lprice', '')
        hprice = item.get('hprice', '')
        
        try:
            lprice = int(lprice) if lprice else None
            hprice = int(hprice) if hprice else None
        except (ValueError, TypeError):
            lprice = None
            hprice = None
        
        # metadata 필드 구성 (JSONB 형태)
        metadata = {
            "title": title,
            "description": description,
            "url": product_url,
            "search_keyword": keyword,
            "source": "naver_shopping",
            "collected_at": datetime.now().isoformat(),
            "price": {
                "lprice": lprice,  # 최저가
                "hprice": hprice   # 최고가
            },
            "product_info": {
                "productId": item.get('productId', ''),
                "productType": item.get('productType', ''),
                "maker": item.get('maker', ''),
                "brand": item.get('brand', ''),
                "category1": item.get('category1', ''),
                "category2": item.get('category2', ''),
                "category3": item.get('category3', ''),
                "category4": item.get('category4', '')
            },
            "mall_info": {
                "mallName": item.get('mallName', ''),
                "image": item.get('image', '')
            }
        }
        
        # 임베딩 생성
        embedding = generate_embedding(content)
        if not embedding:
            print(f"임베딩 생성 실패로 스킵: {title[:30]}...")
            continue
        
        # Supabase에 데이터 삽입
        try:
            insert_data = {
                'content': content,
                'metadata': metadata,
                'embedding': embedding
            }
            
            result = supabase.table('documents').insert(insert_data).execute()
            
            if result.data:
                saved_count += 1
                print(f"✅ 저장완료: {title[:50]}...")
            else:
                print(f"❌ 저장실패: {title[:30]}...")
                
        except Exception as e:
            print(f"저장 오류: {e}")
            print(f"실패한 상품: {title[:30]}...")
        
        # API 호출 간격 조절
        time.sleep(0.1)
    
    return saved_count, skipped_count

def get_database_stats():
    """데이터베이스 통계 정보"""
    try:
        # 전체 문서 수
        total_result = supabase.table('documents').select('id', count='exact').execute()
        total_count = total_result.count if hasattr(total_result, 'count') else len(total_result.data)
        
        # 네이버 쇼핑 데이터 수
        shopping_result = supabase.table('documents').select('id', count='exact').eq('metadata->>source', 'naver_shopping').execute()
        shopping_count = shopping_result.count if hasattr(shopping_result, 'count') else len(shopping_result.data)
        
        return total_count, shopping_count
    except Exception as e:
        print(f"통계 조회 오류: {e}")
        return None, None

def main():
    """메인 실행 함수"""
    print(f"🚀 네이버 쇼핑 크롤링 시작: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 환경 변수 확인
    required_vars = [NAVER_CLIENT_ID, NAVER_CLIENT_SECRET, SUPABASE_URL, SUPABASE_KEY, OPENAI_API_KEY]
    if not all(required_vars):
        print("❌ 필수 환경 변수가 설정되지 않았습니다.")
        return
    
    total_saved = 0
    total_skipped = 0
    
    # 시작 전 데이터베이스 상태
    before_total, before_shopping = get_database_stats()
    if before_total is not None:
        print(f"📊 시작 전 DB 상태 - 전체: {before_total}개, 쇼핑: {before_shopping}개")
    
    # 각 키워드별 크롤링
    for i, keyword in enumerate(SEARCH_KEYWORDS, 1):
        print(f"\n🔍 [{i}/{len(SEARCH_KEYWORDS)}] '{keyword}' 검색 중...")
        
        # 네이버 쇼핑 API 호출
        result = search_naver_shopping(keyword, display=100)
        
        if not result or 'items' not in result:
            print(f"❌ '{keyword}' 검색 결과 없음")
            continue
        
        items = result['items']
        print(f"📦 '{keyword}': {len(items)}개 상품 발견")
        
        # Supabase에 저장
        saved, skipped = save_to_supabase(items, keyword)
        total_saved += saved
        total_skipped += skipped
        
        print(f"✅ '{keyword}' 처리완료: {saved}개 저장, {skipped}개 중복스킵")
        
        # API 요청 간격
        if i < len(SEARCH_KEYWORDS):
            time.sleep(2)
    
    # 완료 후 데이터베이스 상태
    after_total, after_shopping = get_database_stats()
    
    print(f"\n🎉 크롤링 완료: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📈 결과 요약:")
    print(f"   - 새로 저장: {total_saved}개")
    print(f"   - 중복 스킵: {total_skipped}개")
    
    if after_total is not None and before_total is not None:
        print(f"📊 DB 상태 변화:")
        print(f"   - 전체 문서: {before_total} → {after_total} (+{after_total - before_total})")
        print(f"   - 쇼핑 데이터: {before_shopping} → {after_shopping} (+{after_shopping - before_shopping})")

if __name__ == "__main__":
    main()
