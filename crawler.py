import requests
import os
import json
from datetime import datetime
from supabase import create_client
from sentence_transformers import SentenceTransformer
import re
import time
import dotenv

# 환경 변수 로드
dotenv.load_dotenv()

NAVER_CLIENT_ID = os.environ.get("NAVER_CLIENT_ID")
NAVER_CLIENT_SECRET = os.environ.get("NAVER_CLIENT_SECRET")
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

# 클라이언트 초기화
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Sentence Transformer 모델 초기화
def load_embedding_model():
    """임베딩 모델 로드"""
    return SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

# 전역 변수로 모델 로드
print("🔄 임베딩 모델 로딩 중...")
embedding_model = load_embedding_model()
print("✅ 임베딩 모델 로드 완료")

# 아이폰 15 본체만 수집하도록 수정된 키워드
SEARCH_KEYWORDS = [
    "아이폰 15 128GB",
    "iPhone 15 128GB", 
    "아이폰15 본체",
    "아이폰 15 256GB",
    "iPhone 15 256GB",
    "아이폰 15 512GB",
    "iPhone 15 512GB"
]

# 액세서리 제외 키워드 (제목에 포함되면 제외)
EXCLUDE_KEYWORDS = [
    "케이스", "case", "필름", "film", "케이블", "cable", "충전기", "charger",
    "어댑터", "adapter", "스탠드", "stand", "홀더", "holder", "그립", "grip",
    "스트랩", "strap", "파우치", "pouch", "범퍼", "bumper", "커버", "cover",
    "보호", "protection", "액세서리", "accessory", "젤리", "jelly",
    "실리콘", "silicon", "투명", "clear", "클리어", "하드", "hard"
]

# 브랜드/제조사 키워드 (포함되면 우선 선택)
PHONE_BRANDS = ["애플", "apple", "아이폰", "iphone"]

def generate_embedding(text):
    """텍스트에서 임베딩 생성"""
    try:
        if not text or text.strip() == "":
            return [0.0] * 768
        
        embedding = embedding_model.encode(text)
        embedding_list = embedding.tolist()
        
        # 768차원을 1536차원으로 확장
        if len(embedding_list) < 1536:
            embedding_list.extend([0.0] * (1536 - len(embedding_list)))
        
        return embedding_list[:1536]
        
    except Exception as e:
        print(f"임베딩 생성 오류: {e}")
        return None

def clean_html_tags(text):
    """HTML 태그 제거"""
    if not text:
        return ""
    return re.sub(r'<.*?>', '', text)

def is_phone_product(item):
    """아이폰 15 본체인지 판별하는 함수"""
    title = item.get('title', '').lower()
    brand = item.get('brand', '').lower()
    category2 = item.get('category2', '').lower()
    category3 = item.get('category3', '').lower()
    lprice = item.get('lprice', '')
    
    # 1. 액세서리 키워드가 포함된 경우 제외
    for exclude_word in EXCLUDE_KEYWORDS:
        if exclude_word in title:
            return False
    
    # 2. 가격이 너무 낮으면 제외 (아이폰 본체는 최소 80만원 이상)
    try:
        price = int(lprice) if lprice else 0
        if price > 0 and price < 800000:  # 80만원 미만 제외
            return False
        if price > 2000000:  # 200만원 초과도 제외 (이상치)
            return False
    except (ValueError, TypeError):
        pass
    
    # 3. 카테고리가 휴대폰 관련인지 확인
    phone_categories = ["휴대폰", "스마트폰", "mobile", "phone"]
    category_match = any(cat in category2 or cat in category3 for cat in phone_categories)
    
    # 4. 용량 키워드 포함 여부 확인
    storage_keywords = ["128gb", "256gb", "512gb", "1tb"]
    has_storage = any(storage in title for storage in storage_keywords)
    
    # 5. 브랜드가 애플/아이폰인지 확인
    brand_match = any(brand_word in title or brand_word in brand for brand_word in PHONE_BRANDS)
    
    # 최종 판별: (카테고리 매치 OR 용량 포함) AND 브랜드 매치
    return (category_match or has_storage) and brand_match

def search_naver_shopping(keyword, display=100, sort="sim"):
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
        result = supabase.table('documents').select('id').eq('metadata->>url', url).execute()
        return len(result.data) > 0
    except Exception as e:
        print(f"중복 확인 오류: {e}")
        return False

def save_to_supabase(items, keyword):
    """Supabase에 상품 데이터 저장 (아이폰 본체만)"""
    saved_count = 0
    skipped_count = 0
    filtered_count = 0
    
    for item in items:
        # 아이폰 본체 필터링
        if not is_phone_product(item):
            filtered_count += 1
            continue
            
        # 기본 정보 추출
        title = clean_html_tags(item.get('title', ''))
        description = clean_html_tags(item.get('description', ''))
        product_url = item.get('link', '')
        
        # 중복 확인
        if check_duplicate_by_url(product_url):
            skipped_count += 1
            continue
        
        # content 필드 생성
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
        
        # 용량 정보 추출
        storage_info = extract_storage_info(title)
        
        # metadata 필드 구성
        metadata = {
            "title": title,
            "description": description,
            "url": product_url,
            "search_keyword": keyword,
            "source": "naver_shopping",
            "collected_at": datetime.now().isoformat(),
            "embedding_model": "sentence-transformers/all-mpnet-base-v2",
            "price": {
                "lprice": lprice,
                "hprice": hprice
            },
            "product_info": {
                "productId": item.get('productId', ''),
                "productType": item.get('productType', ''),
                "maker": item.get('maker', ''),
                "brand": item.get('brand', ''),
                "category1": item.get('category1', ''),
                "category2": item.get('category2', ''),
                "category3": item.get('category3', ''),
                "category4": item.get('category4', ''),
                "storage": storage_info  # 용량 정보 추가
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
                print(f"✅ 저장완료: {title[:50]}... (가격: {lprice:,}원)" if lprice else f"✅ 저장완료: {title[:50]}...")
            else:
                print(f"❌ 저장실패: {title[:30]}...")
                
        except Exception as e:
            print(f"저장 오류: {e}")
            print(f"실패한 상품: {title[:30]}...")
        
        time.sleep(0.05)
    
    return saved_count, skipped_count, filtered_count

def extract_storage_info(title):
    """제목에서 용량 정보 추출"""
    title_lower = title.lower()
    
    if "128gb" in title_lower or "128" in title_lower:
        return "128GB"
    elif "256gb" in title_lower or "256" in title_lower:
        return "256GB"
    elif "512gb" in title_lower or "512" in title_lower:
        return "512GB"
    elif "1tb" in title_lower:
        return "1TB"
    else:
        return "Unknown"

def get_database_stats():
    """데이터베이스 통계 정보"""
    try:
        total_result = supabase.table('documents').select('id', count='exact').execute()
        total_count = total_result.count if hasattr(total_result, 'count') else len(total_result.data)
        
        shopping_result = supabase.table('documents').select('id', count='exact').eq('metadata->>source', 'naver_shopping').execute()
        shopping_count = shopping_result.count if hasattr(shopping_result, 'count') else len(shopping_result.data)
        
        return total_count, shopping_count
    except Exception as e:
        print(f"통계 조회 오류: {e}")
        return None, None

def main():
    """메인 실행 함수"""
    print(f"🚀 아이폰 15 본체 크롤링 시작: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 환경 변수 확인
    required_vars = [NAVER_CLIENT_ID, NAVER_CLIENT_SECRET, SUPABASE_URL, SUPABASE_KEY]
    if not all(required_vars):
        print("❌ 필수 환경 변수가 설정되지 않았습니다.")
        return
    
    print("✅ 모든 환경 변수 확인 완료")
    
    total_saved = 0
    total_skipped = 0
    total_filtered = 0
    
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
        
        # Supabase에 저장 (필터링 적용)
        saved, skipped, filtered = save_to_supabase(items, keyword)
        total_saved += saved
        total_skipped += skipped
        total_filtered += filtered
        
        print(f"✅ '{keyword}' 처리완료: {saved}개 저장, {skipped}개 중복스킵, {filtered}개 액세서리 제외")
        
        # API 요청 간격
        if i < len(SEARCH_KEYWORDS):
            time.sleep(2)
    
    # 완료 후 데이터베이스 상태
    after_total, after_shopping = get_database_stats()
    
    print(f"\n🎉 아이폰 15 본체 크롤링 완료: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📈 결과 요약:")
    print(f"   - 새로 저장: {total_saved}개 (아이폰 15 본체만)")
    print(f"   - 중복 스킵: {total_skipped}개")
    print(f"   - 액세서리 제외: {total_filtered}개")
    print(f"   - 임베딩 모델: sentence-transformers/all-mpnet-base-v2 (1536차원)")
    
    if after_total is not None and before_total is not None:
        print(f"📊 DB 상태 변화:")
        print(f"   - 전체 문서: {before_total} → {after_total} (+{after_total - before_total})")
        print(f"   - 쇼핑 데이터: {before_shopping} → {after_shopping} (+{after_shopping - before_shopping})")

if __name__ == "__main__":
    main()
