# -*- coding: utf-8 -*-
import streamlit as st
import urllib.request
import urllib.parse
import json
import pandas as pd
from datetime import datetime
import base64
from io import BytesIO

class NaverApiClient:
    def __init__(self, client_id, client_secret):  # 클라이언트 아이디와 시크릿키를 받아서 초기화하는 함수
        self.client_id = client_id
        self.client_secret = client_secret
        self.base_url = "https://openapi.naver.com/v1/search/"
    
    def get_data(self, media, count, query, start=1, sort="date"):
        """
        네이버 API에서 데이터를 가져오는 메소드
        
        Parameters:
        - media: 검색 미디어 타입 (news, blog, image 등)
        - count: 검색 결과 개수
        - query: 검색어
        - start: 검색 시작 위치 (페이징용)
        - sort: 정렬 방식 (date, sim 등)
        """
        encText = urllib.parse.quote(query) # 검색어 인코딩
        url = f"{self.base_url}{media}?sort={sort}&display={count}&start={start}&query={encText}"
        
        # 요청
        request = urllib.request.Request(url)
        request.add_header("X-Naver-Client-Id", self.client_id)
        request.add_header("X-Naver-Client-Secret", self.client_secret)
        
        # 응답
        try:
            response = urllib.request.urlopen(request)
            rescode = response.getcode()  # 응답 코드 확인 
            
            if(rescode==200):  # 정상 응답이면 
                response_body = response.read()  # 응답 본문 읽기 
                result = response_body.decode('utf-8')  # utf-8 로 디코딩 한글로 예쁘게 출력 
                return result
            else:
                st.error(f"Error Code: {rescode}")
                return None
        except Exception as e:
            st.error(f"Exception occurred: {e}")
            return None
    
    def get_news(self, query, count=10, start=1, sort="date"):
        """뉴스 검색 결과를 가져오는 편의 메소드"""
        return self.get_data("news", count, query, start, sort)
    
    def get_blog(self, query, count=10, start=1, sort="date"):
        """블로그 검색 결과를 가져오는 편의 메소드"""
        return self.get_data("blog", count, query, start, sort)
    
    def get_image(self, query, count=10, start=1, sort="sim"):
        """이미지 검색 결과를 가져오는 편의 메소드"""
        return self.get_data("image", count, query, start, sort)
    
    def get_shop(self, query, count=10, start=1, sort="date"):
        """쇼핑 검색 결과를 가져오는 편의 메소드"""
        return self.get_data("shop", count, query, start, sort)
    
    def parse_json(self, data):  # json 파일을 dictionary 형태로 변환하는 함수
        """API 응답을 JSON으로 파싱하는 메소드"""
        if data:
            return json.loads(data)
        return None

#  csv 파일 다운로드 링크 생성 함수 
def get_csv_download_link(df, filename):
    """
    데이터프레임을 CSV로 변환하고 다운로드 링크 생성
    """
    csv = df.to_csv(index=False, encoding='utf-8-sig') # 판다스 데이터 프레임을 csv 파일로 변환 
    b64 = base64.b64encode(csv.encode('utf-8-sig')).decode()  # csv 파일을 base64 인코딩 
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">CSV 파일 다운로드</a>' # 링크 생성 
    return href

# 화면 개발 
def main():
    st.set_page_config(page_title="네이버 API 검색", layout="wide") # layout의 형태를 wide 로 설정정
    
    # 제목 및 설명
    st.title("노동자로 살건인가? 자본가로 살것인가? 네이버 검색 API")
    st.markdown("---")
    
    # API 키 입력 (기본값 설정, 실제 앱에서는 숨기는 것이 좋음)
    with st.sidebar: # 사이드바 화면 개발 
        st.header("네이버 검색 API 설정")
        client_id = st.text_input("Client ID", value="9XhhxLV1IzDpTZagoBr1")
        client_secret = st.text_input("Client Secret", value="J14HFxv3B6", type="password")
    
    # 네이버 API 클라이언트 생성
    naver_client = NaverApiClient(client_id, client_secret)
    
    # 검색 설정 UI
    col1, col2 = st.columns(2) # 두 개의 열을 생성하여 화면을 나누어 줌
    
    with col1:
        search_type = st.selectbox(
            "검색 타입:", 
            options=[("뉴스", "news"), ("블로그", "blog"), ("이미지", "image"),("쇼핑","shop")],
            format_func=lambda x: x[0] # 옵션의 첫번째 요소 화면에 표시(뉴스,블로그,이미지,쇼핑)
        )
        search_type = search_type[1]  # news, blog, image, shop 중 하나를 선택하여 실제값 추출
        
        query = st.text_input("검색어:", value="안성탕면")
        count = st.slider("결과 수:", min_value=1, max_value=100, value=50)
    
    with col2: # 두번째 열의 검색 옵션을 설정
        sort_options = st.selectbox(
            "정렬:", 
            options=[("최신순", "date"), ("정확도순", "sim")],
            format_func=lambda x: x[0] # 옵션의 첫번째 요소를 화면에 표시(최신순, 정확도순)
        )
        sort_options = sort_options[1]  # date, sim 중 하나를 선택하여 sort_options 에 할당
        
        start_page = st.slider("시작 위치:", min_value=1, max_value=100, value=1)
    
    # 검색 버튼
    if st.button("검색", type="primary"): # type="primary" ? 강조된 스타일의 버튼, secondary ? 일반 버튼
        st.session_state.search_clicked = True # 검색 버튼 클릭 시 세션 상태를 True 로 설정
        
        with st.spinner(f"'{query}' 검색 중... 잠시만 기다려주세요"):
            # 검색 타입에 따라 적절한 메소드 호출
            if search_type == 'news':
                data = naver_client.get_news(query, count, start_page, sort_options)
            elif search_type == 'blog':
                data = naver_client.get_blog(query, count, start_page, sort_options)
            elif search_type == 'shop':
                data = naver_client.get_shop(query, count, start_page, sort_options)
            else:  # image
                data = naver_client.get_image(query, count, start_page, sort_options)
            
            # 결과 파싱 및 표시(json 파일을 dictionary 형태로 변환)
            parsed_data = naver_client.parse_json(data)
            
            if parsed_data:
                # 결과 저장
                st.session_state.search_results = parsed_data # 검색 결과를 세션 상태에 저장
                
                # 결과 표시 (검색 타입에 따라 다르게)
                st.subheader(f"검색 결과 (총 {parsed_data['total']}개 중 {len(parsed_data['items'])}개 표시)")
                
                if search_type == 'image': # 이미지 검색 결과일 경우 
                    # 이미지 그리드 형태로 표시
                    image_cols = 4
                    for i in range(0, len(parsed_data['items']), image_cols):
                        cols = st.columns(image_cols)
                        for j in range(image_cols):
                            if i+j < len(parsed_data['items']):
                                item = parsed_data['items'][i+j]
                                with cols[j]:
                                    st.image(item['thumbnail'], use_container_width=True)
                                    st.markdown(item['title'].replace("<b>", "").replace("</b>", ""))
                                    st.markdown(f"[원본 링크]({item['link']})")
                else:
                    # 뉴스나 블로그, 쇼핑은 테이블 형태로 표시
                    df = pd.DataFrame(parsed_data['items'])
                    # 불필요한 HTML 태그 제거
                    for col in ['title', 'description']:
                        if col in df.columns:
                            df[col] = df[col].str.replace('<b>', '').str.replace('</b>', '').str.replace('&quot;', '"')
                    
                    # 필요한 열만 선택하여 표시
                    if search_type == 'news':
                        display_cols = ['title', 'description', 'pubDate', 'link']
                    elif search_type == 'shop':
                        display_cols = ['title','link','image','lprice','hprice','mallname','productname']
                    else:  # blog
                        display_cols = ['title', 'description', 'postdate', 'link', 'bloggername']
                    
                    # 가능한 열만 선택
                    display_cols = [col for col in display_cols if col in df.columns]
                    
                    # 테이블 표시
                    st.dataframe(df[display_cols], use_container_width=True)
                    
                    # 파일 내보내기 (타임스탬프 생성)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    
                    # 검색 타입에 따라 파일명 접두어 설정
                    type_prefix = {
                        'news': 'news',
                        'blog': 'blog',
                        'shop': 'shopping',
                        'image': 'image'
                    }.get(search_type, 'naver')
                    
                    base_filename = f"{type_prefix}_{timestamp}"

                    # 버튼을 나란히 배치하기 위한 컬럼 생성
                    col_export1, col_export2 = st.columns(2)

                    with col_export1:
                        # CSV 다운로드 버튼
                        csv_filename = f"{base_filename}.csv"
                        st.download_button(
                            label="CSV 내보내기",
                            data=df.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig'),
                            file_name=csv_filename,
                            mime='text/csv',
                        )

                    with col_export2:
                        # JSON 다운로드 버튼
                        json_filename = f"{base_filename}.json"
                        # 데이터프레임을 JSON으로 변환 (한글 인코딩 처리)
                        json_data = df.to_json(orient='records', force_ascii=False, indent=4)
                        st.download_button(
                            label="JSON 내보내기",
                            data=json_data,
                            file_name=json_filename,
                            mime='application/json',
                        )
            else:
                st.error("검색 결과가 없거나 오류가 발생했습니다.")

if __name__ == "__main__":
    main()

