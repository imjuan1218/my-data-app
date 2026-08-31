import streamlit as st
import pandas as pd
import numpy as np

# 페이지 기본 설정
st.set_page_config(
    page_title="서울 100년 기온 변화 분석",
    page_icon="🌡️",
    layout="wide"
)

# 제목 및 안내 문구
st.title("🌡️ 지난 100년간 서울 연평균 기온 변화")
st.markdown("""
이 앱은 지난 100여 년간 서울의 기온 데이터를 분석하여 연평균 기온의 변화 추이를 보여줍니다.
- **데이터 출처**: [기상청 기후자료](https://raw.githubusercontent.com/greatsong/modudata/main/data/seoul.csv)
- **차트**: Pandas / Streamlit 내장 차트 사용
""")

DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/seoul.csv"

# 데이터 로드 및 캐싱 함수
@st.cache_data
def load_data(url):
    encodings = ['cp949', 'euc-kr', 'utf-8']
    df = None
    for enc in encodings:
        try:
            df = pd.read_csv(url, encoding=enc)
            break
        except Exception:
            continue
    
    if df is None:
        st.error("데이터를 불러오는 데 실패했습니다.")
        return None

    # 열 이름 공백 제거
    df.columns = [col.strip() for col in df.columns]

    # 컬럼 탐색 (날짜, 평균기온)
    date_col = next((c for c in df.columns if '날짜' in c or 'Date' in c or 'date' in c), df.columns[0])
    temp_col = next((c for c in df.columns if '평균기온' in c or 'Mean' in c), df.columns[2])

    # 데이터 전처리
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df['평균기온'] = pd.to_numeric(df[temp_col], errors='coerce')
    
    # 결측치 제거 및 연도 추출
    df = df.dropna(subset=[date_col, '평균기온'])
    df['연도'] = df[date_col].dt.year

    return df

with st.spinner("데이터를 분석하는 중입니다..."):
    raw_df = load_data(DATA_URL)

if raw_df is not None:
    # 연도별 평균 기온 계산
    yearly_df = raw_df.groupby('연도')['평균기온'].mean().reset_index()
    
    # 관측 데이터가 300일 이상 존재하는 유효 연도 필터링
    valid_years = raw_df.groupby('연도').size()
    valid_years = valid_years[valid_years >= 300].index
    yearly_df = yearly_df[yearly_df['연도'].isin(valid_years)].reset_index(drop=True)

    # 사이드바 설정
    min_year = int(yearly_df['연도'].min())
    max_year = int(yearly_df['연도'].max())

    st.sidebar.header("⚙️ 분석 옵션")
    selected_range = st.sidebar.slider(
        "조회 연도 범위 선택",
        min_value=min_year,
        max_value=max_year,
        value=(min_year, max_year)
    )

    show_ma = st.sidebar.checkbox("10년 이동평균선 표시", value=True)
    show_trend = st.sidebar.checkbox("선형 추세선 표시", value=True)

    # 필터링 및 이동평균 계산
    filtered_df = yearly_df[(yearly_df['연도'] >= selected_range[0]) & (yearly_df['연도'] <= selected_range[1])].copy()
    filtered_df['10년 이동평균'] = filtered_df['평균기온'].rolling(window=10, min_periods=1).mean()

    # 선형 추세선 계산
    if show_trend and len(filtered_df) > 1:
        x_vals = filtered_df['연도']
        y_vals = filtered_df['평균기온']
        z = np.polyfit(x_vals, y_vals, 1)
        p = np.poly1d(z)
        filtered_df['선형 추세선'] = p(x_vals)

    # 요약 지표 카드
    col1, col2, col3, col4 = st.columns(4)
    
    start_temp = filtered_df.iloc[0]['평균기온']
    end_temp = filtered_df.iloc[-1]['평균기온']
    temp_diff = end_temp - start_temp

    max_row = filtered_df.loc[filtered_df['평균기온'].idxmax()]
    min_row = filtered_df.loc[filtered_df['평균기온'].idxmin()]

    col1.metric(f"시작 연도 ({int(filtered_df.iloc[0]['연도'])}년)", f"{start_temp:.1f} °C")
    col2.metric(f"최근 연도 ({int(filtered_df.iloc[-1]['연도'])}년)", f"{end_temp:.1f} °C", f"{temp_diff:+.1f} °C")
    col3.metric("최고 연평균 기온", f"{max_row['평균기온']:.1f} °C", f"{int(max_row['연도'])}년")
    col4.metric("최저 연평균 기온", f"{min_row['평균기온']:.1f} °C", f"{int(min_row['연도'])}년")

    st.markdown("---")

    st.subheader(f"📈 서울 연평균 기온 변화 추이 ({selected_range[0]}년 ~ {selected_range[1]}년)")

    # 시각화할 컬럼 선정
    chart_cols = ['평균기온']
    if show_ma:
        chart_cols.append('10년 이동평균')
    if show_trend and '선형 추세선' in filtered_df.columns:
        chart_cols.append('선형 추세선')

    # 연도를 인덱스로 설정하여 Pandas DataFrame 기반 Line Chart 출력
    chart_data = filtered_df.set_index('연도')[chart_cols]
    st.line_chart(chart_data, height=450)

    # 데이터 보기 및 다운로드
    with st.expander("📄 데이터 상세보기 및 CSV 다운로드"):
        st.dataframe(
            filtered_df.rename(columns={'평균기온': '연평균 기온(°C)', '10년 이동평균': '10년 이동평균(°C)'}),
            use_container_width=True
        )
        csv = filtered_df.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            label="📥 CSV 데이터 다운로드",
            data=csv,
            file_name=f'seoul_temp_{selected_range[0]}_{selected_range[1]}.csv',
            mime='text/csv'
        )
