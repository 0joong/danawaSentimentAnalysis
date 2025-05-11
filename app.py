import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from danawaCrawler import DanawaCrawler
from danawaReviewPreprocessing import DanawaReviewPreprocessor
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ✅ 한글 폰트 설정 (윈도우 기준)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 페이지 설정
st.set_page_config(page_title="GPU 리뷰 감성 분석", layout="wide")
st.title("GPU 리뷰 감성 분석 대시보드")

# 사이드바
st.sidebar.title("📊 프로젝트 소개")
st.sidebar.markdown("다나와 GPU 상품의 리뷰를 수집하고 감성 분석을 수행합니다.")
st.sidebar.info("검색어를 입력하면 리뷰 수집 → 전처리 → 감성 분석까지 자동 실행됩니다.")

# 검색어 입력
query = st.text_input("🔍 검색어를 입력하세요", placeholder="예: RTX 4070")
log_box = st.empty()

# 경로
raw_path = "./data/temp/temp.csv"
pre_path = "./data/preprocessedReview.csv"
pred_path = "./data/predictedReview.csv"
model_path = "./model/sentiment_rnn_model.keras"
tokenizer_path = "./model/sentiment_tokenizer.pkl"

# 감성 예측 함수
def predict_sentiment(texts, model, tokenizer, max_len=100):
    sequences = tokenizer.texts_to_sequences(texts)
    X = pad_sequences(sequences, maxlen=max_len)
    preds = model.predict(X, verbose=0)
    labels = ['negative', 'neutral', 'positive']
    results = [labels[np.argmax(p)] for p in preds]
    probs = [float(np.max(p)) for p in preds]
    return results, probs

# 전체 실행 버튼
if st.button("🚀 리뷰 수집 → 전처리 → 감성 분석 시작"):
    if not query.strip():
        st.warning("❗ 검색어를 입력해주세요.")
    else:
        with st.spinner("1️⃣ 리뷰 수집 중입니다..."):
            try:
                crawler = DanawaCrawler(headless=True, logger=log_box.write)
                df_raw = crawler.crawl_top_k_products(query, top_k=5)
                crawler.quit()

                os.makedirs("./data/temp", exist_ok=True)
                df_raw.to_csv(raw_path, index=False, encoding="utf-8-sig")
                st.success(f"✅ 리뷰 수집 완료! ({len(df_raw)}개)")
                st.dataframe(df_raw.head(5))
            except Exception as e:
                st.error(f"❌ 리뷰 수집 오류: {e}")
                st.stop()

        with st.spinner("2️⃣ 리뷰 전처리 중입니다..."):
            try:
                processor = DanawaReviewPreprocessor(logger=log_box.write)
                df_pre = processor.preprocess(raw_path, pre_path)
                st.success(f"✅ 전처리 완료! ({len(df_pre)}개)")
                st.dataframe(df_pre.head(5))
            except Exception as e:
                st.error(f"❌ 전처리 오류: {e}")
                st.stop()

        with st.spinner("3️⃣ 감성 분석 예측 중입니다..."):
            try:
                model = load_model(model_path)
                tokenizer = joblib.load(tokenizer_path)

                if 'tokens' not in df_pre.columns:
                    from konlpy.tag import Okt
                    okt = Okt()
                    df_pre['tokens'] = df_pre['review'].apply(lambda x: ' '.join(okt.morphs(str(x))))

                preds, probs = predict_sentiment(df_pre['tokens'], model, tokenizer)
                df_pre['predicted'] = preds
                df_pre['confidence'] = probs
                df_pre.to_csv(pred_path, index=False, encoding="utf-8-sig")

                st.success("✅ 감성 분석 완료!")
                st.dataframe(df_pre[['review', 'predicted', 'confidence']])

                # 📊 감정 분포 차트
                st.subheader("📊 감정 비율 차트")
                fig, ax = plt.subplots()
                value_counts = df_pre['predicted'].value_counts().reindex(['positive', 'neutral', 'negative']).fillna(0)
                value_counts.plot(kind='bar', color=['green', 'gray', 'red'], ax=ax)
                ax.set_title("감성 분석 결과 분포")
                ax.set_ylabel("리뷰 수")
                ax.set_xticklabels(['긍정', '중립', '부정'], rotation=0)
                st.pyplot(fig)

                # ✅ 감정별 리뷰 수 텍스트로 출력
                st.markdown("#### 감정별 리뷰 개수")
                st.markdown(f"""
                - 긍정: {int(value_counts['positive'])}개  
                - 중립: {int(value_counts['neutral'])}개  
                - 부정: {int(value_counts['negative'])}개
                """)

            except Exception as e:
                st.error(f"❌ 감성 분석 오류: {e}")
