import pandas as pd
import re

class DanawaReviewPreprocessor:
    def __init__(self, neutral_threshold=4, logger=print):
        self.neutral_threshold = neutral_threshold
        self.logger = logger

        # 브랜드, 칩셋, 유통사 패턴
        self.manufacturers = ['MSI', '갤럭시', 'ZOTAC', 'PALIT', '이엠텍',
                              'GIGABYTE', 'SAPPHIRE', 'PowerColor', 'ASRock', 'INNO3D']
        self.man_pat = r'(?i)\b(' + '|'.join(self.manufacturers) + r')\b'
        self.chip_pat = r'(?i)\b(rtx\s*\d{3,4}(?:ti)?)\b'
        self.dist_pat = r'\b(제이씨현|대원씨티에스)\b'

    def preprocess(self, input_path: str, output_path: str):
        # 1. load CSV
        df = pd.read_csv(input_path, encoding='utf-8-sig')

        # 2. 제조사 추출
        df['manufacturer'] = df['product_name'].str.extract(self.man_pat, expand=False)

        # 3. 칩셋 추출
        df['chipset'] = (
            df['product_name']
              .str.extract(self.chip_pat, expand=False)
              .str.replace(r'\s+', ' ', regex=True)
              .str.upper()
        )

        # 4. 유통사 추출
        df['distributor'] = df['product_name'].str.extract(self.dist_pat, expand=False)
        df['distributor'] = df['distributor'].fillna('Official')

        # 5. 불필요 컬럼 제거
        if 'product_link' in df.columns:
            df = df.drop(columns=['product_link'])

        # 6. 평점 정규화
        df['rating'] = (
            df['rating']
              .astype(str)
              .str.replace(r'[^0-9.]', '', regex=True)
              .astype(float)
              .div(20)
        )

        # 7. 감성 레이블링
        df['sentiment'] = df['rating'].apply(self._label_sentiment)

        # 8. 저장
        df.to_csv(output_path, index=False, encoding='utf-8-sig')

        # 9. 결과 출력
        sentiment_counts = df['sentiment'].value_counts()
        self.logger(f"✅ 전처리 완료: {output_path}")
        self.logger("📊 감성 분포:")
        for label in ['positive', 'neutral', 'negative']:
            self.logger(f"  {label}: {sentiment_counts.get(label, 0)}")

        return df

    def _label_sentiment(self, r):
        if r > self.neutral_threshold:
            return 'positive'
        elif r < self.neutral_threshold:
            return 'negative'
        else:
            return 'neutral'
