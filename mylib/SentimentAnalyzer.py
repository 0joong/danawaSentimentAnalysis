import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from konlpy.tag import Okt
import numpy as np

class SentimentAnalyzer:
    def __init__(self, tokenizer_file, sa_model_file):
        self.tokenizer = joblib.load(tokenizer_file)
        self.model = load_model(sa_model_file)
        self.ktokenizer = Okt().morphs

    def sentiment_analysis(self, review):
        morphs = [word for word in self.ktokenizer(review)]  # 형태소 분석
        sequences = self.tokenizer.texts_to_sequences([morphs])  # Integer Encoding
        X = pad_sequences(sequences, maxlen=self.model.input_shape[1])  # Padding
        preds = self.model.predict(X)
        label = ['부정', '중립', '긍정']
        max_index = np.argmax(preds[0])
        result = label[max_index]
        return result, preds[0][max_index]
