import gym
import pandas as pd
import numpy as np
from gym import spaces
from sklearn.preprocessing import OneHotEncoder, RobustScaler
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.ppo import MlpPolicy
import torch
import logging


# 데이터 전처리 클래스
class DataPreprocessor:
    def __init__(self, csv_file):
        self.csv_file = csv_file
        self.data_frame = None
        self.scalers = {}  # 각 컬럼에 대한 스케일러를 저장할 딕셔너리

    def preprocess_data(self):
        # 데이터 읽기 및 컬럼 지정
        self.data_frame = pd.read_csv(self.csv_file, header=None)
        self.data_frame.columns = [
            "trade_id",
            "price",
            "qty",
            "quoteQty",
            "time",
            "isBuyerMaker",
            "isBestMatch",
        ]
        # 필요한 컬럼만 선택
        self.data_frame = self.data_frame[["price", "qty", "time", "isBuyerMaker"]]

        # 원핫 인코딩
        ohe = OneHotEncoder(sparse_output=False)
        isBuyerMaker_ohe = ohe.fit_transform(self.data_frame[["isBuyerMaker"]])
        self.data_frame["isBuyer"] = isBuyerMaker_ohe[:, 0]
        self.data_frame["isMaker"] = isBuyerMaker_ohe[:, 1]

        # isBuyerMaker 컬럼 제거
        self.data_frame.drop("isBuyerMaker", axis=1, inplace=True)

        # 시간 차이 컬럼 계산
        self.data_frame["time_diff"] = self.data_frame["time"].diff().fillna(0)

    def scale_features(self):
        if self.data_frame is not None:
            # RobustScaler를 사용한 스케일링
            scaled_columns = ["price", "qty", "time_diff", "time"]
            for col in scaled_columns:
                scaler = RobustScaler()  # 각 컬럼에 대한 스케일러 생성
                self.data_frame[col] = scaler.fit_transform(self.data_frame[[col]])
                self.scalers[col] = scaler  # 스케일러 저장

    def get_processed_data(self):
        return self.data_frame


def main():
    # CSV 파일 경로 설정
    csv_file_path = "prepare_data/extracted_files/XRPUSDT-trades-2023-10.csv"  # 여기에 실제 CSV 파일 경로를 입력하세요.

    # 데이터 전처리 클래스 인스턴스화
    preprocessor = DataPreprocessor(csv_file_path)

    # 데이터 전처리 수행
    preprocessor.preprocess_data()

    # 피쳐 스케일링 수행
    preprocessor.scale_features()

    # 전처리된 데이터 가져오기
    processed_data = preprocessor.get_processed_data()

    # 결과 확인
    print(processed_data.head())  # 처음 몇 행을 출력하여 결과를 확인

    # 저장
    processed_data.to_csv(
        "prepare_data/data_preprocessed/PP-XRPUSDT-trades-2023-10.csv"
    )


if __name__ == "__main__":
    main()
