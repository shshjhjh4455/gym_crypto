import gym
import pandas as pd
import numpy as np
from gym import spaces
from sklearn.preprocessing import RobustScaler

# 데이터 전처리 클래스
class DataPreprocessor:
    def __init__(self, csv_file):
        self.csv_file = csv_file
        self.data_frame = None

    def preprocess_data(self):
        data_frame = pd.read_csv(self.csv_file, header=None)
        data_frame.columns = [
            "trade_id",
            "price",
            "qty",
            "quoteQty",
            "time",
            "isBuyerMaker",
            "isBestMatch",
        ]

        # 가격 변화 감지
        price_change_indices = (
            data_frame["price"].diff().fillna(0).abs().to_numpy().nonzero()[0]
        )

        # 구간별 누적 거래량 및 isBuyerMaker 비율 계산
        cumulative_qty = []
        buyer_maker_ratio = []
        last_idx = 0
        for idx in price_change_indices:
            cumulative_qty.append(data_frame["qty"][last_idx : idx + 1].sum())
            buyer_maker_ratio.append(
                round(data_frame["isBuyerMaker"][last_idx : idx + 1].mean(), 2)
            )

            last_idx = idx + 1

        # 새로운 데이터프레임 생성
        new_data = {
            "price_change": data_frame["price"].iloc[price_change_indices],
            "time_diff": data_frame["time"].diff().iloc[price_change_indices].fillna(0),
            "cumulative_qty": cumulative_qty,
            "buyer_maker_ratio": buyer_maker_ratio,
        }
        self.data_frame = pd.DataFrame(new_data)

    def scale_features(self):
        if self.data_frame is not None:
            # 로그 변환 및 RobustScaler for price_change
            self.data_frame["price_change"] = np.log1p(self.data_frame["price_change"])
            scaler_price = RobustScaler()
            self.data_frame["price_change"] = scaler_price.fit_transform(
                self.data_frame["price_change"].values.reshape(-1, 1)
            )

            scaler_time = RobustScaler()
            scaler_qty = RobustScaler()
            self.data_frame["time_diff"] = scaler_time.fit_transform(
                self.data_frame["time_diff"].values.reshape(-1, 1)
            )
            self.data_frame["cumulative_qty"] = scaler_qty.fit_transform(
                self.data_frame["cumulative_qty"].values.reshape(-1, 1)
            )

    def get_processed_data(self):
        return self.data_frame

# 강화학습 환경 클래스
class CryptoTradingEnv(gym.Env):
    def __init__(self, csv_file):
        super(CryptoTradingEnv, self).__init__()
        self.preprocessor = DataPreprocessor(csv_file)
        self.preprocessor.preprocess_data()
        self.preprocessor.scale_features()
        self.data_frame = self.preprocessor.get_processed_data()

        self.current_step = 0
        self.action_space = spaces.Discrete(3)  # 매수, 매도, 보류
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32  # price_change, time_diff, cumulative_qty, buyer_maker_ratio
        )

    def step(self, action):
        current_data = self.data_frame.iloc[self.current_step]
        self.current_step += 1
        reward = self.calculate_reward(action, current_data)

        done = self.current_step >= len(self.data_frame)
        next_state = self.data_frame.iloc[self.current_step] if not done else np.zeros(4)

        return next_state, reward, done, {}

    def reset(self):
        self.current_step = 0
        return self.data_frame.iloc[self.current_step]

    def render(self, mode='human', close=False):
        # 시각화 로직 (선택적)
        pass

    def calculate_reward(self,action, data):
        # 각 특성의 영향력을 결정하는 가중치 설정
        weight_price_change = 0.3
        weight_time_diff = 0.2
        weight_cumulative_qty = 0.4
        weight_buyer_maker_ratio = 0.1

        # 보상 계산
        reward = 0
        reward += weight_price_change * data['price_change']
        reward += weight_time_diff * (1 - data['time_diff'])  # 짧은 시간 간격일수록 더 큰 보상
        reward += weight_cumulative_qty * data['cumulative_qty']
        reward += weight_buyer_maker_ratio * (2 * data['buyer_maker_ratio'] - 1)  # 0.5를 중심으로 -1에서 1 사이의 값

        # 행동에 따른 보상 조정
        if action == 0:  # 매수
            reward *= 1
        elif action == 1:  # 매도
            reward *= -1
        else:  # 보류
            reward *= 0.1  # 보류 시 보상 감소

        return reward


if __name__ == "__main__":
    env = CryptoTradingEnv("prepare_data/extracted_files/XRPUSDT-trades-2023-10.csv")
