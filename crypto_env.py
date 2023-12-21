import gym
import pandas as pd
import numpy as np
from gym import spaces
from sklearn.preprocessing import RobustScaler
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
        self.actions = []  # 에이전트의 행동을 기록할 리스트

        self.current_step = 0
        self.action_space = spaces.Discrete(3)  # 매수, 매도, 보류
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(4,),
            dtype=np.float32,  # price_change, time_diff, cumulative_qty, buyer_maker_ratio
        )

    def step(self, action):
        current_data = self.data_frame.iloc[self.current_step]
        self.current_step += 1
        reward = self.calculate_reward(action, current_data)

        done = self.current_step >= len(self.data_frame)
        next_state = (
            self.data_frame.iloc[self.current_step] if not done else np.zeros(4)
        )
        # 에이전트의 행동을 기록
        self.actions.append(action)

        return next_state, reward, done, {}

    def reset(self):
        self.current_step = 0
        self.actions = []  # 환경이 리셋될 때 행동 기록도 초기화
        return self.data_frame.iloc[self.current_step]

    def render(self, mode="human", close=False):

        window = 50  # 그래프에 표시할 최근 데이터 포인트 수
        start = max(0, self.current_step - window)
        end = self.current_step

        plt.figure(figsize=(15, 8))
        plt.subplot(3, 1, 1)
        plt.title("Price Change and Actions")
        plt.plot(self.data_frame["price_change"][start:end], label="Price Change")
        plt.scatter(
            range(start, end),
            self.data_frame["price_change"][start:end],
            c=[
                "green" if action == 0 else "red" if action == 1 else "blue"
                for action in self.actions[start:end]
            ],
            label="Actions (Buy: Green, Sell: Red, Hold: Blue)",
        )
        plt.legend()

        plt.subplot(3, 1, 2)
        plt.title("Cumulative Quantity")
        plt.plot(
            self.data_frame["cumulative_qty"][start:end], label="Cumulative Quantity"
        )
        plt.legend()

        plt.subplot(3, 1, 3)
        plt.title("Buyer Maker Ratio")
        plt.plot(
            self.data_frame["buyer_maker_ratio"][start:end], label="Buyer Maker Ratio"
        )
        plt.legend()

        plt.tight_layout()
        plt.savefig("crypto_trading_ppo.png")
        plt.show()

    def calculate_reward(self, action, data):
        # 각 특성의 영향력을 결정하는 가중치 설정
        weight_price_change = 0.3
        weight_time_diff = 0.2
        weight_cumulative_qty = 0.4
        weight_buyer_maker_ratio = 0.1

        # 보상 계산
        reward = 0
        reward += weight_price_change * data["price_change"]
        reward += weight_time_diff * data["time_diff"]  # 긴 시간 간격일수록 더 큰 보상
        reward += weight_cumulative_qty * data["cumulative_qty"]  # 높은 거래량일수록 더 큰 보상
        reward += weight_buyer_maker_ratio * (
            2 * data["buyer_maker_ratio"] - 1
        )  # 0.5를 중심으로 -1에서 1 사이의 값

        # 행동에 따른 보상 조정
        if action == 0:  # 매수
            reward *= 1
        elif action == 1:  # 매도
            reward *= -1
        else:  # 보류
            reward *= 0.1  # 보류 시 보상 감소

        return reward


# 환경 및 모델 초기화
if __name__ == "__main__":
    # 로깅 설정
    logging.basicConfig(level=logging.INFO)
    # 환경 초기화
    env = CryptoTradingEnv("prepare_data/extracted_files/XRPUSDT-trades-2023-10.csv")

    # 벡터 환경 만들기
    env = make_vec_env(lambda: env, n_envs=1)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # PPO 모델 초기화
    model = PPO(
        policy=MlpPolicy,
        env=env,
        learning_rate=0.00025,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=None,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        use_sde=False,
        sde_sample_freq=-1,
        target_kl=None,
        tensorboard_log="./ppo_tensorboard/",
        policy_kwargs=None,
        verbose=1,
        seed=None,
        device=device,
        _init_setup_model=True,
    )

    # 모델 학습
    model.learn(total_timesteps=100000)

    # 모델 저장
    model.save("crypto_trading_ppo")
