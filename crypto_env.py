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
            scaler = RobustScaler()
            scaled_columns = ["price", "qty", "time_diff"]
            for col in scaled_columns:
                self.data_frame[col] = scaler.fit_transform(self.data_frame[[col]])

    def get_processed_data(self):
        return self.data_frame


# 강화학습 환경 클래스
class CryptoTradingEnv(gym.Env):
    def __init__(self, csv_file, render_mode=None):
        super(CryptoTradingEnv, self).__init__()
        self.render_mode = render_mode
        self.preprocessor = DataPreprocessor(csv_file)
        self.preprocessor.preprocess_data()
        self.preprocessor.scale_features()
        self.data_frame = self.preprocessor.get_processed_data()
        self.action_history = []  # 에이전트의 행동을 기록할 리스트
        self.last_buy_price = None  # 마지막 매수 가격 저장을 위한 변수

        self.current_step = 0
        self.action_space = spaces.Discrete(3)  # 매수, 매도, 보류
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(5,),
            dtype=np.float32,  # price_change, time_diff, cumulative_qty, buyer_maker_ratio
        )

    def step(self, action):
        if self.current_step >= len(self.data_frame) - 1:
            done = True
            reward = 0
            next_state = np.zeros(5)
        else:
            current_data = self.data_frame.iloc[self.current_step]
            self.current_step += 1
            reward = self.calculate_reward(action, current_data)
            done = self.current_step >= len(self.data_frame) - 1
            next_state = (
                self.data_frame.iloc[self.current_step] if not done else np.zeros(5)
            )

        if not done:
            current_price_change_origin = self.data_frame["price_change_origin"].iloc[
                self.current_step
            ]
            self.action_history.append(
                (self.current_step, action, current_price_change_origin)
            )

        return next_state, reward, done, {}

    def reset(self):
        logging.info("Resetting environment")
        self.current_step = 0
        self.actions = []  # 환경이 리셋될 때 행동 기록도 초기화
        return self.data_frame.iloc[self.current_step]

    # def render(self, mode="human", close=False):
    #     if self.data_frame.empty or self.current_step >= len(self.data_frame):
    #         logging.warning("No data to render or current step out of range.")
    #         return

    #     window = 50
    #     start = max(0, self.current_step - window)
    #     end = min(self.current_step, len(self.data_frame))

    #     plt.figure(figsize=(15, 8))
    #     plt.subplot(2, 1, 1)
    #     plt.title("Price Change with Agent Actions")
    #     plt.plot(self.data_frame["price_change"][start:end], label="Price Change")

    #     for i, action in enumerate(self.actions[start:end]):
    #         action_idx = start + i
    #         if action_idx in self.data_frame.index:  # 인덱스의 유효성 검사
    #             if action == 0:  # 매수
    #                 plt.scatter(
    #                     action_idx,
    #                     self.data_frame["price_change"][action_idx],
    #                     color="green",
    #                     label="Buy",
    #                 )
    #             elif action == 1:  # 매도
    #                 plt.scatter(
    #                     action_idx,
    #                     self.data_frame["price_change"][action_idx],
    #                     color="red",
    #                     label="Sell",
    #                 )

    #     plt.legend()

    #     plt.subplot(3, 1, 2)
    #     plt.title("Cumulative Quantity")
    #     plt.plot(
    #         self.data_frame["cumulative_qty"][start:end], label="Cumulative Quantity"
    #     )
    #     plt.legend()

    #     plt.subplot(3, 1, 3)
    #     plt.title("Buyer Maker Ratio")
    #     plt.plot(
    #         self.data_frame["buyer_maker_ratio"][start:end], label="Buyer Maker Ratio"
    #     )
    #     plt.legend()

    #     plt.tight_layout()
    #     plt.savefig("crypto_trading_ppo.png")
    #     plt.show()
    def render(self, mode="human", close=False):
        pass

    def calculate_reward(self, action, data):
        # 기존 보상 계산 로직
        weight_price_change = 0.3
        weight_time_diff = 0.2
        weight_cumulative_qty = 0.4
        weight_buyer_maker_ratio = 0.1

        reward = 0
        reward += weight_price_change * data["price_change"]
        reward += weight_time_diff * data["time_diff"]
        reward += weight_cumulative_qty * data["cumulative_qty"]
        reward += weight_buyer_maker_ratio * (2 * data["buyer_maker_ratio"] - 1)

        # 매수 행동 시
        if action == 0:  # 매수
            self.last_buy_price = data["price_change_origin"]  # 매수 가격 기록
            reward *= 1  # 기존 매수 보상 로직

        # 매도 행동 시
        elif action == 1:  # 매도
            if self.last_buy_price is not None:
                # 매도 가격이 매수 가격보다 높은 경우 긍정적인 보상
                reward += max(0, data["price_change_origin"] - self.last_buy_price)
            self.last_buy_price = None  # 매도 후 매수 가격 초기화

        # 보류 행동 시
        else:  # 보류
            reward *= 0.1  # 보류 시 보상 감소

        return reward


# 환경 및 모델 초기화
if __name__ == "__main__":
    # 로깅 설정
    logging.basicConfig(level=logging.INFO)

    # 환경 생성
    env = CryptoTradingEnv(
        csv_file="prepare_data/extracted_files/XRPUSDT-trades-2023-10.csv",
        render_mode="human",
    )

    # 벡터 환경 만들기
    vec_env = make_vec_env(lambda: env, n_envs=1)

    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device("mps:0" if torch.backends.mps.is_available() else "cpu")
    print(f"MPS 장치가 사용 가능한지: {torch.backends.mps.is_available()}")

    # PPO 모델 초기화
    model = PPO(
        policy=MlpPolicy,
        env=vec_env,  # 여기서 vec_env를 사용
        learning_rate=0.00025,
        n_steps=2048,
        batch_size=1024,
        n_epochs=100,
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
    model.learn(total_timesteps=1000000)

    # 모델 저장
    model.save("crypto_trading_ppo")

    # # 학습 종료 후 그래프 생성
    # plt.figure(figsize=(15, 8))
    # plt.title("Agent Actions and Market Price Change")
    # plt.plot(
    #     env.data_frame["price_change"], label="Market Price Change"
    # )  # env 대신 vec_env를 사용하지 않음

    # # 매수 및 매도 행동 표시
    # for step, action, price_change in env.action_history:
    #     if action == 0:  # 매수
    #         plt.scatter(step, price_change, color="green", marker="^", label="Buy")
    #     elif action == 1:  # 매도
    #         plt.scatter(step, price_change, color="red", marker="v", label="Sell")

    # plt.xlabel("Step")
    # plt.ylabel("Price Change")
    # plt.legend(loc="best")
    # plt.show()
    # plt.savefig("crypto_trading_ppo_test.png")
