from enum import Enum
import gym
import pandas as pd
import numpy as np
from gym import spaces
from stable_baselines3 import PPO
import math


# 데이터 로딩 및 전처리 함수
def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath)
    df = df[["price", "qty", "time", "isBuyerMaker"]]
    df["price"] = df["price"].astype(float)
    df["qty"] = df["qty"].astype(float)
    df["time"] = pd.to_datetime(df["time"], unit="ms")
    df["isBuyerMaker"] = df["isBuyerMaker"].astype(int)
    return df


# 거래 위치 및 행동 정의
class Positions(int, Enum):
    SHORT = -1
    FLAT = 0
    LONG = 1


class Actions(int, Enum):
    DOUBLE_SELL = 0
    SELL = 1
    HOLD = 2
    BUY = 3
    DOUBLE_BUY = 4


# 암호화폐 거래를 위한 Gym 환경 클래스
class CryptoTradingEnv(gym.Env):
    def __init__(self, df, window_size):
        super(CryptoTradingEnv, self).__init__()
        self.df = df
        self.window_size = window_size
        self.current_step = 0
        self.position = Positions.FLAT
        self.cost = 1  # 기본 비용

        # 행동 공간 정의
        self.action_space = spaces.Discrete(len(Actions))

        # 관찰 공간 정의
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(self.window_size, len(df.columns)), dtype=np.float32
        )

    def reset(self):
        self.current_step = 0
        self.position = Positions.FLAT
        return self._next_observation()

    def _next_observation(self):
        return self.df.iloc[
            self.current_step : self.current_step + self.window_size
        ].values

    def step(self, action):
        prev_price = self.df.loc[self.current_step, "price"]
        self.current_step += 1
        next_price = self.df.loc[self.current_step, "price"]

        self.position, trade_made = self._transform(self.position, action)
        reward = self._calculate_reward(prev_price, next_price, trade_made)

        done = self.current_step > len(self.df) - self.window_size
        obs = self._next_observation()
        return obs, reward, done, {}

    def _transform(self, position, action):
        trade_made = False

        if action == Actions.DOUBLE_SELL:
            if position in [Positions.FLAT, Positions.LONG]:
                position = Positions.SHORT
                trade_made = True

        elif action == Actions.SELL:
            if position == Positions.LONG:
                position = Positions.FLAT
                trade_made = True
            elif position == Positions.FLAT:
                position = Positions.SHORT
                trade_made = True

        elif action == Actions.BUY:
            if position == Positions.SHORT:
                position = Positions.FLAT
                trade_made = True
            elif position == Positions.FLAT:
                position = Positions.LONG
                trade_made = True

        elif action == Actions.DOUBLE_BUY:
            if position in [Positions.FLAT, Positions.SHORT]:
                position = Positions.LONG
                trade_made = True

        # Hold 액션의 경우 위치나 상태는 변하지 않습니다.
        # elif action == Actions.HOLD:
        #     pass

        return position, trade_made

    def _calculate_reward(self, prev_price, next_price, trade_made):
        # 주어진 로직에 따라 보상 계산
        if trade_made:
            if self.position == Positions.LONG:
                return math.log(next_price / prev_price) + math.log(self.cost)
            elif self.position == Positions.SHORT:
                return math.log(2 - next_price / prev_price) + math.log(self.cost)
        return 0

    def render(self, mode="human"):
        pass

    def close(self):
        pass


# 데이터 로드
data = load_and_preprocess_data("prepare_data/output_file.csv")

# 환경 및 에이전트 초기화
env = CryptoTradingEnv(data, window_size=60)
model = PPO("MlpPolicy", env, verbose=1)

# 학습
model.learn(total_timesteps=10000)

# 학습된 모델 저장
model.save("crypto_trading_ppo")
