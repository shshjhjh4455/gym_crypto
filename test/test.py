import gym
import pandas as pd
import numpy as np
from gym import spaces
from stable_baselines3 import PPO
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from enum import Enum
from tqdm import tqdm
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)

# device 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CryptoDataset(Dataset):
    def __init__(self, csv_file):
        self.data_frame = pd.read_csv(csv_file, header=None)
        self.data_frame.columns = [
            "trade_id",
            "price",
            "qty",
            "quoteQty",
            "time",
            "isBuyerMaker",
            "isBestMatch",
        ]
        self.data_frame = self.data_frame[["price", "qty", "time", "isBuyerMaker"]]
        self.normalize()

    def __getitem__(self, index):
        return self.data_frame.iloc[index].to_numpy(dtype=np.float32)

    def __len__(self):
        return len(self.data_frame)

    def normalize(self):
        for col in ["price", "qty"]:
            max_val = self.data_frame[col].max()
            min_val = self.data_frame[col].min()
            self.data_frame[col] = (self.data_frame[col] - min_val) / (
                max_val - min_val
            )
        self.data_frame["isBuyerMaker"] = self.data_frame["isBuyerMaker"].astype(int)
        self.data_frame = self.data_frame.to_numpy(dtype=np.float32)


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


class CryptoTradingEnv(gym.Env):
    def __init__(self, dataset, window_size):
        super(CryptoTradingEnv, self).__init__()
        self.dataset = dataset
        self.window_size = window_size
        self.current_step = 0
        self.position = Positions.FLAT
        self.cost = 10000  # 초기 비용

        self.action_space = spaces.Discrete(len(Actions))
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(self.window_size, 4), dtype=np.float32
        )

        self.trade_fee_bid_percent = 0.005  # 매수 수수료
        self.trade_fee_ask_percent = 0.005  # 매도 수수료

    def reset(self):
        self.current_step = 0
        self.position = Positions.FLAT
        return self._next_observation()

    def _next_observation(self):
        start = self.current_step
        end = start + self.window_size
        obs = self.dataset[start:end]
        if len(obs) < self.window_size:
            padding = np.zeros((self.window_size - len(obs), 4))
            obs = np.vstack((obs, padding))
        return obs

    def step(self, action):
        prev_price = self.dataset[self.current_step, 0]
        self.current_step += 1
        next_price = self.dataset[self.current_step, 0]

        self.position, trade_made = self._transform(self.position, action)
        reward = self._calculate_reward(prev_price, next_price, trade_made)

        done = self.current_step >= len(self.dataset) - self.window_size
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

        return position, trade_made

    def _calculate_reward(self, prev_price, next_price, trade_made):
        step_reward = 0.0

        if trade_made:
            ratio = next_price / prev_price
            cost = np.log(
                (1 - self.trade_fee_ask_percent) * (1 - self.trade_fee_bid_percent)
            )

            # 롱 또는 숏 포지션에 따른 보상 계산
            if self.position == Positions.LONG:
                step_reward = max(0, np.log(ratio)) + cost
            elif self.position == Positions.SHORT:
                step_reward = max(0, np.log(2 - ratio)) + cost

            # isBuyerMaker 필드를 활용하여 메이커 거래에 추가 보상 제공
            is_buyer_maker = self.dataset[self.current_step - 1, 3]
            if is_buyer_maker == 1:  # 메이커 거래인 경우
                additional_reward = 0.01  # 예시 값, 상황에 맞게 조정 가능
                step_reward += additional_reward  # 추가 보상 값

        return step_reward


class CustomActorCriticPolicy(nn.Module):
    def __init__(self):
        super(CustomActorCriticPolicy, self).__init__()
        # 신경망 레이어 정의
        self.actor = nn.Sequential(
            nn.Linear(4 * 60, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, len(Actions)),
        )
        self.critic = nn.Sequential(
            nn.Linear(4 * 60, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

    def forward(self, x):
        # 입력 x를 처리
        x = x.view(x.size(0), -1)  # Flatten
        return self.actor(x), self.critic(x)


# 데이터셋 로드 및 환경 초기화
crypto_dataset = CryptoDataset("prepare_data/XRPUSDT-trades-2023-11.csv")
env = CryptoTradingEnv(crypto_dataset, window_size=60)

# PPO 모델 초기화 및 정책 네트워크 설정
model = PPO(CustomActorCriticPolicy, env, verbose=1, device=device)

# 학습 루프
total_epochs = 10
for epoch in tqdm(range(total_epochs), desc="Training Progress"):
    logging.info(f"Epoch {epoch + 1}/{total_epochs}")
    model.learn(total_timesteps=10000)

# 학습된 모델 저장
model.save("crypto_trading_ppo")
