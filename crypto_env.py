import gym
import pandas as pd
import numpy as np
from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
import torch
from torch.utils.data import Dataset
from torch import nn
from enum import Enum
from tqdm import tqdm
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)

# device 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 데이터셋 클래스 정의
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
        # NumPy 배열에서 행 인덱스를 사용하여 데이터 반환
        return self.data_frame[index, :]

    def __len__(self):
        return len(self.data_frame)

    def normalize(self):
        # 가격 및 수량에 대한 차분 계산
        self.data_frame["price_diff"] = self.data_frame["price"].diff().fillna(0)
        self.data_frame["qty_diff"] = self.data_frame["qty"].diff().fillna(0)

        # 차분에 대한 정규화
        for col in ["price_diff", "qty_diff"]:
            max_val = self.data_frame[col].max()
            min_val = self.data_frame[col].min()
            self.data_frame[col] = (self.data_frame[col] - min_val) / (
                max_val - min_val
            )


# 환경 클래스 정의
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
            low=0, high=1, shape=(self.window_size, 6), dtype=np.float32
        )

        self.trade_fee_bid_percent = 0.005  # 매수 수수료
        self.trade_fee_ask_percent = 0.005  # 매도 수수료
        self.last_trade_step = 0  # 마지막 거래가 발생한 스텝 초기화

    def reset(self):
        self.current_step = 0
        self.position = Positions.FLAT
        return self._next_observation()

    def _next_observation(self):
        start = self.current_step
        end = start + self.window_size
        obs = self.dataset.data_frame[start:end]  # Correct slicing for NumPy array
        if len(obs) < self.window_size:
            padding = np.zeros((self.window_size - len(obs), obs.shape[1]))
            obs = np.vstack((obs, padding))
        return obs

    def step(self, action):
        prev_price = self.dataset[self.current_step, 0]
        self.current_step += 1
        next_price = self.dataset[self.current_step, 0]

        self.position, trade_made = self._transform(self.position, action)

        # 거래 수수료 계산
        trade_fee = (
            self.trade_fee_ask_percent + self.trade_fee_bid_percent
        ) * prev_price

        # 롱 포지션과 숏 포지션에 대한 이익 및 손실 계산
        if self.position == Positions.LONG:
            profit = (next_price - prev_price) - trade_fee
        elif self.position == Positions.SHORT:
            profit = (prev_price - next_price) - trade_fee
        else:
            profit = 0

        step_reward = profit

        done = self.current_step >= len(self.dataset) - self.window_size
        obs = self._next_observation()
        return obs, step_reward, done, {}

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
            trade_fee = (
                self.trade_fee_ask_percent + self.trade_fee_bid_percent
            ) * prev_price
            if self.position == Positions.LONG:
                profit = (next_price - prev_price) - trade_fee
            elif self.position == Positions.SHORT:
                profit = (prev_price - next_price) - trade_fee
            else:
                profit = 0

            step_reward = profit

        return step_reward

    def render(self, mode="human"):
        pass

    def close(self):
        pass


# 정책 네트워크 클래스 정의
class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, *args, **kwargs):
        super(CustomActorCriticPolicy, self).__init__(
            observation_space, action_space, lr_schedule, *args, **kwargs
        )

        input_dims = (
            observation_space.shape[0] * observation_space.shape[1]
        )  # window_size * 4
        self.actor = nn.Sequential(
            nn.Linear(input_dims, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, action_space.n),
        )
        self.critic = nn.Sequential(
            nn.Linear(input_dims, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

    def forward(self, obs, deterministic=False):
        x = self.extract_features(obs)
        action_logits = self.actor(x)
        values = self.critic(x)

        action_dist = torch.distributions.Categorical(logits=action_logits)
        if deterministic:
            actions = torch.argmax(action_logits, dim=1)
        else:
            actions = action_dist.sample()

        log_probs = action_dist.log_prob(actions)

        return actions, values, log_probs


# 데이터셋 로드 및 환경 초기화
crypto_dataset = CryptoDataset("prepare_data/XRPUSDT-trades-2023-10.csv")
env = CryptoTradingEnv(crypto_dataset, window_size=10000)

# PPO 모델 초기화
model = PPO(
    CustomActorCriticPolicy,
    env,
    tensorboard_log="./ppo_tensorboard/",
    verbose=1,
    device=device,
    learning_rate=0.0001,  # 학습률 감소
    n_steps=4096,  # 스텝 수 증가
    batch_size=128,  # 배치 크기 증가
    n_epochs=20,  # 에포크 수 증가
    ent_coef=0.01,  # 엔트로피 계수 증가
)


# 학습 루프
total_epochs = 10
for epoch in tqdm(range(total_epochs), desc="Training Progress"):
    logging.info(f"Epoch {epoch + 1}/{total_epochs}")
    model.learn(total_timesteps=20480)


# 학습된 모델 저장
model.save("crypto_trading_ppo")
