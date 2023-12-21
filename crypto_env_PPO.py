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
import matplotlib.pyplot as plt

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
        start_idx = max(0, index - self.window_size + 1)
        data = self.data_frame[start_idx : index + 1]
        padding = np.zeros((self.window_size - len(data), data.shape[1]))
        return np.vstack((padding, data))

    def __len__(self):
        return len(self.data_frame)

    def normalize(self):
        self.data_frame["price_diff"] = self.data_frame["price"].diff().fillna(0)
        for col in ["price_diff", "qty"]:
            max_val = self.data_frame[col].max()
            min_val = self.data_frame[col].min()
            self.data_frame[col] = (self.data_frame[col] - min_val) / (
                max_val - min_val
            )
        self.data_frame["isBuyerMaker"] = self.data_frame["isBuyerMaker"].astype(int)
        self.data_frame.drop(columns=["price"], inplace=True)
        self.data_frame = self.data_frame.to_numpy(dtype=np.float32)


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
            low=0, high=1, shape=(self.window_size, 3), dtype=np.float32  # 변경: 열 개수 수정
        )

        self.trade_fee_bid_percent = 0.005  # 매수 수수료
        self.trade_fee_ask_percent = 0.005  # 매도 수수료

        self.profit_history = []
        self.position_history = []

    def reset(self):
        self.current_step = 0
        self.position = Positions.FLAT
        self.profit_history = []  # 초기화
        self.position_history = []  # 초기화
        return self._next_observation()

    def _next_observation(self):
        obs = np.array(
            [self.dataset[self.current_step + i] for i in range(self.window_size)]
        )
        if len(obs) < self.window_size:
            padding = np.zeros((self.window_size - len(obs), 3))  # 변경: 열 개수 수정
            obs = np.vstack((obs, padding))
        return obs

    def step(self, action):
        prev_price_diff = self.dataset[self.current_step, 0]
        next_price_diff = self.dataset[
            min(self.current_step + 1, len(self.dataset) - 1), 0
        ]
        self.current_step += 1

        self.position, trade_made = self._transform(self.position, action)

        trade_fee = (
            self.trade_fee_ask_percent + self.trade_fee_bid_percent
        ) * prev_price_diff

        profit = 0
        if self.position == Positions.LONG:
            profit = next_price_diff - trade_fee
        elif self.position == Positions.SHORT:
            profit = prev_price_diff - trade_fee

        self.profit_history.append(profit)
        self.position_history.append(self.position)

        done = self.current_step >= len(self.dataset) - self.window_size
        obs = self._next_observation()
        return obs, profit, done, {}

    def _transform(self, position, action):
        trade_made = False
        if action == Actions.DOUBLE_SELL:
            position = Positions.SHORT
            trade_made = True
        elif action == Actions.SELL:
            if position == Positions.LONG:
                position = Positions.FLAT
            else:
                position = Positions.SHORT
            trade_made = True
        elif action == Actions.BUY:
            if position == Positions.SHORT:
                position = Positions.FLAT
            else:
                position = Positions.LONG
            trade_made = True
        elif action == Actions.DOUBLE_BUY:
            position = Positions.LONG
            trade_made = True
        return position, trade_made

    def render(self) -> None:
        plt.clf()
        plt.xlabel("trading days")
        plt.ylabel("profit")
        plt.plot(self.profit_history)
        plt.savefig("profit.png")

        plt.clf()
        plt.xlabel("trading days")
        plt.ylabel("price difference")
        window_ticks = np.arange(len(self.position_history))
        plt.plot(window_ticks, self.dataset[window_ticks, 0], label="Price Difference")

        short_ticks = [
            i for i, pos in enumerate(self.position_history) if pos == Positions.SHORT
        ]
        long_ticks = [
            i for i, pos in enumerate(self.position_history) if pos == Positions.LONG
        ]
        flat_ticks = [
            i for i, pos in enumerate(self.position_history) if pos == Positions.FLAT
        ]

        plt.plot(
            long_ticks, self.dataset[long_ticks, 0], "g^", markersize=3, label="Long"
        )
        plt.plot(
            flat_ticks, self.dataset[flat_ticks, 0], "bo", markersize=3, label="Flat"
        )
        plt.plot(
            short_ticks, self.dataset[short_ticks, 0], "rv", markersize=3, label="Short"
        )
        plt.legend(loc="upper left", bbox_to_anchor=(0.05, 0.95))
        plt.savefig("price_diff.png")

    def close(self):
        import matplotlib.pyplot as plt

        plt.close()


# 정책 네트워크 클래스 정의
class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, *args, **kwargs):
        super(CustomActorCriticPolicy, self).__init__(
            observation_space, action_space, lr_schedule, *args, **kwargs
        )
        input_dims = observation_space.shape[0] * observation_space.shape[1]
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

# PPO 모델 초기화 및 정책 네트워크 설정
model = PPO(
    CustomActorCriticPolicy,
    env,
    verbose=1,
    tensorboard_log="./ppo_tensorboard/",
    device=device,
    learning_rate=0.00025,  # 학습률 조정
    n_steps=2048,  # 스텝 크기 조정
    batch_size=64,  # 배치 크기 조정
    n_epochs=10,  # 에폭 수 조정
)

# 학습 루프
total_epochs = 10
for epoch in tqdm(range(total_epochs), desc="Training Progress"):
    logging.info(f"Epoch {epoch + 1}/{total_epochs}")
    model.learn(total_timesteps=20480)

# 학습된 모델 저장
model.save("crypto_trading_ppo")
