import gym
import pandas as pd
import numpy as np
from gym import spaces
from stable_baselines3 import PPO
import torch
from torch.utils.data import Dataset, DataLoader
from enum import Enum
from tqdm import tqdm
import logging

# device = torch.device("mps:0" if torch.backends.mps.is_available() else "cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CryptoDataset(Dataset):
    def __init__(self, csv_file):
        # header=None 옵션으로 컬럼명 없이 데이터를 불러온다.
        self.data_frame = pd.read_csv(csv_file, header=None)
        # 컬럼명을 수동으로 할당한다.
        self.data_frame.columns = [
            "trade_id",
            "price",
            "qty",
            "quoteQty",
            "time",
            "isBuyerMaker",
            "isBestMatch",
        ]
        self.extract_time_features()
        self.normalize()

    def extract_time_features(self):
        self.data_frame["time"] = pd.to_datetime(self.data_frame["time"], unit="ms")
        self.data_frame["year"] = self.data_frame["time"].dt.year
        self.data_frame["month"] = self.data_frame["time"].dt.month
        self.data_frame["day"] = self.data_frame["time"].dt.day
        self.data_frame["hour"] = self.data_frame["time"].dt.hour
        self.data_frame["minute"] = self.data_frame["time"].dt.minute
        self.data_frame["second"] = self.data_frame["time"].dt.second
        self.data_frame["millisecond"] = self.data_frame["time"].dt.microsecond // 1000
        self.data_frame.drop(columns=["time"], inplace=True)

    def normalize(self):
        numeric_cols = [
            "price",
            "qty",
            "year",
            "month",
            "day",
            "hour",
            "minute",
            "second",
            "millisecond",
        ]
        for col in numeric_cols:
            self.data_frame[col] = self.data_frame[col].astype(float)
            self.data_frame[col] = (
                self.data_frame[col] - self.data_frame[col].min()
            ) / (self.data_frame[col].max() - self.data_frame[col].min())
        self.data_frame["isBuyerMaker"] = self.data_frame["isBuyerMaker"].astype(int)


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
        self.cost = 1  # 기본 비용

        self.action_space = spaces.Discrete(len(Actions))
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(self.window_size, len(self.dataset[0])),
            dtype=np.float32,
        )

        self.trade_fee_bid_percent = 0.005  # 매수 수수료
        self.trade_fee_ask_percent = 0.005  # 매도 수수료
        self.raw_prices = self.dataset.data_frame["price"].to_numpy()

    def reset(self):
        self.current_step = 0
        self.position = Positions.FLAT
        return self._next_observation()

    def _next_observation(self):
        return self.dataset.iloc[
            self.current_step : self.current_step + self.window_size
        ].values

    def step(self, action):
        prev_price = self.dataset.loc[self.current_step, "price"]
        self.current_step += 1
        next_price = self.dataset.loc[self.current_step, "price"]

        self.position, trade_made = self._transform(self.position, action)
        reward = self._calculate_reward(prev_price, next_price, trade_made)

        done = self.current_step > len(self.dataset) - self.window_size
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

    def _calculate_reward(self, action: int) -> np.float32:
        step_reward = 0.0
        current_price = self.raw_prices[self.current_step]
        last_trade_price = self.raw_prices[self.current_step - 1]
        ratio = current_price / last_trade_price
        cost = np.log(
            (1 - self.trade_fee_ask_percent) * (1 - self.trade_fee_bid_percent)
        )

        if action == Actions.BUY and self.position == Positions.SHORT:
            step_reward = np.log(2 - ratio) + cost

        if action == Actions.SELL and self.position == Positions.LONG:
            step_reward = np.log(ratio) + cost

        if action == Actions.DOUBLE_SELL and self.position == Positions.LONG:
            step_reward = np.log(ratio) + cost

        if action == Actions.DOUBLE_BUY and self.position == Positions.SHORT:
            step_reward = np.log(2 - ratio) + cost

        return float(step_reward)


# 데이터셋 로드 및 DataLoader 설정
# crypto_dataset = CryptoDataset("prepare_data/output_file.csv")
crypto_dataset = CryptoDataset("prepare_data/XRPUSDT-trades-2023-11.csv")

data_loader = DataLoader(crypto_dataset, batch_size=1024, shuffle=True)

# 환경 및 에이전트 초기화
env = CryptoTradingEnv(crypto_dataset, window_size=60)
model = PPO("MlpPolicy", env, verbose=1, device=device)

# 학습
total_epochs = 10
logger.info("Training starts...")
for epoch in tqdm(range(total_epochs), desc="Training Progress"):
    for batch in data_loader:
        env.load_batch(batch)
        model.learn(total_timesteps=10000)
    logger.info(f"Epoch {epoch + 1}/{total_epochs} completed.")

# 학습된 모델 저장
model.save("crypto_trading_ppo")
logger.info("Training completed and model saved.")
