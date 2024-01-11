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
import matplotlib.pyplot as plt


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
    """암호화폐 트레이딩을 위한 강화학습 환경"""

    def __init__(
        self,
        csv_file,
        initial_balance=10000,
        lookback_window_size=50,
        max_risk_threshold=0.2,
    ):
        super(CryptoTradingEnv, self).__init__()

        # 데이터 불러오기
        self.data_preprocessor = DataPreprocessor(csv_file)
        self.data_preprocessor.preprocess_data()
        self.data_preprocessor.scale_features()
        self.data = self.data_preprocessor.get_processed_data()

        # 환경 매개변수
        self.initial_balance = initial_balance
        self.lookback_window_size = lookback_window_size
        self.current_step = 0
        self.max_risk_threshold = max_risk_threshold

        # 평균 거래량과 평균 시간 차이 계산
        self.avg_qty = self.data["qty"].mean()
        self.avg_time_diff = self.data["time_diff"].mean()

        # 액션 공간 및 상태 공간 정의
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(lookback_window_size, len(self.data.columns)),
            dtype=np.float32,
        )

        # 내부 상태 초기화
        self.balance = initial_balance
        self.portfolio = dict()
        self.trade_history = []

    def reset(self):
        # 초기 잔액 및 포트폴리오 설정
        self.balance = self.initial_balance
        self.portfolio = dict()
        self.current_step = 0
        self.trade_history = []

        # 최초의 관찰 상태 반환
        return self._next_observation()

    def _next_observation(self):
        # 현재 스텝에서 lookback_window_size만큼의 데이터 프레임을 가져옵니다.
        frame = self.data.iloc[
            self.current_step : self.current_step + self.lookback_window_size
        ]

        # lookback_window_size보다 데이터가 적은 경우, 패딩을 추가합니다.
        if len(frame) < self.lookback_window_size:
            padding = [frame.iloc[0]] * (self.lookback_window_size - len(frame))
            frame = pd.concat(padding + [frame], ignore_index=True)

        # 상태는 현재 시점의 시장 데이터를 나타내는 numpy 배열입니다.
        observation = frame.values

        return observation

    def _get_real_price(self, step):
        # step 인덱스에 해당하는 정규화된 가격 데이터를 역변환하여 실제 가격을 반환
        normalized_price = self.data["price"].iloc[step]
        real_price = self.data_preprocessor.scaler.inverse_transform(
            [[normalized_price]]
        )[0, 0]
        return real_price

    def step(self, action):
        # 정규화된 가격 데이터를 실제 가격으로 역변환
        current_price = self._get_real_price(self.current_step)

        # 선택된 행동에 따라 거래 실행 및 수수료 적용
        self._execute_trade_action(action, current_price)

        # 다음 스텝으로 이동
        self.current_step += 1
        done = self.current_step >= len(self.data) - self.lookback_window_size

        # 다음 관찰 상태를 얻음
        next_observation = self._next_observation()

        # 보상 계산
        reward = self._calculate_reward()

        # 거래 로그 기록
        self._log_trade(action, current_price)

        return next_observation, reward, done, {}

    def _calculate_expected_return(self, current_price):
        # 간단한 예시: 고정된 예상 수익률을 반환
        # 실제 사용 시에는 복잡한 모델 또는 분석 기법을 적용하여 미래 가격을 예측하고 수익률을 계산해야 합니다.
        expected_future_price = current_price * 1.01  # 예상되는 1% 가격 상승
        expected_return = (expected_future_price - current_price) / current_price
        return expected_return

    def _evaluate_risk(self):
        # 간단한 예시: 포트폴리오의 변동성을 기반으로 리스크 평가
        # 실제 구현에서는 보다 복잡한 리스크 평가 모델이 필요할 수 있습니다.

        # 현재 포트폴리오 가치 계산
        current_price = self._get_real_price(self.current_step)
        crypto_holding = self.portfolio.get("crypto", 0)
        portfolio_value = self.balance + crypto_holding * current_price

        # 변동성 기반 리스크 평가 (예시)
        recent_prices = self.data["price"].iloc[
            self.current_step - 10 : self.current_step
        ]
        volatility = recent_prices.std()
        risk = volatility / portfolio_value

        return risk

    def _is_loss_exceeding_threshold(self):
        # 손실 한도 설정 (예: 5%)
        loss_threshold = 0.05

        # 포트폴리오 가치 및 구매 가격 대비 현재 가치 계산
        current_price = self._get_real_price(self.current_step)
        crypto_holding = self.portfolio.get("crypto", 0)
        current_value = crypto_holding * current_price
        purchase_value = self.portfolio.get("purchase_value", 0)

        # 손실이 한도를 초과하는지 확인
        return (purchase_value - current_value) / purchase_value > loss_threshold

    def _execute_trade_action(self, action, current_price):
        # 현재 상태에서 추가적인 정보 가져오기
        current_qty = self.data["qty"].iloc[self.current_step]
        current_time_diff = self.data["time_diff"].iloc[self.current_step]
        is_buyer = self.data["isBuyer"].iloc[self.current_step]
        is_maker = self.data["isMaker"].iloc[self.current_step]

        # 시장 활동성 및 매수/매도 경향 분석
        high_market_activity = (
            current_qty > self.avg_qty and current_time_diff < self.avg_time_diff
        )
        buyer_dominant = is_buyer > is_maker  # 매수 주문이 매도 주문보다 우세한 경우

        # 리스크 평가
        risk = self._evaluate_risk()
        expected_return = self._calculate_expected_return(current_price)

        # 리스크가 허용 범위 내이고, 활발한 시장 활동 및 매수 우세 시 거래 실행
        if risk < self.max_risk_threshold and high_market_activity:
            # 매수
            if (
                action == 1 and expected_return > 0.001 and buyer_dominant
            ):  # 수수료보다 높은 수익률 기대 시 매수
                # 매수 금액 결정 (예: 현재 잔액의 일정 비율)
                buy_amount = self.balance * 0.1  # 예시: 잔액의 10%로 매수
                self.balance -= buy_amount
                self.balance -= buy_amount * 0.001  # 수수료 적용
                self.portfolio["crypto"] = self.portfolio.get("crypto", 0) + (
                    buy_amount / current_price
                )
            # 매도
            elif (
                action == 2 and expected_return > 0.001 and not buyer_dominant
            ):  # 수수료보다 높은 수익률 기대 시 매도
                # 포트폴리오의 모든 암호화폐 매도
                sell_amount = self.portfolio.get("crypto", 0)
                self.balance += sell_amount * current_price
                self.balance -= sell_amount * current_price * 0.001  # 수수료 적용
                self.portfolio["crypto"] = 0
            # 보류 및 기타 행동의 경우, 특별한 행동을 취하지 않음

    def _calculate_reward(self):
        # 포트폴리오 가치 계산
        current_price = self._get_real_price(self.current_step)
        crypto_holding = self.portfolio.get("crypto", 0)
        portfolio_value = self.balance + crypto_holding * current_price
        reward = portfolio_value - self.initial_balance
        return reward

    def _log_trade(self, action):
        current_price = self._get_real_price(self.current_step)
        crypto_holding = self.portfolio.get("crypto", 0)
        self.trade_history.append(
            {
                "step": self.current_step,
                "action": action,
                "balance": self.balance,
                "crypto_holding": crypto_holding,
                "crypto_value": crypto_holding * current_price,
                "total_value": self.balance + crypto_holding * current_price,
            }
        )

    def render(self, mode="human"):
        if mode == "human":
            # 현재 스텝, 잔액, 포트폴리오, 최근 거래 내역 등을 출력
            print(f"Step: {self.current_step}")
            print(f"Balance: {self.balance} USD")
            print(f"Crypto Holding: {self.portfolio.get('crypto', 0)} units")
            print(f"Current Price: {self._get_real_price(self.current_step)} USD")

            if self.trade_history:
                print("Recent trade: ", self.trade_history[-1])

        elif mode == "system":
            # 시각화 출력
            plt.figure(figsize=(15, 5))

            # 가격 추세 그래프
            plt.subplot(1, 2, 1)
            plt.plot(self.data["price"][: self.current_step], label="Price")
            for trade in self.trade_history:
                color = "red" if trade["action"] == 1 else "green"
                plt.scatter(
                    trade["step"],
                    self._get_real_price(trade["step"]),
                    color=color,
                    label="Trade Point",
                )
            plt.title("Price Trend")
            plt.xlabel("Step")
            plt.ylabel("Price")
            plt.legend()

            # 포트폴리오 가치 변화
            plt.subplot(1, 2, 2)
            values = [trade["total_value"] for trade in self.trade_history]
            steps = [trade["step"] for trade in self.trade_history]
            plt.plot(steps, values, label="Total Value")
            plt.title("Portfolio Value Over Time")
            plt.xlabel("Step")
            plt.ylabel("Total Value")
            plt.legend()

            plt.show()
