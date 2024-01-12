import gym
import pandas as pd
import numpy as np
from gym import spaces
from sklearn.preprocessing import OneHotEncoder, RobustScaler
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.evaluation import evaluate_policy
import torch


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


# 강화학습 환경 클래스
class CryptoTradingEnv(gym.Env):
    """암호화폐 트레이딩을 위한 강화학습 환경"""

    def __init__(
        self,
        csv_file,
        initial_balance=10000,
        lookback_window_size=50,
        max_risk_threshold=0.05,
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
            low=self.data.min(axis=0).values,
            high=self.data.max(axis=0).values,
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

    def reset(self):
        # 초기 잔액 및 포트폴리오 설정만 리셋
        self.balance = self.initial_balance
        self.portfolio = dict()
        self.current_step = 0
        self.trade_history = []
        return self._next_observation()

    def _next_observation(self):
        # 현재 스텝부터 lookback_window_size 만큼 데이터 가져오기
        start = max(self.current_step - self.lookback_window_size, 0)
        frame = self.data.iloc[start : self.current_step]

        # 패딩 필요 시 패딩 추가
        if len(frame) < self.lookback_window_size:
            padding = [frame.iloc[0]] * (self.lookback_window_size - len(frame))
            frame = pd.concat(padding + [frame], ignore_index=True)

        return frame.values

    def _get_real_price(self, step):
        # step 인덱스에 해당하는 정규화된 가격 데이터를 역변환하여 실제 가격을 반환
        normalized_price = self.data["price"].iloc[step]
        real_price = self.data_preprocessor.scalers["price"].inverse_transform(
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
        # 모의 시장 분석 또는 예측 모델
        # 예시: 단순 이동 평균을 사용한 추세 예측
        ma_short_term = self.data["price"].rolling(window=5).mean().iloc[-1]
        ma_long_term = self.data["price"].rolling(window=20).mean().iloc[-1]

        if ma_short_term > ma_long_term:
            # 단기 이동 평균이 장기 이동 평균보다 높으면 상승 추세로 예측
            expected_future_price = current_price * 1.02  # 예상되는 2% 가격 상승
        elif ma_short_term < ma_long_term:
            # 단기 이동 평균이 장기 이동 평균보다 낮으면 하락 추세로 예측
            expected_future_price = current_price * 0.98  # 예상되는 2% 가격 하락
        else:
            # 단기 이동 평균과 장기 이동 평균이 같으면 중립적 상태로 예측
            expected_future_price = current_price  # 가격 변화 없음으로 예측

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

    def _is_profit_exceeding_threshold(self):
        # 수익률 목표 설정 (예: 10%)
        profit_threshold = 0.10

        # 현재 가치 및 구매 가격 대비 수익률 계산
        current_price = self._get_real_price(self.current_step)
        crypto_holding = self.portfolio.get("crypto", 0)
        current_value = crypto_holding * current_price
        purchase_value = self.portfolio.get("purchase_value", 0)

        # 수익률 계산
        return (current_value - purchase_value) / purchase_value > profit_threshold

    def _execute_trade_action(self, action, current_price):
        # 매수 및 매도 로직 간소화
        if action == 1:
            self._buy_crypto(current_price)
        elif action == 2:
            self._sell_crypto(current_price)
    
    def _calculate_reward(self):
        # 포트폴리오 가치 계산
        current_price = self._get_real_price(self.current_step)
        crypto_holding = self.portfolio.get("crypto", 0)
        portfolio_value = self.balance + crypto_holding * current_price
        reward = portfolio_value - self.initial_balance
        return reward

    def _log_trade(self, action, current_price):
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


def main():
    # GPU 사용 설정
    device = torch.device("mps:0" if torch.backends.mps.is_available() else "cpu")
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 환경 생성 및 초기화
    env = gym.make("CryptoTradingEnv-v0")
    env = DummyVecEnv([lambda: env])  # 벡터화된 환경 사용
    set_random_seed(0, using_cuda=torch.cuda.is_available())

    # PPO 모델 설정
    model = PPO(
        "MlpPolicy", env, verbose=1, tensorboard_log="./ppo_crypto_trading_tensorboard/"
    )
    model.set_device(device)

    # 학습 실행
    model.learn(total_timesteps=10000)

    # 성능 평가
    mean_reward, std_reward = evaluate_policy(
        model, model.get_env(), n_eval_episodes=10
    )

    # 결과 출력
    print(f"Mean reward: {mean_reward}, std: {std_reward}")

    # 모델 저장
    model.save("ppo_crypto_trading_model")


if __name__ == "__main__":
    main()
