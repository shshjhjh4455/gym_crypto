import gym
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
import matplotlib.pyplot as plt
from crypto_env import CryptoTradingEnv
from tqdm import tqdm


class InferenceEnvironment:
    def __init__(self, model_path, csv_file):
        self.env = CryptoTradingEnv(csv_file)
        self.model = PPO.load(model_path)

    def run_inference(self):
        state = self.env.reset()
        done = False
        total_rewards = 0
        actions = []
        rewards = []
        price_changes = []
        successful_buys = []
        successful_sells = []
        asset_value = initial_capital = 1000  # 가정된 초기 자본
        asset_values = [asset_value]

        prev_price = None  # 이전 스텝의 가격

        for _ in tqdm(range(len(self.env.data_frame)), desc="Running Inference"):
            if done:
                break

            action, _ = self.model.predict(state, deterministic=True)
            next_state, reward, done, info = self.env.step(action)

            price_change_origin = next_state[1] if not done else 0
            total_rewards += reward
            actions.append(action)
            rewards.append(reward)
            price_changes.append(price_change_origin)

            # 매수/매도 성공률 계산 (보다 현실적인 계산 방법)
            if prev_price is not None:
                if action == 0:  # 매수
                    successful_buys.append(price_change_origin > prev_price)
                elif action == 1:  # 매도
                    successful_sells.append(price_change_origin < prev_price)

            # 자산 가치 변화 추정 (보다 현실적인 예시)
            if action == 0 or action == 1:  # 매수 또는 매도
                asset_value += reward  # 보상을 자산 가치 변화로 가정
                asset_values.append(asset_value)
            else:
                asset_values.append(asset_value)  # 보류시 자산 가치 유지

            prev_price = price_change_origin
            state = next_state

        self.plot_results(actions, rewards, price_changes)
        self.plot_additional_results(
            successful_buys, successful_sells, asset_values, initial_capital, rewards
        )
        print(f"Total rewards obtained: {total_rewards}")

    def plot_results(self, actions, rewards, price_changes):
        steps = range(len(actions))

        plt.figure(figsize=(12, 8))

        # 가격 변화와 매수/매도 행동 시각화
        plt.subplot(2, 1, 1)
        plt.plot(steps, price_changes, label="Price Change (Original)", color="blue")
        plt.scatter(
            [step for step, action in zip(steps, actions) if action == 0],
            [price_changes[i] for i in range(len(actions)) if actions[i] == 0],
            color="green",
            marker="^",
            label="Buy",
        )
        plt.scatter(
            [step for step, action in zip(steps, actions) if action == 1],
            [price_changes[i] for i in range(len(actions)) if actions[i] == 1],
            color="red",
            marker="v",
            label="Sell",
        )
        plt.title("Price Changes with Buy/Sell Actions")
        plt.xlabel("Step")
        plt.ylabel("Price Change (Original)")
        plt.legend()

        # 누적 보상 시각화
        plt.subplot(2, 1, 2)
        plt.plot(steps, np.cumsum(rewards), label="Cumulative Rewards", color="purple")
        plt.title("Cumulative Rewards Over Time")
        plt.xlabel("Step")
        plt.ylabel("Cumulative Rewards")
        plt.legend()

        plt.tight_layout()
        plt.show()

    def plot_additional_results(
        self, successful_buys, successful_sells, asset_values, initial_capital, rewards
    ):
        plt.figure(figsize=(12, 8))

        # 매수/매도 성공률 그래프
        plt.subplot(3, 1, 1)
        buy_success_rate = (
            sum(successful_buys) / len(successful_buys) if successful_buys else 0
        )
        sell_success_rate = (
            sum(successful_sells) / len(successful_sells) if successful_sells else 0
        )
        plt.bar(
            ["Buy Success Rate", "Sell Success Rate"],
            [buy_success_rate, sell_success_rate],
        )
        plt.ylabel("Success Rate")

        # 슬라이딩 윈도우를 사용한 평균 보상 시각화
        plt.subplot(3, 1, 2)
        window_size = 1000  # 윈도우 크기 조정
        rolling_rewards = pd.Series(rewards).rolling(window=window_size).mean()
        plt.plot(rolling_rewards, label="Rolling Average Reward")
        plt.ylabel("Average Reward")
        plt.legend()

        # 자산 가치 변화 그래프
        plt.subplot(3, 1, 3)
        plt.plot(asset_values, label="Asset Value Over Time")
        plt.hlines(
            initial_capital,
            0,
            len(asset_values) - 1,
            colors="red",
            linestyles="dashed",
            label="Initial Capital",
        )
        plt.ylabel("Asset Value")
        plt.xlabel("Step")
        plt.legend()

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    inference_env = InferenceEnvironment(
        model_path="crypto_trading_ppo.zip",
        csv_file="prepare_data/extracted_files/XRPUSDT-trades-2023-11.csv",
    )
    inference_env.run_inference()
