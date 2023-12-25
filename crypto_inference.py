import gym
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
import matplotlib.pyplot as plt
from crypto_env import CryptoTradingEnv
from tqdm import tqdm


class InferenceEnvironment:
    def __init__(self, model_path, csv_file):
        self.env = CryptoTradingEnv(csv_file)  # 원본 CSV 파일 경로 전달
        self.model = PPO.load(model_path)  # 학습된 모델 로드

    def run_inference(self):
        state = self.env.reset()
        done = False
        total_rewards = 0
        actions = []
        rewards = []
        price_changes = []

        for _ in tqdm(range(len(self.env.data_frame)), desc="Running Inference"):
            if done:
                break

            action, _ = self.model.predict(state, deterministic=True)
            next_state, reward, done, info = self.env.step(action)

            if done:
                break

            price_change_origin = next_state[1]  # price_change_origin에 해당하는 인덱스
            total_rewards += reward
            actions.append(action)
            rewards.append(reward)
            price_changes.append(price_change_origin)

            state = next_state

        self.plot_results(actions, rewards, price_changes)
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


if __name__ == "__main__":
    inference_env = InferenceEnvironment(
        model_path="crypto_trading_ppo.zip",
        csv_file="prepare_data/extracted_files/XRPUSDT-trades-2023-11.csv",
    )
    inference_env.run_inference()
