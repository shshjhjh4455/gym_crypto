import gym
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
import matplotlib.pyplot as plt

from crypto_env import CryptoTradingEnv, CryptoDataset  # 환경과 데이터셋 클래스를 가져옵니다.


def load_test_data(csv_file):
    # 테스트 데이터 로딩 함수
    return CryptoDataset(csv_file)


def evaluate_model(model, env, num_steps=1000):
    obs = env.reset()
    total_rewards = 0
    actions, prices, rewards = [], [], []

    for step in range(num_steps):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        total_rewards += reward
        actions.append(action)
        prices.append(obs[0][1])  # price 정보는 observation의 두 번째 요소에 있음
        rewards.append(total_rewards)
        if done:
            break

    return total_rewards, actions, prices, rewards


def plot_results(actions, prices, rewards, filename):
    plt.figure(figsize=(15, 9))

    # 가격 변동 시각화
    plt.subplot(3, 1, 1)
    plt.plot(prices, label="Price")
    plt.title("Price, Actions, and Rewards Over Time")
    plt.ylabel("Price")
    plt.legend()

    # 행동 시각화
    plt.subplot(3, 1, 2)
    plt.step(range(len(actions)), actions, label="Action", where="mid")
    plt.ylabel("Action")
    plt.legend()

    # 누적 보상 시각화
    plt.subplot(3, 1, 3)
    plt.plot(rewards, label="Cumulative Reward")
    plt.xlabel("Step")
    plt.ylabel("Cumulative Reward")
    plt.legend()

    plt.tight_layout()
    plt.savefig(filename)
    plt.show()


def main():
    # 테스트 데이터셋 로드
    test_dataset = load_test_data("prepare_data/XRPUSDT-trades-2023-10.csv")

    # 환경 초기화
    test_env = CryptoTradingEnv(test_dataset, window_size=60)

    # 모델 로드
    model = PPO.load("crypto_trading_ppo")

    # 모델 평가
    total_rewards, actions, prices, rewards = evaluate_model(model, test_env)
    print(f"Total rewards on test data: {total_rewards}")

    # 결과 시각화 및 파일 저장
    plot_results(actions, prices, rewards, "trading_results.png")


if __name__ == "__main__":
    main()
