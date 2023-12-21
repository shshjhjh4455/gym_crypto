import gym
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from crypto_env_PPO import CryptoTradingEnv, CryptoDataset  # 이전에 정의한 클래스를 사용


def load_test_data(csv_file):
    # 테스트 데이터 로딩 함수
    dataset = CryptoDataset(csv_file)
    return np.array([dataset[i] for i in range(len(dataset))])


def evaluate_model(model, env, num_steps=1000):
    obs = env.reset()
    total_rewards = 0
    actions, prices, times, rewards = [], [], [], []

    for step in range(num_steps):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        total_rewards += reward
        actions.append(action)
        prices.append(obs[0][0])  # price 정보는 observation의 첫 번째 요소에 있음
        times.append(obs[0][2])  # time 정보는 observation의 세 번째 요소에 있음
        rewards.append(total_rewards)
        if done:
            break

    return total_rewards, actions, prices, times, rewards


def plot_results(actions, prices, times, rewards, filename):
    fig, axs = plt.subplots(2, 1, figsize=(15, 12))

    # 첫 번째 그래프: 가격 및 매수/매도 포인트
    axs[0].plot(times, prices, label="Price", color="black")
    axs[0].scatter(
        times,
        [prices[i] if a == 3 else None for i, a in enumerate(actions)],
        color="red",
        label="Buy Point",
    )  # 매수 포인트
    axs[0].scatter(
        times,
        [prices[i] if a == 1 else None for i, a in enumerate(actions)],
        color="blue",
        label="Sell Point",
    )  # 매도 포인트
    axs[0].set_title("Price and Trading Points Over Time")
    axs[0].set_xlabel("Time")
    axs[0].set_ylabel("Price")
    axs[0].legend()

    # 두 번째 그래프: 수익률
    axs[1].plot(times, rewards, label="Cumulative Returns", color="purple")
    axs[1].set_title("Cumulative Returns Over Time")
    axs[1].set_xlabel("Time")
    axs[1].set_ylabel("Cumulative Returns")
    axs[1].legend()

    plt.tight_layout()
    plt.savefig(filename)
    plt.show()


def main():
    # 테스트 데이터셋 로드
    test_dataset = load_test_data("prepare_data/XRPUSDT-trades-2023-11.csv")

    # 환경 초기화
    test_env = CryptoTradingEnv(test_dataset, window_size=5000)

    # 모델 로드
    model = PPO.load("crypto_trading_ppo")

    # 모델 평가
    total_rewards, actions, prices, times, rewards = evaluate_model(model, test_env)
    print(f"Total rewards on test data: {total_rewards}")

    # 결과 시각화 및 파일 저장
    plot_results(actions, prices, times, rewards, "trading_results.png")


if __name__ == "__main__":
    main()
