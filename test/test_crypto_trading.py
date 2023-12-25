import gym
import pandas as pd
import numpy as np
from gym import spaces
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
import torch
import logging

# 별도의 모듈에서 CryptoTradingEnv 클래스 임포트
from crypto_env import CryptoTradingEnv


def evaluate_model(test_csv_file, model_path, render=False):
    """
    Evaluate the trained model on a test dataset.
    """
    test_env = CryptoTradingEnv(test_csv_file)
    model = PPO.load(model_path)

    obs = test_env.reset()
    total_rewards = 0
    num_steps = 0
    actions_distribution = {0: 0, 1: 0, 2: 0}  # 매수, 매도, 보류에 대한 행동 분포

    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = test_env.step(action)
        if render:
            test_env.render()

        total_rewards += reward
        actions_distribution[action] += 1
        num_steps += 1

        if done:
            break

    test_env.close()

    # 평가 결과 출력
    avg_reward = total_rewards / num_steps
    print(f"Total rewards: {total_rewards}, Average reward per step: {avg_reward}")
    print(f"Actions distribution: {actions_distribution}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    evaluate_model(
        test_csv_file="prepare_data/extracted_files/XRPUSDT-trades-2023-11.csv",
        model_path="crypto_trading_ppo",
        render=True,
    )
