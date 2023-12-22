# 암호화폐 거래 전략 강화학습 프로젝트

## 프로젝트 개요

이 프로젝트는 암호화폐 거래 데이터를 활용하여 강화학습을 통해 매수 및 매도 전략을 자동으로 학습하는 시스템을 구축하는 것을 목표로 합니다. PPO(Proximal Policy Optimization) 알고리즘을 사용하여, 실시간 거래 데이터에 기반한 최적의 거래 결정을 내리는 모델을 개발합니다.

## 주요 특징

- 체결량 기반의 데이터 전처리 및 특성 추출
- PPO 기반의 강화학습 모델 구현
- 학습 파라미터 조정과 모델 최적화
- 매수 및 매도 시점 결정 로직 개발
- 추가 잔고 관리 로직 설계 고려 중

## 설치 방법

프로젝트를 로컬 시스템에 설치하려면 다음 단계를 따르세요:

```
git clone [https://github.com/shshjhjh4455/gym_crypto]
cd [https://github.com/shshjhjh4455/gym_crypto]
pip install -r requirements.txt
```

## 사용 방법

모델을 학습시키고 테스트하려면 다음 명령을 실행하세요:

```
python crypto_env.py
```
