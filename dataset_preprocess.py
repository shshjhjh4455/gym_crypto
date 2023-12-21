import pandas as pd
import numpy as np
import tqdm

def preprocess_data(csv_file):
    data_frame = pd.read_csv(csv_file, header=None)
    data_frame.columns = ["trade_id", "price", "qty", "quoteQty", "time", "isBuyerMaker", "isBestMatch"]

    # 가격 변화 감지
    price_change_indices = data_frame['price'].diff().fillna(0).abs().to_numpy().nonzero()[0]

    # 구간별 누적 거래량 및 isBuyerMaker 비율 계산
    cumulative_qty = []
    buyer_maker_ratio = []
    last_idx = 0
    for idx in tqdm(price_change_indices):
        cumulative_qty.append(data_frame['qty'][last_idx:idx+1].sum())
        buyer_maker_ratio.append(data_frame['isBuyerMaker'][last_idx:idx+1].mean())
        last_idx = idx + 1
        

    # 새로운 데이터프레임 생성
    new_data = {
        'price_change': data_frame['price'].iloc[price_change_indices],
        'time_diff': data_frame['time'].diff().iloc[price_change_indices].fillna(0),
        'cumulative_qty': cumulative_qty,
        'buyer_maker_ratio': buyer_maker_ratio
    }
    processed_data_frame = pd.DataFrame(new_data)

    return processed_data_frame


# 데이터셋 로드 및 전처리
data_frame = preprocess_data("prepare_data/XRPUSDT-trades-2023-10.csv")
# data_frame 저장
data_frame.to_csv("prepare_data/change-XRPUSDT-trades-2023-10.csv")
