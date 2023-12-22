import pandas as pd
from sklearn.preprocessing import RobustScaler
import numpy as np


class DataPreprocessor:
    def __init__(self, csv_file):
        self.csv_file = csv_file
        self.data_frame = None

    def preprocess_data(self):
        data_frame = pd.read_csv(self.csv_file, header=None)
        data_frame.columns = [
            "trade_id",
            "price",
            "qty",
            "quoteQty",
            "time",
            "isBuyerMaker",
            "isBestMatch",
        ]

        # 가격 변화 감지
        price_change_indices = (
            data_frame["price"].diff().fillna(0).abs().to_numpy().nonzero()[0]
        )

        # 구간별 누적 거래량 및 isBuyerMaker 비율 계산
        cumulative_qty = []
        buyer_maker_ratio = []
        last_idx = 0
        for idx in price_change_indices:
            cumulative_qty.append(data_frame["qty"][last_idx : idx + 1].sum())
            buyer_maker_ratio.append(
                round(data_frame["isBuyerMaker"][last_idx : idx + 1].mean(), 2)
            )

            last_idx = idx + 1

        # 새로운 데이터프레임 생성
        new_data = {
            "price_change_origin": data_frame["price"].iloc[price_change_indices],
            "time_diff": data_frame["time"].diff().iloc[price_change_indices].fillna(0),
            "cumulative_qty": cumulative_qty,
            "buyer_maker_ratio": buyer_maker_ratio,
        }
        self.data_frame = pd.DataFrame(new_data)

    def scale_features(self):
        if self.data_frame is not None:
            # 로그 변환 및 RobustScaler for price_change
            self.data_frame["price_change"] = np.log1p(
                self.data_frame["price_change_origin"]
            )
            scaler_price = RobustScaler()
            self.data_frame["price_change"] = scaler_price.fit_transform(
                self.data_frame["price_change"].values.reshape(-1, 1)
            )

            scaler_time = RobustScaler()
            scaler_qty = RobustScaler()
            self.data_frame["time_diff"] = scaler_time.fit_transform(
                self.data_frame["time_diff"].values.reshape(-1, 1)
            )
            self.data_frame["cumulative_qty"] = scaler_qty.fit_transform(
                self.data_frame["cumulative_qty"].values.reshape(-1, 1)
            )

    def get_processed_data(self):
        return self.data_frame


if __name__ == "__main__":
    # 데이터셋 로드 및 전처리
    preprocessor = DataPreprocessor(
        "prepare_data/extracted_files/XRPUSDT-trades-2023-10.csv"
    )
    preprocessor.preprocess_data()
    preprocessor.scale_features()

    # 데이터프레임 저장
    processed_data_frame = preprocessor.get_processed_data()
    processed_data_frame.to_csv(
        "prepare_data/data_preprocessed/PP-XRPUSDT-trades-2023-10.csv"
    )
