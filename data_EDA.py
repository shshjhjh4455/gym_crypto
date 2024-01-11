import pandas as pd
import matplotlib.pyplot as plt

# 데이터 로드
df = pd.read_csv("prepare_data/extracted_files/XRPUSDT-trades-2023-10.csv", header=None)
df.columns = [
    "trade_id",
    "price",
    "qty",
    "quoteQty",
    "time",
    "isBuyerMaker",
    "isBestMatch",
]

# 동일 가격이 지속되는 행 계산
df["price_change"] = df["price"].diff()
df["same_price"] = df["price_change"] == 0
df["group"] = (df["same_price"] != df["same_price"].shift()).cumsum()
df["same_price_count"] = df.groupby("group")["same_price"].cumsum()

# 연속된 동일 가격 행의 길이에 따라 범주화
bins = [1, 10, 30, 50, 100, float("inf")]
labels = ["1-10", "11-30", "31-50", "51-100", "100+"]
df["category"] = pd.cut(df["same_price_count"], bins=bins, labels=labels, right=False)

# 각 범주별 빈도 계산
category_counts = df["category"].value_counts()

# 원 그래프 시각화
plt.figure(figsize=(8, 8))
plt.pie(
    category_counts, labels=category_counts.index, autopct="%1.1f%%", startangle=140
)
plt.title("Distribution of Same Price Sequences Length")
plt.legend(title="Sequence Lengths", loc="best")
plt.show()

# 평균 연속 행 수 계산
average_count = df[df["same_price"] == True]["same_price_count"].mean()
print("\nAverage length of same price sequences:", average_count)
