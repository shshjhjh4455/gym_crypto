import pandas as pd
import os


def combine_csv_files(input_folder, output_file):
    all_files = [
        os.path.join(input_folder, file)
        for file in os.listdir(input_folder)
        if file.endswith(".csv")
    ]
    all_files.sort()  # 파일을 정렬하여 순서대로 처리

    combined_df = pd.DataFrame()

    for file in all_files:
        print(f"Processing {file}...")
        try:
            # 각 파일을 읽을 때 필요한 컬럼만 지정
            chunk = pd.read_csv(file, header=None, usecols=[0, 1, 2, 3, 4, 5])
            chunk.columns = [
                "id",
                "price",
                "qty",
                "base_qty",
                "time",
                "is_buyer_maker",
            ]  # 필요한 컬럼 이름 지정

            # 병합된 DataFrame에 추가
            combined_df = pd.concat([combined_df, chunk], ignore_index=True)

        except Exception as e:
            print(f"Error processing {file}: {e}")
            continue

    # 필요한 경우 병합된 데이터를 정렬
    combined_df = combined_df.sort_values(by="id")

    # 병합된 데이터를 CSV 파일로 저장
    combined_df.to_csv(output_file, index=False)
    print(f"Combined file saved as {output_file}")


def main():
    input_folder = "extracted_files"  # 입력 폴더 경로
    output_file = "output_file.csv"  # 출력 파일 경로
    combine_csv_files(input_folder, output_file)


if __name__ == "__main__":
    main()
