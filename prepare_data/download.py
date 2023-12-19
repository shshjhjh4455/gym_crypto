import os
import requests
from zipfile import ZipFile
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException


def download_and_extract(url, download_path, extract_path):
    response = requests.get(url)
    zip_path = os.path.join(download_path, url.split("/")[-1])
    with open(zip_path, "wb") as file:
        file.write(response.content)

    with ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_path)


def main():
    target_url = "https://data.binance.vision/?prefix=data/spot/monthly/trades/XRPUSDT/"
    download_path = "downloads"
    extract_path = "extracted_files"

    if not os.path.exists(download_path):
        os.makedirs(download_path)
    if not os.path.exists(extract_path):
        os.makedirs(extract_path)

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))

    try:
        driver.get(target_url)

        # 페이지가 완전히 로드될 때까지 대기
        wait = WebDriverWait(driver, 10)
        try:
            # XPath 수정
            wait.until(
                EC.presence_of_all_elements_located(
                    (
                        By.XPATH,
                        "//td/a[contains(@href, '.zip') and not(contains(@href, '.zip.CHECKSUM')) and substring(@href, string-length(@href) - string-length('.zip') +1) = '.zip']",
                    )
                )
            )
        except TimeoutException:
            print("페이지 로딩 시간 초과")
            return

        # XPath 수정
        elements = driver.find_elements(
            By.XPATH,
            "//td/a[contains(@href, '.zip') and not(contains(@href, '.zip.CHECKSUM')) and substring(@href, string-length(@href) - string-length('.zip') +1) = '.zip']",
        )
        for element in elements:
            download_url = element.get_attribute("href")
            download_and_extract(download_url, download_path, extract_path)

    finally:
        driver.quit()


if __name__ == "__main__":
    main()
