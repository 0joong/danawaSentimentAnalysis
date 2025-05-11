import time
import re
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException

class DanawaCrawler:
    def __init__(self, headless=True, logger=None):
        chromedriver_autoinstaller.install()  # ✅ 자동설치

        options = Options()
        if headless:
            options.add_argument("--headless")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-gpu")
        options.add_argument("--remote-debugging-port=9222")
        options.add_argument("user-agent=Mozilla/5.0")

        self.driver = webdriver.Chrome(options=self.options)
        self.wait = WebDriverWait(self.driver, 10)
        self.logger = logger or print

    def log(self, message):
        if self.logger:
            self.logger(message)

    def search_products(self, query, top_k=5):
        self.driver.get("https://www.danawa.com/")
        self.wait.until(EC.element_to_be_clickable((By.ID, "AKCSearch"))).send_keys(query)
        self.wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, "button.search__submit"))).click()
        self.wait.until(EC.presence_of_all_elements_located((
            By.CSS_SELECTOR, "ul.product_list li.prod_item p.prod_name > a"
        )))

        elems = self.driver.find_elements(
            By.CSS_SELECTOR,
            "ul.product_list li.prod_item p.prod_name > a"
        )[:top_k]

        return [(el.text.strip(), el.get_attribute("href")) for el in elems]

    def crawl_reviews(self, product_url):
        self.driver.get(product_url)
        self.wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))
        return self._fetch_all_reviews()

    def crawl_top_k_products(self, query, top_k=5):
        products = self.search_products(query, top_k=top_k)
        all_data = []

        for idx, (name, link) in enumerate(products, 1):
            self.log(f"[{idx}/{top_k}] {name} 리뷰 수집 시작")
            reviews = self.crawl_reviews(link)
            for rv in reviews:
                all_data.append({
                    "product_name": name,
                    "product_link": link,
                    "rating": rv["rating"],
                    "review": rv["review"]
                })

        return pd.DataFrame(all_data, columns=["product_name", "product_link", "rating", "review"])

    def _fetch_all_reviews(self):
        reviews = []

        try:
            tab = self.wait.until(EC.element_to_be_clickable((By.XPATH, "//a[.//h4[text()='쇼핑몰 상품리뷰']]")))
            self.driver.execute_script("arguments[0].click();", tab)
            time.sleep(0.3)
        except TimeoutException:
            return reviews

        try:
            self.wait.until(EC.frame_to_be_available_and_switch_to_it((
                By.CSS_SELECTOR, "iframe[src*='companyProductReview']"
            )))
        except TimeoutException:
            self.driver.switch_to.default_content()

        sel_li = "div#danawa-prodBlog-companyReview-content-list ul.rvw_list > li"
        page = 1

        while True:
            try:
                self.wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, sel_li)))
            except TimeoutException:
                break

            items = self.driver.find_elements(By.CSS_SELECTOR, sel_li)
            for li in items:
                try:
                    rating = li.find_element(
                        By.CSS_SELECTOR,
                        "div.top_info span.point_type_s span.star_mask"
                    ).text.strip()
                except:
                    rating = ""
                try:
                    title = li.find_element(By.CSS_SELECTOR, "div.rvw_atc .tit_W p.tit").text.strip()
                    body = li.find_element(By.CSS_SELECTOR, "div.rvw_atc .atc_cont .atc").text.strip()
                    text = f"{title} {body}".strip()
                except:
                    lines = li.text.splitlines()
                    text = lines[-1].strip() if lines else ""
                reviews.append({"rating": rating, "review": text})

            next_page = page + 1
            xpath_num = f"//div[@id='danawa-prodBlog-companyReview-content-list']//a[normalize-space(text())='{next_page}']"
            try:
                num_btn = self.driver.find_element(By.XPATH, xpath_num)
                self.driver.execute_script("arguments[0].click();", num_btn)
                page = next_page
                time.sleep(0.3)
                continue
            except NoSuchElementException:
                pass

            try:
                arrow = self.driver.find_element(By.CSS_SELECTOR, "span.point_arw_r")
                self.driver.execute_script("arguments[0].click();", arrow)
                time.sleep(0.3)
                num_btn = self.wait.until(EC.element_to_be_clickable((By.XPATH, xpath_num)))
                self.driver.execute_script("arguments[0].click();", num_btn)
                page = next_page
                time.sleep(0.3)
                continue
            except (NoSuchElementException, TimeoutException):
                break

        return reviews

    def quit(self):
        self.driver.quit()
