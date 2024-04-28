from selenium import webdriver
from selenium.webdriver.common.by import By
import time


driver = webdriver.Chrome()


driver.get("https://tver.hh.ru/vacancy/96127386?query=Python-%D1%80%D0%B0%D0%B7%D1%80%D0%B0%D0%B1%D0%BE%D1%82%D1%87%D0%B8%D0%BA&hhtmFrom=vacancy_search_list")

time.sleep(5)


print(driver.find_element(By.XPATH, "/html/body").text)


driver.close()