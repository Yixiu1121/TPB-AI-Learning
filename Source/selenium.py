from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time
import os
options = Options()
options.add_argument("--disable-notifications")
abspath = os.path.abspath(r"C:\Program Files\Google\Chrome\Application\chromedriver.exe")
time.sleep(5) 
chrome = webdriver.Chrome(executable_path=abspath, chrome_options=options)
chrome.get("https://fhy.wra.gov.tw/ReservoirPage_2011/StorageCapacity.aspx")
time.sleep(2)  


news = chrome.find_element_by_xpath("//td[text()='石門水庫']")
news = chrome.find_element_by_xpath("//td[@align = 'right']").text
print( news )