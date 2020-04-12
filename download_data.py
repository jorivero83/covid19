import re
from selenium import webdriver
from datetime import datetime
import pandas as pd
#from worldometers_api.config import config
#import sqlalchemy
from selenium.webdriver.chrome.options import Options


class Epdata:

    def __init__(self,
                 driver_path='/Users/jorge/Documents/covid19/chromedriver',
                 default_download_path='/Users/jorge/Documents/covid19/data'):
        self.default_path = default_download_path
        self.chrome_options = self.chrome_config(default_path=self.default_path)
        self.driver = webdriver.Chrome(executable_path=driver_path)

    def chrome_config(self, default_path):
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--window-size=1920x1080")
        chrome_options.add_argument("--disable-notifications")
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--verbose')
        chrome_options.add_experimental_option("prefs", {
            "download.default_directory": default_path,
            "download.prompt_for_download": False,
            "download.directory_upgrade": True,
            "safebrowsing_for_trusted_sources_enabled": False,
            "safebrowsing.enabled": False
        })
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('--disable-software-rasterizer')
        return chrome_options

    def enable_download_headless(self):
        self.driver.command_executor._commands["send_command"] = ("POST", '/session/$sessionId/chromium/send_command')
        params = {'cmd': 'Page.setDownloadBehavior', 'params': {'behavior': 'allow', 'downloadPath': self.default_path}}
        self.driver.execute("send_command", params)

    def get_data(self, url='https://www.epdata.es/datos/coronavirus-china-datos-graficos/498'):

        # pass url to driver
        self.driver.get(url)

        # function to handle setting up headless download
        self.enable_download_headless()

        try:# document.querySelector("#componente-representacion-botonera-172543 > helper-botonerarepresentacion").shadowRoot.querySelector("div > div.caja.izquierda > div > button:nth-child(2) > font-awesome-generator").shadowRoot.querySelector("#path")
            # document.querySelector("#componente-representacion-botonera-172543 > helper-botonerarepresentacion").shadowRoot.querySelector("div > div.caja.izquierda > div > button:nth-child(2) > span > span.hidden-sm")
            button = self.driver.find_element_by_xpath('//*[@id="componente-representacion-botonera-172543"]/helper-botonerarepresentacion//div/div[1]/div/button[2]/span/span[1]')
            button.click()
            # button2 = self.driver.find_element_by_xpath('//*[@id="contenedordatos"]/button[2]')
            # button2.click()
        except Exception as e:
            print(e)
        finally:
            self.driver.close()

if __name__ == '__main__':
    #parser = WorldometersTable()
    #df = parser.get_data()
    #print(df.head())
    parser = Epdata()
    parser.get_data()

    # todo: try again with some others xpath
