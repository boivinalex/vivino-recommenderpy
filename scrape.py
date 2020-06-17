# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 09:28:30 2020

@author: Alex Boivin
"""

from selenium.webdriver import Firefox
from selenium.webdriver.firefox.options import Options
# from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
# from selenium.webdriver.common.action_chains import ActionChains
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
import time
import pandas as pd
import re

URL = 'https://www.vivino.com/explore?e=eJwNyUEKgCAQBdDb_LVC21lE3SIiJptESI1RrG6fm7d5UckihkTWIPJLg4H7aBrhOjPuvv6kxhqk8oW8k3INyZeNmyh7QaZDisNTl5XsD-oNGk4='

class wine_data():
    """Scrape wine data and reviews from Vivino."""
    
    def __init__(self,scroll_to_bottom=False):
        """
        Scrape data using selenium and store as a pandas DataFrame.

        Parameters
        ----------
        scroll_to_bottom : TYPE, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        None.

        """
        opts = Options()
        # opts.set_headless()
        self.driver = Firefox(options=opts)
        self.driver.get(URL)
        # wait for page load. timeout after 20 seconds.
        timeout = 20
        try:
            element_present = EC.presence_of_element_located((By.CLASS_NAME, 'vintageTitle__winery--2YoIr'))
            WebDriverWait(self.driver, timeout).until(element_present)
        except TimeoutException:
            print("Timed out waiting for page to load")
        
        self._main_window = self.driver.current_window_handle
        # get number of results
        number_of_results = self.driver.find_element_by_class_name('querySummary__querySummary--39WP2').text
        self.number_of_results = int(re.findall('\d+',number_of_results)[0]) # extract number of results using regular expressions
        print("Found {} wines.".format(self.number_of_results))
        
        self.scroll_to_bottom = scroll_to_bottom
        
        if self.scroll_to_bottom:
            self._infinity_scroll()
                
        self.wine_data = self._get_wine_info()
                
    def _infinity_scroll(self,element=False):
        """Infinite scroll to bottom of page."""
        if element: # scroll the page if no element is provided
            el = element
        else:
            el = self.driver.find_element_by_class_name('inner-page')
        SCROLL_PAUSE_TIME = 0.5 # wait a bit before each scroll
        # Get scroll height
        last_height = self.driver.execute_script("return arguments[0].scrollHeight", el)
        while True:
            # Scroll down to bottom
            if element: #scroll the element
                self.driver.execute_script('arguments[0].scrollTop = arguments[0].scrollHeight', el)
            else: #scroll the window
                self.driver.execute_script("window.scrollTo(0, arguments[0].scrollHeight);", el)
        
            # Wait to load page
            time.sleep(SCROLL_PAUSE_TIME)
        
            # Calculate new scroll height and compare with last scroll height
            new_height = self.driver.execute_script("return arguments[0].scrollHeight", el)
            if new_height == last_height:
                break #break at the bottom
            last_height = new_height
    
    def _get_wine_info(self):
        """
        Iterate through tabs and scrape data.

        Returns
        -------
        wine_data : DataFrame
            DESCRIPTION.

        """
        global discover_wines
        discover_wines = self.driver.find_elements_by_class_name('vintageTitle__winery--2YoIr')
        global dict_list
        dict_list = []
        
        ##TEST
        del discover_wines[0]
        
        # for i, wine in enumerate(discover_wines):
        for wine in discover_wines:
            wine.click()
            self.driver.switch_to.window(self.driver.window_handles[1]) #switch to latest tab (firefox always opens a new tab next to the main tab)
            timeout = 20
            try: #make sure all relevant page items have loaded
                element_present = EC.presence_of_element_located((By.CLASS_NAME, 'anchor__anchor--3DOSm.communityReviews__showAllReviews--1e12c.anchor__vivinoLink--29E1-'))
                WebDriverWait(self.driver, timeout).until(element_present)
            except TimeoutException:
                print("Timed out waiting for tab to load")
            
            # get wine info
            winery_name = self.driver.find_element_by_class_name('winery').text
            wine_name = self.driver.find_element_by_class_name('vintage').text
            wine_country = self.driver.find_element_by_class_name('wineLocationHeader__country--1RcW2').text
            wine_rating = self.driver.find_element_by_class_name('vivinoRatingWide__averageValue--1zL_5').text
            wine_rating_number = self.driver.find_element_by_class_name('vivinoRatingWide__basedOn--s6y0t').text
            wine_price = self.driver.find_element_by_class_name('purchaseAvailabilityPPC__amount--2_4GT').text
            global wine_dict
            wine_dict = {'Name':wine_name,'Winery':winery_name,'Country':wine_country,'Rating':wine_rating,'NumberOfRatings':wine_rating_number,'Price':wine_price}
            dict_list.append(wine_dict)
            
            # get reviews
            review_link = self.driver.find_element_by_class_name('anchor__anchor--3DOSm.communityReviews__showAllReviews--1e12c.anchor__vivinoLink--29E1-')
            review_link.click()
            try: #make sure review popup has loaded
                element_present = EC.presence_of_element_located((By.CLASS_NAME, 'allReviews__reviews--EpUem'))
                WebDriverWait(self.driver, timeout).until(element_present)
            except TimeoutException:
                print("Timed out waiting for tab to load")
            review_pane = self.driver.find_element_by_class_name('baseModal__window--3r5PC.baseModal__themeNoPadding--T_ROG')
            self._infinity_scroll(element=review_pane)
            
            ##TEST
            break
        
            self.driver.switch_to.window(self._main_window)
            time.sleep(1) # pause for 1 second 
        wine_data = pd.DataFrame(dict_list)
            
        return wine_data
        
        
