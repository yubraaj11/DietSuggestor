import os
import time
import requests
from selenium import webdriver
from bs4 import BeautifulSoup

def download_meal_image(meal_name, download_folder="static/images"):
    image_path = os.path.join(download_folder, f"{meal_name}.jpg")
    
    if os.path.exists(image_path):
        return image_path
    
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    driver = webdriver.Chrome(options=options)
    
    try:
        search_url = "https://www.foodiesfeed.com/?s=" + meal_name.replace(" ", "+")
        driver.get(search_url)
        
        time.sleep(3)
        
        soup = BeautifulSoup(driver.page_source, "html.parser")
        
        getty_results_div = soup.find("div", {"id": "getty-results"})
        if getty_results_div:
            image_element = getty_results_div.find("img")
            if image_element and "src" in image_element.attrs:
                image_url = image_element["src"]
                
                response = requests.get(image_url, stream=True)
                if response.status_code == 200:
                    os.makedirs(download_folder, exist_ok=True)
                    with open(image_path, "wb") as file:
                        for chunk in response.iter_content(1024):
                            file.write(chunk)
                    return image_path
                else:
                    return "static/img/No Suggestions.jpg"
            else:
                return "static/img/No Suggestions.jpg"
        else:
            return "Target div not found."
    
    finally:
        driver.quit()
