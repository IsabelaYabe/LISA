from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.options import Options 
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
from lisa.sub_lisa.logger import logger
import os
import time

from dataclasses import dataclass, field 
from typing import List

@dataclass
class ResearchPaper:
    title: str
    year: int
    abstract: str
    publisher: str    
    DOI: str
    source_url: str
    authors: List[str] = field(default_factory=list)
    
@dataclass
class Research:
    num_results: int # ok
    keywords: str # ok
    years: str # ok
    publisher: str 
    url: str
    content_type: List[str] = field(default_factory=list) # ok
    papers: List[ResearchPaper] = field(default_factory=list)

class WebDriverConfig:
    def __init__(self, wait_xpath, limit_showing=25, wait_time=10, headless=True):
        self._headless = headless
        self._chrome_options = None
        self._driver = None
        self._wait_time = wait_time
        self._wait_xpath = wait_xpath
        self._driver_wait = WebDriverWait(self.driver, self.wait_time)
        self._limit_showing = limit_showing
        
    def setup_chrome_options(self):
        os.environ['WDM_LOG_LEVEL'] = '0'
        chrome_options = Options()
        if self.headless:
            chrome_options.add_argument("--headless=new")
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--user-agent=Mozilla/5.0 (...) Safari/537.36")

        return chrome_options
    
    def initialize_driver(self):
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=self.chrome_options)
        return driver
    
    def load_url(self, url):
        self.driver.get(url)
        if self.wait_xpath:
            self.driver_wait.until(EC.presence_of_element_located((By.XPATH, self.wait_xpath)))
        
    @property
    def chrome_options(self):
        if self._chrome_options is None:
            self._chrome_options = self.setup_chrome_options()
        return self._chrome_options
    
    @property
    def driver(self):
        if self._driver is None: 
            self._driver = self.initialize_driver()
        return self._driver 
    
    @property
    def  wait_time(self):
        return self._wait_time

    @wait_time.setter
    def wait_time(self, value):
        if isinstance(value, int) and value > 0:
            self._wait_time = value
            self._driver_wait = WebDriverWait(self.driver, self.wait_time)
        else:
            raise ValueError("wait_time must be a positive integer")
    
    @property
    def wait_xpath(self):
        return self._wait_xpath
    
    @wait_xpath.setter
    def wait_xpath(self, new_xpath):
        if isinstance(new_xpath, str) and new_xpath:
            self._wait_xpath = new_xpath
        else:
            raise ValueError("wait_xpath must be a non-empty string")
    
    @property
    def headless(self):
        return self._headless
    
    @property
    def driver_wait(self):
        return self._driver_wait    
    
    @property
    def showing_limit(self):
        return self._limit_showing
        
class IEEESources(WebDriverConfig):
    """
    Recebe uma url do IEEEXplore com a pesquisa filtrada
    """
    def  __init__(self, wait_xpath="//div[contains(@class, 'personal-login-header')]"):
        super().__init__(wait_xpath=wait_xpath)
        self.limit_showing_per_page=25        
        
    def research_datas(self):
        xpath = "//div[contains(@class, 'Dashboard-section') and contains(@class, 'Dashboard-section-gray') and contains(@class, 'text-base-md-lh')]"
        div = self.driver.find_elements(By.XPATH, xpath)
        
        if len(div) != 1:
            raise ValueError(f"Expected exactly 1 matching div, but found {len(div)}.")
        div = div[0]
        
        split_text = div.text.split("\n")
        num_of_results_and_keywords = split_text[0].split(" ")
        years = split_text[1].split(" ")
        content_type = split_text[2:]
            
        research = {
            "num_results": num_of_results_and_keywords[3],  
            "keywords": " ".join(num_of_results_and_keywords[5:]),
            "years": " ".join(years[2:]),  
            "content_type": " ".join(content_type) 
        }
        logger.debug(f"reaserch: {research}")
        
        return research
    
    def get_title(self, url):
        # <h1 _ngcontent-ng-c1019330821="" class="document-title text-2xl-md-lh"><span _ngcontent-ng-c1019330821="">Automated Classification and Identification of Non-Functional Requirements in Agile-Based Requirements Using Pre-Trained Language Models</span></h1>
        pass
    
    def get_authors(self):
        # <div _ngcontent-ng-c1019330821="" class="document-main-author-banner"><div _ngcontent-ng-c1019330821="" class="document-authors-banner stats-document-authors-banner">
            # <div _ngcontent-ng-c1131135293="" class="authors-info-container overflow-ellipsis text-base-md-lh authors-minimized" id="indexTerms-container-1748305523957-0">
                # for  <span _ngcontent-ng-c1131135293="" class="authors-info"> in div 
                    # get <span _ngcontent-ng-c1131135293="">Abdulrahim Alhaizaey</span> pegar o nome Abdulrahim Alhaizaey
        pass
    
    def get_abstract(self): 
        # <div _ngcontent-ng-c1937908680="" class="u-mb-1">
            # <div _ngcontent-ng-c1937908680="" xplmathjax="">Non-functional requirements (NFRs) are critical factors for software quality and success. A frequently reported challenge in agile requirements engineering is that NFRs are often neglected due to the focus on functional requirements (FRs) and the limited capability of agile requirements documented as user stories to represent NFRs. With the emergence of transfer learning and large pre-trained language models, various applications in requirements engineering have become feasible, alleviating several longstanding challenges. This study evaluates transformer-based models for the automated identification and classification of NFRs. We leveraged transfer learning with pre-trained transformer models to automate the identification and classification of NFRs in agile textual requirements documented as user stories. A dataset of over 10k user stories was collected and labeled, and pre-trained transformer models, including BERT, RoBERTa, XLNet, and DistilBERT, were fine-tuned to automate the identification of NFRs. We incorporated Focal Loss during training to mitigate the dominance of functionally driven requirements and class imbalances. In addition, thorough experiments on hyperparameter optimization were employed using Bayesian hyperparameter optimization to obtain the combination of hyperparameters that best correlated with the aim of enhancing each model’s performance. Our evaluation demonstrated that the finetuned pre-trained models significantly outperformed comparable prior approaches relying on rule-based techniques or traditional machine learning, with a fine-tuned BERT model achieving an F1 Score of 93.4 %. These findings highlight the potential of pre-trained language models in agile requirements engineering, enabling more efficient NFRs identification, reducing manual review burden, and facilitating a viable and efficient approach to address the neglect of NFRs in agile development processes.</div>
        pass
    
    def go_to_research_metadatas(self):
        # <div _ngcontent-ng-c1937908680="" class="row g-0 u-pt-1"><div _ngcontent-ng-c1937908680="" class="col-6">
        pass 
    
    def get_date_of_publication(self):
        # <div _ngcontent-ng-c1937908680="" class="u-pb-1 doc-abstract-pubdate"><strong _ngcontent-ng-c1937908680="">Date of Publication:</strong> 15 May 2025 <xpl-help-link _ngcontent-ng-c1937908680="" arialabel="Get help with using Publication Dates" helplinktext="Help with using Publication Dates" helplink="http://ieeexplore.ieee.org/Xplorehelp/Help_Pubdates.html" _nghost-ng-c2622260443="">
        pass
        
    def get_doi(self):
        # <div _ngcontent-ng-c1937908680="" class="u-pb-1 stats-document-abstract-doi"><strong _ngcontent-ng-c1937908680="">DOI: </strong><a _ngcontent-ng-c1937908680="" append-to-href="?src=document" target="_blank" href="https://doi.org/10.1109/ACCESS.2025.3570359">10.1109/ACCESS.2025.3570359</a>
        pass
    
    def get_publishier(self):
        # <div _ngcontent-ng-c1937908680="" class="u-pb-1 doc-abstract-publisher"><xpl-publisher _ngcontent-ng-c1937908680="" _nghost-ng-c2723103032=""><span _ngcontent-ng-c2723103032="" class="text-base-md-lh publisher-info-container black-tooltip"><button _ngcontent-ng-c2723103032="" xplhighlight=""><span _ngcontent-ng-c2723103032="" class="title">Publisher: </span><!----><span _ngcontent-ng-c2723103032="">IEEE</span>
        pass
    
    def get_searches_founds(self):
        # CASO 1,2,3... to desktop
        # Ir em <xpl-paginator _ngcontent-ng-c540643953="" _nghost-ng-c2457837566="">
            # <div _ngcontent-ng-c2457837566="" class="pagination-bar hide-mobile text-base-md-lh">
                # achar isso <ul _ngcontent-ng-c2457837566="" class="my-3">
                    # achar isso <li _ngcontent-ng-c2457837566="" class="next-btn">
                        # clicar nesse botão <button _ngcontent-ng-c2457837566="" class="stats-Pagination_arrow_next_2"> &gt; </button>
                    
        # Encontrar <xpl-results-list _ngcontent-ng-c540643953="" _nghost-ng-c277760274="">
            # Dentro dessa parte encontrar as div com <div _ngcontent-ng-c277760274="" class="List-results-items" id="11005451"> só o id muda de um a pra outra
                # ir par aessa div <xpl-results-item _ngcontent-ng-c277760274="" _nghost-ng-c3551078498="">
                    # ir apra essa div <div _ngcontent-ng-c3551078498="" class="hide-mobile">
                        # acessar o html dessa ancora <a _ngcontent-ng-c3551078498="" xplanchortagroutinghandler="" xplhighlight="" xplmathjax="" class="fw-bold" href="/document/11005451/">
                            # <div _ngcontent-ng-c3551078498="" class="d-flex result-item">
                                # <div _ngcontent-ng-c3551078498="" class="d-flex result-item">
                                    # <div _ngcontent-ng-c3551078498="" class="col result-item-align px-3">
                                        # <h3 _ngcontent-ng-c3551078498="" class="text-md-md-lh">
                                            # <a _ngcontent-ng-c3551078498="" xplanchortagroutinghandler="" xplhighlight="" xplmathjax="" class="fw-bold" href="/document/11005451/">
        xpath = "//xpl-results-list"
        researches = self.driver.find_elements(By.XPATH, xpath)
        for research in researches:
            logger.debug(f"Reaserch founded: {research}")

# 540643953
# <xpl-results-list _ngcontent-ng-c540643953="" _nghost-ng-c277760274="">

#<div _ngcontent-ng-c277760274="" class="List-results-items" id="10009796">

if __name__ == "__main__":
    try: 
        url = "https://ieeexplore.ieee.org/search/searchresult.jsp?action=search&newsearch=true&matchBoolean=true&queryText=(%22Full%20Text%20.AND.%20Metadata%22:requirements%20elicitation)%20AND%20(%22All%20Metadata%22:language%20model)%20AND%20(%22Abstract%22:agile)&highlight=true&returnType=SEARCH&matchPubs=true&pageNumber=1&ranges=2020_2025_Year&returnFacets=ALL"

        ieee = IEEESources()
        ieee.load_url(url)

        ieee.get_searches_founds()

    except Exception as e:
        logger.error(f"Erro ao executar scraping: {e}")