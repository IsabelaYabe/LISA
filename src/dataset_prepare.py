import pandas as pd
import spacy
import re
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), 'src')))
from logger import setup_logger

# Essa classe será chamada pelo dataset

logger = setup_logger()
# pure_req_user_stories_path = os.path.join("data", "pure_req_user_stories.csv")
# pure_req_us_df = pd.read_csv(pure_req_user_stories_path)
class DatasetPrepare:
    def __init__(self, csv_file, pattern=r"As (.*?), I want (.*?) so that (.*?)(?:\.|$)", nlp=spacy.load("en_core_web_sm")):
        self._csv_file = csv_file       
        self._pattern = pattern
        self._nlp = nlp
        self._dataframe = self._get_dataframe()

    def _extract_user_story_parts(self, text):
        user_story = []
        user_story_parts = []
        logger.debug(f"Extracting user story parts from text: {text}")
        try:
            matches = list(re.findall(self._pattern, text, re.IGNORECASE))

            num_matches = len(matches)
            for i in range(num_matches):            
                user_story.append(f"As {matches[i][0]}, I want {matches[i][1]} so that {matches[i][2]}.")
                user_story_parts.append({
                    "type of user": "".join(matches[i][0].strip(" ")[1:]),
                    "goal": matches[i][1],
                    "reason": matches[i][2]
                })

            return user_story, user_story_parts
        
        except IndexError:
            logger.error(f"Error extracting user story parts from text (r'As a (.*?), I want (.*?) so that (.*?)(?:\.|$)'): {text}")
            return [], []

    def _get_dataframe(self):
        logger.debug("Starting the dataset preparation process")
        pure_req_us_df = pd.read_csv(self._csv_file)
        
        logger.debug("Renaming columns")
        pure_req_us_df.rename(columns={"databricks-llama-4-maverick": "user story llama-4-maverick", "databricks-meta-llama-3-3-70b-instruct": "user story llama-3-3-70b"}, inplace=True)

        logger.debug("Extracting user story parts llama-4-maverick")

        pure_req_us_df[["user story (us llama-4-maverick)", "user story parts (us llama-4-maverick)"]] = pure_req_us_df.apply(lambda story_list: self._extract_user_story_parts(story_list["user story llama-4-maverick"]), axis=1, result_type="expand")

        logger.debug("Extracting user story parts llama-3-3-70b")
        pure_req_us_df[["user story (us llama-3-3-70b)", "user story parts (us llama-3-3-70b)"]] = pure_req_us_df.apply(lambda story_list: self._extract_user_story_parts(story_list["user story llama-3-3-70b"]), axis=1, result_type="expand")

        return pure_req_us_df
    
    @property
    def dataframe(self): 
        return self._dataframe
    
    @dataframe.setter
    def dataframe(self, value):
        self._dataframe = value

if __name__ == "__main__":
    dataset = DatasetPrepare(os.path.join("data", "pure_req_user_stories.csv"))
    logger.debug("Saving the dataset")
    pure_req_us_df = dataset.dataframe
    #print(pure_req_us_df.head())
    pure_req_us_df.to_csv(os.path.join("data", "pure_req_user_stories_annotateSS.csv"), index=False)