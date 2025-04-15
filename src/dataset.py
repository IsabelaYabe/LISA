import pandas as pd
import os
import spacy
import re
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), 'src')))
from logger import setup_logger

logger = setup_logger()
class dataset:
    def __init__(self, documents, usage_scenarios, requirements, user_stories):
        self.documents = documents
        self.usage_scenarios = usage_scenarios
        self.requirements = requirements
        self.user_stories = user_stories

    def extract_subkeys(key):
        key_split = key.split(".")
        key_0 = ".".join(key_split[:2])
        key_1 = key_split[2]
        key_2 = ".".join(key_split[3:])

        return key_0, key_1, key_2

if __name__ == "__main__":
    logger.debug("Starting the dataset preparation process")
    pure_req_user_stories_path = os.path.join("data", "pure_req_user_stories.csv")
    pure_req_us_df = pd.read_csv(pure_req_user_stories_path)
    
    logger.debug("Renameing columns")
    pure_req_us_df.rename(columns={"databricks-llama-4-maverick": "user story llama-4-maverick", "databricks-meta-llama-3-3-70b-instruct": "user story llama-3-3-70b"}, inplace=True)
    
    logger.debug("Extracting subkeys from the keys column")
    pure_req_us_df[["keys_0", "keys_1", "keys_2"]] = pure_req_us_df["keys"].apply(extract_subkeys).apply(pd.Series)

    logger.debug("Extracting user story parts")
    nlp = spacy.load("en_core_web_sm")
    pure_req_us_df["user_story_parts"] = pure_req_us_df["user story llama-4-maverick"].apply(extract_user_story_parts)
    pure_req_us_df[["type_of_user", "goal", "reason"]] = pure_req_us_df["user story llama-4-maverick"].apply(extract_user_story_parts).apply(pd.Series)