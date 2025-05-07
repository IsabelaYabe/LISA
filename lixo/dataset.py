import pandas as pd
import os
import spacy
import re
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "src")))
from logger import logger
from requirement_documentation import RequirementDocumentation
from requirement import Requirement
from lisa.data_prepare import DatasetPrepare

class dataset:
    def __init__(self, csv_file=os.path.join("data", "pure_req_user_stories.csv"), docs_path=os.path.join("data", "ReqList_ReqNet_ReqSim", "0    Requirement Specification Documents"), docs_structure_path=os.path.join("data", "ReqList_ReqNet_ReqSim", "1.3 DocumentStructure - Metadata"), pattern=r"As (.*?), I want (.*?) so that (.*?)(?:\.|$)", nlp=spacy.load("en_core_web_sm")):
        self._csv_file=csv_file
        self._dataset=DatasetPrepare(csv_file, pattern, nlp).dataset
        self._docs_path=docs_path
        self._docs_structure_path=docs_structure_path
        self._nlp=nlp
        self._pattern=pattern
        self._requirement_documentations=None
        self._usage_scenarios=None
        self._requirements={}
        self._user_stories=None

    def __extract_datas_from_csv(self):
        files_visited = []
        pure_req_user_stories = self._dataset_prepared.dataframe
        for index_doc, row_doc in pure_req_user_stories.iterrows():
            current_filename = row_doc["Filename"]
            current_filename_just_name = current_filename[:current_filename.find(".")]
            
            if current_filename not in files_visited:
                files_visited.append(current_filename)
                pure_filename = pure_req_user_stories[pure_req_user_stories["Filename"] == current_filename]
                
                user_stories_visited = {}
                requirements_visited = {}
                for index_req, row_req in pure_filename.iterrows():
                    current_req = row_req["Requirements"] 
                    requirement = Requirement(
                        id = len(requirements_visited),
                        key = row_req["key"],
                        text = current_req,
                        nlp = self._nlp
                    )
                    requirements_visited[requirement.key] = requirement
                    self._requirements[requirement.key] = requirement

                    current_user_story_llama_4 = row_req["user story (us llama-4-maverick)"]
                    for user_story in current_user_story_llama_4:
                        if user_story not in user_stories_visited:
                            user_stories_visited[user_story] = []
                        user_stories_visited[user_story].append(requirement.key)


                requirement_documentation = RequirementDocumentation(
                    id = len(files_visited),
                    filename = current_filename_just_name,
                    documentation_path = os.path.join(self._docs_path, current_filename),
                    requirements = requirements_visited
                )
                self._requirement_documentations.append(requirement_documentation)