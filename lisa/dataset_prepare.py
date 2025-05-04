# vou produzir com ia generativa um dataset de cenários de usos
# em seguida vou produzir um modelo de ia generativa para gerar requisitos, aqui devemos retornar um modelo de clusterização para separar os requisitos
# e depois um modelo de ia generativa para gerar requisitos

from dataclasses import dataclass   
import pandas as pd
import spacy
import re
import os
from logger import logger

@dataclass(frozen=True, slots=True)
class RequirementDocumentation:
    id: str
    title: str
    text: str
    file_path: str
    metadata_id: str

@dataclass(frozen=True, slots=True)
class Requirement:
    id: str
    text: str
    metadata_id: str
    req_doc_id: str

@dataclass(frozen=True, slots=True)
class Metadata:
    id: str
    doc_name: str
    text: str
    req_doc_id: str

@dataclass(frozen=True, slots=True)
class UserStory:
    id: str
    text: str
    type_of_user: str
    goal: str
    reason: str
    requirement_id: str
    req_doc_id: str
    metadata_id: str

@dataclass(frozen=True, slots=True)
class UsageScenario:
    id: str
    text: str
    req_doc_id: str
        
@dataclass(frozen=True, slots=True)
class ReqUsageScenarioMap:
    requirement_id: str
    usage_scenario_id: str

@dataclass(frozen=True, slots=True)
class UserStoriesUsageScenariosMAp:
    user_story_id: str
    usage_scenario_id: str

@dataclass(frozen=True, slots=True)
class MetadataUsageScenariosMap:
    metada_id: str
    usage_scenario_id: str
        
# pure_req_user_stories_path = os.path.join("data", "pure_req_user_stories.csv")
# pure_req_us_df = pd.read_csv(pure_req_user_stories_path)
# cols df: filename,keys,requirement,user story llama-4-maverick,user story llama-3-3-70b,user story llama-3-1-405b,user story dbrx,user story mixtral-8x7b
class DataPrepare:
    """
    Class to prepare data for model training.
    This class extracts RequirementDocumentation, Metadatas, Requirements, User Stories, and Usage Scenarios from specified directories. It processes and populates dataclass instances and saves them into separate pickle (.pkl) files for later use.
    """

    def __init__(self, req_user_stories_dataset, raw_text_doc_dir_path, req_lists_dir_path, doc_struct_dir_path, nlp=spacy.load("en_core_web_sm")):
        self._req_user_stories_dataset = req_user_stories_dataset
        self._raw_text_doc_dir_path = raw_text_doc_dir_path       
        self._req_lists_dir_path = req_lists_dir_path
        self._doc_struct_dir_path = doc_struct_dir_path
        self._nlp = nlp
        #self._requeriments_extracted,self._user_stories_extracted, self._usage_scenarios_extracted, self._req_docs_extracted, self._docs_metadatas = self.__get_all_data()
    
    def _extract_text_from_file(self, dir, file): 
        """
        Extracts text from a file. The function checks if the file is a .txt file and reads its content. If the file is not a .txt file, it returns an empty string.
        The function is used to extract text from files in the specified directory.        

        # Extracts raw text of documentation from the directory data/ReqList_ReqNet_ReqSim/0.1 Raw Text
        ## Extracts requirements from the directory data/ReqList_ReqNet_ReqSim/1.1 ReqLists
        ## data/ReqList_ReqNet_ReqSim/1.3 DocumentStructure - Metadata
        
        Args:
            dir str: dir_path
            file str: file_path

        Returns:
            str: text content of the file or empty string if the file is not a .txt file.
        """
        if file.endswith(".txt"):
            with open(os.path.join(dir, file), "r", encoding="utf-8") as file:
                logger.debug(f"Extracting text from file: {file}")
                return file.read()
        return ""
    
    def _extract_user_story_parts(self, text, requirement_id, req_doc_id, metadata_id, pattern=r"As (.*?), I want (.*?) so that (.*?)(?:\.|$)"):
        user_stories = []

        logger.debug(f"Extracting user story parts from text: {text}")
        try:
            matches = list(re.findall(pattern, text, re.IGNORECASE))

            for i, match in enumerate(matches):       
                us = {
                    "id": requirement_id + f"us.{i}",
                    "text": f"As {match[0]}, I want {match[1]} so that {match[2]}.",             
                    "type of user": match[0],
                    "goal": match[1],
                    "reason": match[2],
                    "requirement_id": requirement_id,
                    "req_doc_id": req_doc_id,
                    "metadata_id": metadata_id
                }
                user_stories.append(UserStory(**us))
        except IndexError:
            logger.error(f"Error extracting user story parts from text (r'As a (.*?), I want (.*?) so that (.*?)(?:\.|$)'): {text}")
        return user_stories

    def _extract_metadatas(self, file, metadata_doc, req_doc_id):
        result = []
        lines = metadata_doc.split("\n")
        doc_name = " ".join(lines[0].split()[1:])
        lines = lines[1:]

        for line in lines:
            if not line:
                continue

            line = line.split()    
            key = line[0]
            text = " ".join(line[1:])
            
            metadata_datas = {
                "id": "DSM." + file + f".{key}",
                "doc_name": doc_name,
                "text": text,
                "req_doc_id": req_doc_id
            }
            result.append(Metadata(**metadata_datas))

        return result

    def _extract_usage_scenarios(self):
        pass

    def _extract_req(self, dir, file):
        pass

    def _extract_req_doc(self, dir, file):
        pass
    
    def _get_dataframe(self):
        logger.debug("Starting the dataset preparation process")
        pure_req_us_df = pd.read_csv(self._csv_file)

        logger.debug("Extracting user story parts llama-4-maverick")
        pure_req_us_df[["user story (us llama-4-maverick)", "user story parts (us llama-4-maverick)"]] = pure_req_us_df.apply(lambda story_list: self._extract_user_story_parts(story_list["user story llama-4-maverick"]), axis=1, result_type="expand")

        logger.debug("Extracting user story parts llama-3-3-70b")
        pure_req_us_df[["user story (us llama-3-3-70b)", "user story parts (us llama-3-3-70b)"]] = pure_req_us_df.apply(lambda story_list: self._extract_user_story_parts(story_list["user story llama-3-3-70b"]), axis=1, result_type="expand")

        logger.debug("Extracting user story parts llama-3-1-405b")
        pure_req_us_df[["user story (us llama-3-1-405b)", "user story parts (us llama-3-1-405b)"]] = pure_req_us_df.apply(lambda story_list: self._extract_user_story_parts(story_list["user story llama-3-1-405b"]), axis=1, result_type="expand")
    
        logger.debug("Extracting user story parts dbrx")
        pure_req_us_df[["user story (us dbrx)", "user story parts (us dbrx)"]] = pure_req_us_df.apply(lambda story_list: self._extract_user_story_parts(story_list["user story dbrx"]), axis=1, result_type="expand")
        
        logger.debug("Extracting user story parts mixtral-8x7b")
        pure_req_us_df[["user story (us mixtral-8x7b)", "user story parts (us mixtral-8x7b)"]] = pure_req_us_df.apply(lambda story_list: self._extract_user_story_parts(story_list["user story mixtral-8x7b"]), axis=1, result_type="expand")

        return pure_req_us_df

    def _extract_path_from_file(self, dir, file):
        path = os.path.join(dir, file)
        return path

    def _dir_interable(self, dir, **kwargs):
        returns_functions = {}

        for value in kwargs.keys():
            returns_functions[value] = []
        
        for file in os.listdir(dir):
            for value, function in kwargs.items():
                returns_functions[value].append(function(dir, file))
        
        return returns_functions
    
    def __get_all_data(self):
        logger.debug("Starting the data extraction process")
        requeriments_extracted = []
        user_stories_extracted = []
        usage_scenarios_extracted = []
        req_docs_extracted = []
        docs_metadatas = []

        req_docs_extracted = self._dir_interable(self._raw_text_doc_dir_path, doc=self._extract_path_from_file)["doc"]

        requeriments_extracted = self._dir_interable(self._req_lists_dir_path, req=self._extract_req)["req"]
        
        docs_metadatas = self._dir_interable(self._docs_metadatas, metadata=self._extract_metadatas)["metadata"]

        user_stories_extracted = self._dir_interable(self._raw_text_path, user_stories=self._extract_text_from_file)["user_stories"]
        
        usage_scenarios_extracted = self._dir_interable(self._doc_struct_dir_path, usage_scenarios=self._extract_text_from_file)["usage_scenarios"]

        return requeriments_extracted, user_stories_extracted, usage_scenarios_extracted, req_docs_extracted, docs_metadatas
    
    @property
    def dataframe(self): 
        return self._dataframe
    
    @dataframe.setter
    def dataframe(self, value):
        self._dataframe = value

if __name__ == "__main__":
    #dataset = DatasetPrepare(os.path.join("data", "pure_req_user_stories.csv"))
    #logger.debug("Saving the dataset")
    #pure_req_us_df = dataset.dataframe
    #
    #pure_req_us_df.to_json(os.path.join("data", "pure_req_user_stories_annotateSS.json"), index=False)
    req_user_stories_dataset = os.path.join("data", "pure_req_user_stories.csv")
    raw_text_doc_dir_path = os.path.join("data", "ReqList_ReqNet_ReqSim","0.1 Raw Text")
    req_lists_dir_path = os.path.join("data", "ReqList_ReqNet_ReqSim", "1.1 ReqLists")
    doc_struct_dir_path = os.path.join("data", "ReqList_ReqNet_ReqSim", "1.3 DocumentStructe - Metadata")
    
    data_prepare = DataPrepare(req_user_stories_dataset, raw_text_doc_dir_path, req_lists_dir_path, doc_struct_dir_path)
    
    
    file = "file"
    metadata_doc = """D   POREM APSON Porem
1   Chitzu Porem
2   Golden Porem
2.2 Maki Porem
3   Bacoz Porem"""
    req_doc_id = "req_doc_id"
    
    metadatas = DataPrepare._extract_metadatas(data_prepare, file, metadata_doc, req_doc_id)
    
    logger.debug(f"Metadatas: {metadatas}")
    logger.debug(f"len(metadatas): {len(metadatas)}")
    for i in range(len(metadatas)):
        logger.debug(f"=======")
        logger.debug(f"Metadatas[0].text: {metadatas[i].text}")
        logger.debug(f"Metadatas[0].id: {metadatas[i].id}")
        logger.debug(f"Metadatas[0].doc_name: {metadatas[i].doc_name}")
    