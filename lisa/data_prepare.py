# vou produzir com ia generativa um dataset de cenários de usos
# em seguida vou produzir um modelo de ia generativa para gerar requisitos, aqui devemos retornar um modelo de clusterização para separar os requisitos
# e depois um modelo de ia generativa para gerar requisitos

from dataclasses import dataclass, asdict 
from lisa.utils import generate_filename_map
import pandas as pd
import spacy
import re
import os
from lisa.logger import logger

@dataclass(frozen=True, slots=True)
class RequirementDocumentation:
    id: str
    filename: str
    text: str

@dataclass(frozen=True, slots=True)
class Metadata:
    id: str
    doc_name: str
    text: str
    req_doc_id: str
    
@dataclass(frozen=True, slots=True)
class Requirement:
    id: str
    text: str
    metadata_id: str
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
class MetadataUsageScenariosMap:
    metada_id: str
    usage_scenario_id: str
    
@dataclass(frozen=True, slots=True)
class ReqUsageScenarioMap:
    requirement_id: str
    usage_scenario_id: str

@dataclass(frozen=True, slots=True)
class UserStoriesUsageScenariosMAp:
    user_story_id: str
    usage_scenario_id: str
    
# pure_req_user_stories_path = os.path.join("data", "pure_req_user_stories.csv")
# pure_req_us_df = pd.read_csv(pure_req_user_stories_path)
# cols df: filename,keys,requirement,user story llama-4-maverick,user story llama-3-3-70b,user story llama-3-1-405b,user story dbrx,user story mixtral-8x7b
class DataPrepare:
    """
    Class to prepare data for model training.
    This class extracts RequirementDocumentation, Metadatas, Requirements, User Stories, and Usage Scenarios from specified directories. It processes and populates dataclass instances and saves them into separate pickle (.pkl) files for later use.
    """

    # req_doc_id_keys = generate_filename_map(os.path.join("data", "ReqList_ReqNet_ReqSim", "0.1 Raw Text"))
    def __init__(self, req_user_stories_dataset, raw_text_doc_dir_path, doc_struct_dir_path,req_lists_dir_path, req_doc_id_keys, nlp=spacy.load("en_core_web_sm")):
        self._req_user_stories_dataset = req_user_stories_dataset
        self._raw_text_doc_dir_path = raw_text_doc_dir_path       
        self._req_lists_dir_path = req_lists_dir_path
        self._doc_struct_dir_path = doc_struct_dir_path
        self._req_doc_id_keys = req_doc_id_keys
        self._nlp = nlp
        
    def __extract_req(self, requirement_file, filename):
        """
        Extracts requirements from a text of file.

        Args:
            requirement_fiel str: text of the requirement file
            filename str: name of the file without extension 

        Returns:
            Requirement: Requirement object or empty string if the file is not a .txt file.
        """      
        result = []
        lines = requirement_file.split("\n")
        for line in lines:
            if not line:
                continue

            line = line.split()
            temp_key = line[0].split(".")    
            metadata_id =".".join(temp_key[:-1])
            key = metadata_id + "-" + temp_key[-1]
            text = " ".join(line[1:])
            req_doc_id = self._req_doc_id_keys[filename]
            
            requeriment_datas = {
                "id": req_doc_id+ "-" + key,
                "text": text,
                "metadata_id": req_doc_id + "-" + metadata_id,
                "req_doc_id": req_doc_id 
            }
            result.append(Requirement(**requeriment_datas))
        return result
    
    def __extract_metadatas(self, metadata_file, filename):
        """
        Extracts metadata from a text of file.
        Args:
            metadata_file str: text of the metadata file
            filename str: name of the file without extension
        Returns:
            Metadata: Metadata object or empty string if the file is not a .txt file.
        """
        result = []
        lines = metadata_file.split("\n")
        doc_name = " ".join(lines[0].split()[1:])
        lines = lines[1:]
        req_doc_id = self._req_doc_id_keys[filename]
        
        for line in lines:
            if not line:
                continue

            line = line.split()    
            key = line[0]
            text = " ".join(line[1:])
            
            metadata_datas = {
                "id": req_doc_id + "-" + key,
                "doc_name": doc_name,
                "text": text,
                "req_doc_id": req_doc_id
            }
            result.append(Metadata(**metadata_datas))

        return result
    
    def __extract_user_story_parts(self, text, us_req_id, requirement_id, req_doc_id, metadata_id, pattern=r"As (.*?), I want (.*?) so that (.*?)(?:\.|$)"):
        user_stories = []
        try:
            matches = list(re.findall(pattern, text, re.IGNORECASE))

            for i, match in enumerate(matches):       
                us = {
                    "id": requirement_id + "-" + us_req_id + "." + str(i),
                    "text": f"As {match[0]}, I want {match[1]} so that {match[2]}.",             
                    "type_of_user": match[0],
                    "goal": match[1],
                    "reason": match[2],
                    "requirement_id": requirement_id,
                    "req_doc_id": req_doc_id,
                    "metadata_id": metadata_id
                }
                logger.debug(f"Extracted user story: {us}")
                user_stories.append(UserStory(**us))
        except IndexError:
            logger.error(f"Error extracting user story parts from text (r'As a (.*?), I want (.*?) so that (.*?)(?:\.|$)'): {text}")
        return user_stories
        
    def __get_all_requirements_from_dir(self):
        """
        Extracts all requirements from the directory self._req_lists_dir_path.
        
        Returns:
            list: List of Requirement objects.
        """
        requeriments = []
        for file in sorted(os.listdir(self._req_lists_dir_path)):
            split_file = os.path.splitext(file)
            filename = split_file[0]
            extension = split_file[1]
            
            if not extension == ".txt":
                logger.warning(f"File {file} is not a .txt file, skipping.")
                continue
            
            path = os.path.join(self._req_lists_dir_path, file)
            with open(path, "r", encoding="utf-8") as file:
                requirement_file = file.read()
                requirements = self.__extract_req(requirement_file, filename)
                
            requeriments.extend(requirements)
            logger.debug(f"Extracted {len(requirements)} requirements from file {file}")
        
        return requeriments

    def __get_all_requirements_documentations_from_dir(self):
        """
        Extracts all requirement documentation from the directory self._raw_text_doc_dir_path.
        
        Returns:
            list: List of RequirementDocumentation objects.
        """
        req_docs = []
        for file in sorted(os.listdir(self._raw_text_doc_dir_path)):
            split_file = os.path.splitext(file)
            filename = split_file[0]
            id = self._req_doc_id_keys[filename]

            if not split_file[1] == ".txt":
                logger.warning(f"File {file} is not a .txt file, skipping.")
                continue
            
            path = os.path.join(self._raw_text_doc_dir_path, file)
            with open(path, "r", encoding="utf-8") as file:
                text = file.read()
            
            req_docs.append(RequirementDocumentation(id=id, filename=filename, text=text))
            logger.debug(f"Extracted requirement documentation from file {file}")
        
        return req_docs
    
    def __get_all_metadatas_from_dir(self):
        """
        Extracts all metadata from the directory self._doc_struct_dir_path.
        
        Returns:
            list: List of Metadata objects.
        """
        metadatas = []
        for file in sorted(os.listdir(self._doc_struct_dir_path)):
            split_file = os.path.splitext(file)
            filename = split_file[0]
            
            if not split_file[1] == ".txt":
                logger.warning(f"File {file} is not a .txt file, skipping.")
                continue
            
            path = os.path.join(self._doc_struct_dir_path, file)
            with open(path, "r", encoding="utf-8") as file:
                metadata_file = file.read()
            
            metadatas_from_file = self.__extract_metadatas(metadata_file, filename)
                
            metadatas.extend(metadatas_from_file)
            logger.debug(f"Extracted {len(metadatas_from_file)} metadatas from file {file}")
        return metadatas
    
    def __get_all_user_stories_from_csv(self):
        """
        Extracts all user stories from csv self._req_user_stories_dataset.
        
        Returns:
            list: List of UserStory objects.
        """
        user_stories = []
        pure_req_us_df = pd.read_csv(self._req_user_stories_dataset, quotechar='"', skipinitialspace=True)
        
        for _, row in pure_req_us_df.iterrows():
            ids = row["keys"].split(".")
            req_doc_id = ids[1].replace("D", "")
            metadata_id = req_doc_id + "-" + ".".join(ids[2:-1])
            requirement_id = metadata_id + "-" + ".".join(ids[-1])
            
            for i, col in enumerate(pure_req_us_df.columns[3:]):
                if pd.isna(row[col]):
                    continue
                
                text = row[col]
                us_req_id = str(i)
                user_stories.extend(self.__extract_user_story_parts(text, us_req_id, requirement_id, req_doc_id, metadata_id))
        
        return user_stories
                
    def __extract_usage_scenarios(self):
        pass

    def get_dataframe_from(self, data_list):
        list_of_dicts = [asdict(data) for data in data_list]
        return pd.DataFrame(list_of_dicts)
    
    @property
    def dataframe(self): 
        return self._dataframe
    
    @dataframe.setter
    def dataframe(self, value):
        self._dataframe = value

if __name__ == "__main__":
    req_user_stories_dataset = os.path.join("data", "pure_req_user_stories.csv")
    raw_text_doc_dir_path = os.path.join("data", "ReqList_ReqNet_ReqSim","0.1 Raw Text")
    req_lists_dir_path = os.path.join("data", "ReqList_ReqNet_ReqSim", "1 ReqLists")
    doc_struct_dir_path = os.path.join("data", "ReqList_ReqNet_ReqSim", "2 DocumentStructure - Metadata")
    req_doc_id_keys = generate_filename_map(raw_text_doc_dir_path)
    
    data_prepare = DataPrepare(req_user_stories_dataset, raw_text_doc_dir_path, req_lists_dir_path, doc_struct_dir_path, req_doc_id_keys)
    