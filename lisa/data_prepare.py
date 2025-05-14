# vou produzir com ia generativa um dataset de cenários de usos
# em seguida vou produzir um modelo de ia generativa para gerar requisitos, aqui devemos retornar um modelo de clusterização para separar os requisitos
# e depois um modelo de ia generativa para gerar requisitos
from typing import List, Union
from dataclasses import dataclass, asdict 
import pandas as pd
import spacy
import re
import os
from lisa.logger import logger
from lisa.id_generator import RequirementDocumentationID, MetadataID, RequirementID, UserStoryID, UsageScenarioID, ModelGeneratorID, ClusterID, IDGenerator

@dataclass(frozen=True, slots=True)
class RequirementDocumentation:
    id: RequirementDocumentationID
    filename: str
    text: str

@dataclass(frozen=True, slots=True)
class Metadata:
    id: MetadataID
    doc_name: str
    text: str
    req_doc_id: RequirementDocumentationID
    
@dataclass(frozen=True, slots=True)
class Requirement:
    id: RequirementID
    text: str
    metadata_id: MetadataID
    req_doc_id: RequirementDocumentationID

@dataclass(frozen=True, slots=True)
class UserStory:
    id: UserStoryID
    text: str
    type_of_user: str
    goal: str
    reason: str
    req_doc_id: RequirementDocumentationID
    metadata_id: MetadataID
    requirement_id: RequirementID
    
@dataclass(frozen=True, slots=True)
class UsageScenario:
    id: UsageScenarioID
    text: str
    req_doc_id: RequirementDocumentationID
        
@dataclass(frozen=True, slots=True)
class MetadataUsageScenariosMap:
    metada_id: MetadataID
    usage_scenario_id: UsageScenarioID
    
@dataclass(frozen=True, slots=True)
class ReqUsageScenarioMap:
    requirement_id: RequirementID
    usage_scenario_id: UsageScenarioID

@dataclass(frozen=True, slots=True)
class UserStoriesUsageScenariosMap:
    user_story_id: UserStoryID
    usage_scenario_id: UsageScenarioID
    
# pure_req_user_stories_path = os.path.join("data", "pure_req_user_stories.csv")
# pure_req_us_df = pd.read_csv(pure_req_user_stories_path)
# cols df: filename,keys,requirement,user story llama-4-maverick,user story llama-3-3-70b,user story llama-3-1-405b,user story dbrx,user story mixtral-8x7b
class DataPrepare:
    """
    Class to prepare data for model training.
    This class extracts RequirementDocumentation, Metadatas, Requirements, User Stories, and Usage Scenarios from specified directories. It processes and populates dataclass instances and saves them into separate pickle (.pkl) files for later use.
    """

    # req_doc_id_keys = generate_filename_map(os.path.join("data", "ReqList_ReqNet_ReqSim", "0.1 Raw Text"))
    def __init__(self, 
                 req_user_stories_dataset_path: str, 
                 raw_text_doc_dir_path: str, 
                 doc_struct_dir_path: str, 
                 req_lists_dir_path: str, 
                 id_generator: IDGenerator, nlp=spacy.load("en_core_web_sm")):
        self._req_user_stories_dataset_path = req_user_stories_dataset_path
        self._raw_text_doc_dir_path = raw_text_doc_dir_path       
        self._req_lists_dir_path = req_lists_dir_path
        self._doc_struct_dir_path = doc_struct_dir_path
        self._id_generator = id_generator
        self._nlp = nlp
        self._df_requirements = None
        self._df_user_stories = None
        self._df_metadata = None
        self._df_req_docs = None

        
    def __extract_req(self, requirement_file: str, filename: str) -> List[Requirement]:
        """
        Extracts requirements from a requirement file content.

        Args:
            requirement_file (str): Text content of the requirement file.
            filename (str): Name of the file (without extension).

        Returns:
            List[Requirement]: A list of extracted Requirement instances.
        """  
        result = []
        lines = requirement_file.split("\n")
        for line in lines:
            if not line.strip():
                continue

            tokens = line.split()
            key_parts = tokens[0].split(".")    
            text = " ".join(tokens[1:])
            
            try:
                metadata_key = ".".join(key_parts[:-1])
                requirement_key = int(key_parts[-1])
            except ValueError:
                logger.error(f"Extracting requirement from line: {line}, filename: {filename}") 
                raise ValueError(f"Invalid key format in key_parts: {key_parts}")
            
            metadata_id = self._id_generator.generate_metadata_id(filename, metadata_key)
            req_id = self._id_generator.generate_requirement_id(metadata_id, requirement_key)
            req_doc_id = self._id_generator.generate_requirement_documentation_id(filename)
            
            requeriment_datas = {
                "id": req_id,
                "text": text,
                "metadata_id": metadata_id,
                "req_doc_id": req_doc_id 
            }
            result.append(Requirement(**requeriment_datas))
        return result
    
    def __extract_metadatas(self, metadata_file: str, filename: str) -> List[Metadata]:
        """
        Extracts metadatas from a metadata file content.

        Args:
            metadata_file (str): Text content of the metadata file.
            filename (str): Name of the file without extension.

        Returns:
            List[Metadata]: A list of extracted Metadata instances.
        """
        result = []
        lines = metadata_file.split("\n")
        doc_name = " ".join(lines[0].split()[1:])
        lines = lines[1:]
        
        req_doc_id = self._id_generator.generate_requirement_documentation_id(filename)
        
        for line in lines:
            if not line.strip():
                continue

            tokens = line.split()  
            try:  
                metadata_key = str(tokens[0])
            except ValueError:
                raise ValueError(f"Invalid key format in tokens: {tokens[0]}")
                    
            text = " ".join(tokens[1:])
            metadata_id = self._id_generator.generate_metadata_id(filename, metadata_key)
            #logger.debug(f"Extrating id metadata: metadata_id: {metadata_id}, metadata_parts: {metadata_id.id} and {metadata_id.requirement_documentation_key}, type: {type(metadata_id)}")
            metadata_datas = {
                "id": metadata_id,
                "doc_name": doc_name,
                "text": text,
                "req_doc_id": req_doc_id
            }
            result.append(Metadata(**metadata_datas))

        return result
    
    def __extract_user_story_parts(
        self,
        text: str,
        model_generator_id: ModelGeneratorID,
        requirement_id: RequirementID,
        req_doc_id: RequirementDocumentationID,
        metadata_id: MetadataID,
        pattern: str = r"As (.*?), I want (.*?) so that (.*?)(?:\.|$)"
    ) -> List[UserStory]:
        """
        Extracts user stories from a given text using a specified pattern.

        Args:
            text (str): The input text from which user stories should be extracted.
            us_req_id (int): A unique suffix identifier for the user story within a requirement.
            requirement_id (RequirementID): The ID of the associated requirement.
            req_doc_id (RequirementDocumentationID): The ID of the document the requirement belongs to.
            metadata_id (MetadataID): The ID of the metadata associated with the requirement.
            pattern (str, optional): Regex pattern to match user stories.

        Returns:
            List[UserStory]: A list of extracted UserStory instances.
        """
        user_stories = []
        try:
            matches = list(re.findall(pattern, text, re.IGNORECASE))

            for i, match in enumerate(matches):       
                user_story_id = self._id_generator.generate_user_story_id(
                    requirement_id=requirement_id,
                    model_generator_id=model_generator_id,
                    id=i
                )
                user_story = {
                    "id": user_story_id,
                    "text": f"As {match[0]}, I want {match[1]} so that {match[2]}.",             
                    "type_of_user": match[0],
                    "goal": match[1],
                    "reason": match[2],
                    "requirement_id": requirement_id,
                    "req_doc_id": req_doc_id,
                    "metadata_id": metadata_id
                }
                user_stories.append(UserStory(**user_story))
        
        except Exception as e:
            raise Exception(f"Error extracting user stories from text: {text}. Error: {e}")
        return user_stories
        
    def __get_all_requirements_from_dir(self) -> List[Requirement]:
        """
        Extracts all requirements from the directory self._req_lists_dir_path.
        
        Returns:
            List[Requirement]: List of Requirement objects.
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

    def __get_all_requirements_documentations_from_dir(self) -> List[RequirementDocumentation]:
        """
        Extracts all requirement documentation from the directory self._raw_text_doc_dir_path.
        
        Returns:
            List[RequirementDocumentation]: List of RequirementDocumentation objects.
        """
        req_docs = []
        for file in sorted(os.listdir(self._raw_text_doc_dir_path)):
            split_file = os.path.splitext(file)
            filename = split_file[0]
            id = self._id_generator.generate_requirement_documentation_id(filename)

            if not split_file[1] == ".txt":
                logger.warning(f"File {file} is not a .txt file, skipping.")
                continue
            
            path = os.path.join(self._raw_text_doc_dir_path, file)
            with open(path, "r", encoding="utf-8") as file:
                text = file.read()
            
            req_docs.append(RequirementDocumentation(id=id, filename=filename, text=text))
            #logger.debug(f"Extracted requirement documentation from file {file}")
        
        return req_docs
    
    def __get_all_metadatas_from_dir(self) -> List[Metadata]:
        """
        Extracts all metadata from the directory self._doc_struct_dir_path.
        
        Returns:
            List[Metadata]: List of Metadata objects.
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
    
    def __get_all_user_stories_from_csv(self) -> List[UserStory]:
        """
        Extracts all user stories from csv self._req_user_stories_dataset.
        
        Returns:
            List[UserStory]: List of UserStory objects.
        """
        user_stories = []
        pure_req_us_df = pd.read_csv(self._req_user_stories_dataset_path, quotechar='"', skipinitialspace=True)
        
        for _, row in pure_req_us_df.iterrows():
            keys = row["keys"].split(".")
            
            try:
                req_doc_id = RequirementDocumentationID.from_str(keys[1].replace("D", ""))
                metadata_id = MetadataID.from_str(str(req_doc_id) + "-" + ".".join(keys[2:-1]))
                requirement_id = RequirementID.from_str(str(metadata_id) + "-" + keys[-1])

                for i, col in enumerate(pure_req_us_df.columns[3:]):
                    if pd.isna(row[col]):
                        continue
                    
                    text = row[col]
                    model_generator_id = ModelGeneratorID(id=i)
                    
                    user_stories.extend(
                        self.__extract_user_story_parts(
                            text, 
                            model_generator_id, 
                            requirement_id, 
                            req_doc_id, 
                            metadata_id
                            )
                        )
            except Exception as e:
                raise Exception(f"Error parsing keys '{row['keys']}': {e}")
                
        return user_stories
                
    def __extract_usage_scenarios(self):
        pass
    
    def __extract_metadata_usage_scenarios_map(self):
        pass
    
    def __extract_req_usage_scenarios_map(self):
        pass
    
    def __extract_user_stories_usage_scenarios_map(self):
        pass

    def __get_dataframe_from(self, data_list: List[object], save_as_string: bool = False) -> pd.DataFrame:
        """
        Converts a list of dataclass instances into a pandas DataFrame.
        
        Args:
            data_list (List[object]): List of dataclass instances to convert.
            
        Returns:
            pd.DataFrame: A pandas DataFrame containing the data from the dataclass instances.
        """
        list_of_dicts = []
        for obj in data_list:
            row = {}
            for field in obj.__dataclass_fields__:
                value = getattr(obj, field)
                if save_as_string:
                    value = str(value)
                row[field] = value
                
            list_of_dicts.append(row)
        return pd.DataFrame(list_of_dicts)
        
    @property
    def dataframe(self) -> pd.DataFrame: 
        return self._dataframe
    
    @dataframe.setter
    def dataframe(self, value: pd.DataFrame):
        self._dataframe = value
        
    @property
    def df_requirements(self) -> pd.DataFrame:
        if self._df_requirements is None:
            self._df_requirements = self.__get_dataframe_from(self.__get_all_requirements_from_dir())
        return self._df_requirements

    @property
    def df_user_stories(self) -> pd.DataFrame:
        if self._df_user_stories is None:
            self._df_user_stories = self.__get_dataframe_from(self.__get_all_user_stories_from_csv())
        return self._df_user_stories

    @property
    def df_metadata(self) -> pd.DataFrame:
        if self._df_metadata is None:
            self._df_metadata = self.__get_dataframe_from(self.__get_all_metadatas_from_dir())
        return self._df_metadata

    @property
    def df_req_docs(self) -> pd.DataFrame:
        if self._df_req_docs is None:
            self._df_req_docs = self.__get_dataframe_from(self.__get_all_requirements_documentations_from_dir())
        return self._df_req_docs

    """@property
    def df_usage_scenarios(self) -> pd.DataFrame:
        if self._df_usage_scenarios is None:
            self._df_usage_scenarios = self.__get_dataframe_from(self.__extract_usage_scenarios())
        return self._df_usage_scenarios
    
    @property
    def df_metadata_usage_scenarios_map(self) -> pd.DataFrame:
        if self._df_metadata_usage_scenarios_map is None:
            self._df_metadata_usage_scenarios_map = self.__get_dataframe_from(self.__extract_metadata_usage_scenarios_map())
        return self._df_metadata_usage_scenarios_map
    
    @property
    def df_req_usage_scenarios_map(self) -> pd.DataFrame:
        if self._df_req_usage_scenarios_map is None:
            self._df_req_usage_scenarios_map = self.__get_dataframe_from(self.__extract_req_usage_scenarios_map())
        return self._df_req_usage_scenarios_map
    
    @property
    def df_user_stories_usage_scenarios_map(self) -> pd.DataFrame:
        if self._df_user_stories_usage_scenarios_map is None:
            self._df_user_stories_usage_scenarios_map = self.__get_dataframe_from(self.__extract_user_stories_usage_scenarios_map())
        return self._df_user_stories_usage_scenarios_map"""
        
if __name__ == "__main__":
    from lisa.utils import generate_filename_map
    
    req_user_stories_dataset_path = os.path.join("data", "pure_req_user_stories.csv")
    raw_text_doc_dir_path = os.path.join("data", "ReqList_ReqNet_ReqSim","0.1 Raw Text")
    doc_struct_dir_path = os.path.join("data", "ReqList_ReqNet_ReqSim", "1 DocumentStructure - Metadata")
    req_lists_dir_path = os.path.join("data", "ReqList_ReqNet_ReqSim", "2 ReqLists")
    req_doc_id_keys = generate_filename_map(raw_text_doc_dir_path)
    
    data_prepare = DataPrepare(req_user_stories_dataset_path, raw_text_doc_dir_path, doc_struct_dir_path,req_lists_dir_path, id_generator=IDGenerator(raw_text_doc_dir_path))
    
    data_prepare.df_requirements.to_pickle(os.path.join("data", "df", "df_requirements.pkl"))
    data_prepare.df_user_stories.to_pickle(os.path.join("data", "df", "df_user_stories.pkl"))
    data_prepare.df_metadata.to_pickle(os.path.join("data", "df", "df_metadatas.pkl"))
    data_prepare.df_req_docs.to_pickle(os.path.join("data", "df", "df_req_docs.pkl"))
    
    # data_prepare.df_usage_scenarios.to_pickle(os.path.join("data", "df", "df_usage_scenarios.pkl")))
    # data_prepare.df_metadata_usage_scenarios_map.to_pickle(os.path.join("data", "df", "df_metadata_usage_scenarios_map.pkl")))
    # data_prepare.df_req_usage_scenarios_map.to_pickle(os.path.join("data", "df", "df_req_usage_scenarios_map.pkl")))
    # data_prepare.df_user_stories_usage_scenarios_map.to_pickle(os.path.join("data", "df", "df_user_stories_usage_scenarios_map.pkl")))