from dataclasses import dataclass, field   
import pandas as pd
import spacy
import re
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), 'src')))
from logger import logger

@dataclass(frozen=True, slots=True)
class UserStory:
        id: str
        text: str
        type_of_user: str 
        goal: str 
        reason: str 
        requirement: str
        doc: str
        metadata: str
        verbs: list[str] = field(default_factory=list)
        auxs: list[str] = field(default_factory=list)
        nouns: list[str] = field(default_factory=list) 
        propns: list[str] = field(default_factory=list)
        prons: list[str] = field(default_factory=list) 
        adjs: list[str] = field(default_factory=list) 
        advs: list[str] = field(default_factory=list)
        sconjs: list[str] = field(default_factory=list) 
        parts: list[str] = field(default_factory=list) 
        orgs: list[str] = field(default_factory=list)

@dataclass(frozen=True, slots=True)
class UsageScenario:
        id: str
        text: str
        requirements: str
        verbs: list[str] = field(default_factory=list)
        auxs: list[str] = field(default_factory=list)
        nouns: list[str] = field(default_factory=list) 
        propns: list[str] = field(default_factory=list)
        prons: list[str] = field(default_factory=list) 
        adjs: list[str] = field(default_factory=list) 
        advs: list[str] = field(default_factory=list)
        sconjs: list[str] = field(default_factory=list) 
        parts: list[str] = field(default_factory=list) 
        orgs: list[str] = field(default_factory=list)

@dataclass(frozen=True, slots=True)
class Requirement:
        id: str 
        key: str
        text: str
        metadata: str
        user_stories: list[str] = field(default_factory=list)
        usage_scenarios: list[str] = field(default_factory=list)
        verbs: list[str] = field(default_factory=list)
        auxs: list[str] = field(default_factory=list)
        nouns: list[str] = field(default_factory=list) 
        propns: list[str] = field(default_factory=list)
        prons: list[str] = field(default_factory=list) 
        adjs: list[str] = field(default_factory=list) 
        advs: list[str] = field(default_factory=list)
        sconjs: list[str] = field(default_factory=list) 
        parts: list[str] = field(default_factory=list) 
        orgs: list[str] = field(default_factory=list)

@dataclass(frozen=True, slots=True)
class RequirementDocumentation:
        id: str
        title: str
        text: str
        documentation_path: str
        metadata: str
        requirements: list[Requirement] = field(default_factory=list)
        user_stories: list[UserStory] = field(default_factory=list)
        usage_scenarios: list[UsageScenario] = field(default_factory=list)

# pure_req_user_stories_path = os.path.join("data", "pure_req_user_stories.csv")
# pure_req_us_df = pd.read_csv(pure_req_user_stories_path)
class DatasetPrepare:
    def __init__(self, csv_file, req_doc_path, raw_text_path, doc_struct_path, pattern=r"As (.*?), I want (.*?) so that (.*?)(?:\.|$)", nlp=spacy.load("en_core_web_sm")):
        self._csv_file = csv_file
        self._req_doc_path = req_doc_path
        self._raw_text_path = raw_text_path
        self._doc_struct_path = doc_struct_path       
        self._pattern = pattern
        self._nlp = nlp
        self._requeriments_extracted,self._user_stories_extracted, self._usage_scenarios_extracted, self._req_docs_extracted, self._docs_metadatas = self.__get_all_data()
        self._dataframe = self._get_dataframe()

    def _extract_user_story_parts(self, text):
        user_story = []
        user_story_parts = []
        logger.debug(f"Extracting user story parts from text: {text}")
        pattern = r"As (.*?), I want (.*?) so that (.*?)(?:\.|$)"
        try:
            matches = list(re.findall(pattern, text, re.IGNORECASE))
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

    def _extract_metadatas(self, metadata_doc):
        result = {}
        pattern = re.compile(r"^([A-Z]|\d+(\.\d+)*)(\s+)(.+)")

        for line in metadata_doc:
            line = line.strip()
            if not line:
                continue

            match = pattern.match(line)
            if match:
                key = match.group(1)
                text = match.group(4)
                result[key] = text

        return result

    def _extract_token_pos(self, text):
        doc = self._nlp(text)
        logger.debug("Starting the tokenization and POS tagging process")   

        verb = [] # Verbo (should provide, enable, etc.)
        aux = [] # Auxiliar (should, must, can, is, are, etc.)
        noun = [] # Substantivo (system, user, report, etc.)
        propn = [] # Substantivo próprio (Nome da empresa, nome de funcionário, etc.)
        pron = [] # Pronome (user, it, they, he, she, etc.)
        adj = [] # Adjetivo (detailed, new, old, etc.)
        adv = [] # Advérbio (automatically, only, quickly, slowly, etc.)
        sconj = [] # Conjunção subordinada (so, that, when, if, because, etc.)
        part = [] # Particípio (to provide, to allow, should be, should have, etc.)
        org = [] # Organização (Nome da empresa, nome de funcionário, etc.)
        
        for token in doc:
            token_text = token.text
            pos = token.pos_

            if pos == "VERB":
                verb.append(token_text)
            elif pos == "AUX":
                aux.append(token_text)
            elif pos == "NOUN":
                noun.append(token_text)
            elif pos == "PROPN":
                propn.append(token_text)
            elif pos == "PRON":
                pron.append(token_text)
            elif pos == "ADJ":
                adj.append(token_text)
            elif pos == "ADV":
                adv.append(token_text)
            elif pos == "SCONJ":
                sconj.append(token_text)
            elif pos == "PART":
                part.append(token_text)
            elif pos == "ORG":
                org.append(token_text)
        logger.debug("Token classification completed")
        return verb, aux, noun, propn, pron, adj, adv, sconj, part, org
    
    def _extract_text_from_file(self, dir, file):
        text = ""
        if file.endswith(".txt"):
            with open(os.path.join(dir, file), "r", encoding="utf-8") as file:
                text += file.read()
        return text
    
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

        req_docs_extracted = self._dir_interable(self._req_doc_path, doc=self._extract_path_from_file)["doc"]

        # Extracting the requirements
        requeriments_extracted = self._dir_interable(self._req_doc_path, req=self._extract_req)["req"]
        
        docs_metadatas = self._dir_interable(self._docs_metadatas, metadata=self._extract_metadatas)["metadata"]

        user_stories_extracted = self._dir_interable(self._raw_text_path, user_stories=self._extract_text_from_file)["user_stories"]
        
        usage_scenarios_extracted = self._dir_interable(self._doc_struct_path, usage_scenarios=self._extract_text_from_file)["usage_scenarios"]

        return requeriments_extracted, user_stories_extracted, usage_scenarios_extracted, req_docs_extracted, docs_metadatas
    
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
    
    pure_req_us_df.to_json(os.path.join("data", "pure_req_user_stories_annotateSS.json"), index=False)