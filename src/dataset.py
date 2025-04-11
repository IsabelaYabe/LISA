import pandas as pd
import os
import spacy
import re
from src.logger import setup_logger

logger = setup_logger()

def extract_user_story_parts(text):
    pattern = r"As a (.*?), I want (.*?) so that (.*?)(?:\.|$)"
    match = re.findall(pattern, text, re.IGNORECASE)[0]
    result = {
            "type_of_user": match[0],
            "goal": match[1],
            "reason": match[2]
        }
    return result

def extract_subkeys(key):
    key_split = key.split(".")
    key_0 = ".".join(key_split[:2])
    key_1 = key_split[2]
    key_2 = ".".join(key_split[3:])

    return key_0, key_1, key_2

def extract_token_pos(text, nlp):
    logger.debug("Starting the tokenization and POS tagging process")   
    doc = nlp(text)
    tokens = [(token.text, token.pos_) for token in doc]
    
    logger.debug("Starting the token classification process")
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
    for token, pos in tokens:
        if pos == "VERB":
            verb.append(token)
        elif pos == "AUX":
            aux.append(token)
        elif pos == "NOUN":
            noun.append(token)
        elif pos == "PROPN":
            propn.append(token)
        elif pos == "PRON":
            pron.append(token)
        elif pos == "ADJ":
            adj.append(token)
        elif pos == "ADV":
            adv.append(token)
        elif pos == "SCONJ":
            sconj.append(token)
        elif pos == "PART":
            part.append(token)
        elif pos == "ORG":
            org.append(token)
    return verb, aux, noun, propn, pron, adj, adv, sconj, part, org

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
    