import pandas as pd
import spacy
import re
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), 'src')))
from logger import setup_logger

logger = setup_logger()

def extract_user_story_parts(text):
    pattern = r"As (.*?), I want (.*?) so that (.*?)(?:\.|$)"
    user_story = []
    user_story_parts = []
    logger.debug(f"Extracting user story parts from text: {text}")
    try:
        matches = list(re.findall(pattern, text, re.IGNORECASE))
        
        num_matches = len(matches)
        for i in range(num_matches):            
            logger.debug(f"User story parts: As {matches[i][0]}, I want {matches[i][1]} so that {matches[i][2]}.")
            
            user_story_parts.append(f"As {matches[i][0]}, I want {matches[i][1]} so that {matches[i][2]}.")
        
            user_story.append({
                "type of user": "".join(matches[i][0].strip()[1:]),
                "goal": matches[i][1],
                "reason": matches[i][2]
            })
            logger.debug(f"type of user: {user_story[-1]["type of user"]}, goal: {user_story[-1]["goal"]}, reason: {user_story[-1]["reason"]}")
        return user_story, user_story_parts
    except IndexError:
        logger.error(f"Error extracting user story parts from text (r'As a (.*?), I want (.*?) so that (.*?)(?:\.|$)'): {text}")
        return [], []
    
def extract_subkeys(key):
    key_split = key.split(".")
    key_0 = ".".join(key_split[:2])
    key_1 = key_split[2]
    key_2 = ".".join(key_split[3:])

    return key_0, key_1, key_2

def extract_token_pos(text, nlp):
    doc = nlp(text)
    
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
    return verb, aux, noun, propn, pron, adj, adv, sconj, part, org

def extract_pos_from_user_story_list(user_story_list, nlp):
    verb, aux, noun, propn, pron, adj, adv, sconj, part, org = [], [], [], [], [], [], [], [], [], []
    for story in user_story_list:
        v, a, n, p, pr, aj, ad, sc, pa, o = extract_token_pos(story, nlp)
        verb.append(v)
        aux.append(a)
        noun.append(n)
        propn.append(p)
        pron.append(pr)
        adj.append(aj)
        adv.append(ad)
        sconj.append(sc)
        part.append(pa)
        org.append(o)
    
    return pd.Series([verb, aux, noun, propn, pron, adj, adv, sconj, part, org])

if __name__ == "__main__":
    logger.debug("Starting the dataset preparation process")
    pure_req_user_stories_path = os.path.join("data", "pure_req_user_stories.csv")
    pure_req_us_df = pd.read_csv(pure_req_user_stories_path)
    
    logger.debug("Renameing columns")
    pure_req_us_df.rename(columns={"databricks-llama-4-maverick": "user story llama-4-maverick", "databricks-meta-llama-3-3-70b-instruct": "user story llama-3-3-70b"}, inplace=True)
    
    logger.debug("Extracting subkeys from the keys column")
    pure_req_us_df[["keys_0", "keys_1", "keys_2"]] = pure_req_us_df["keys"].apply(extract_subkeys).apply(pd.Series)

    nlp = spacy.load("en_core_web_sm")
    logger.debug("Extracting pos tags from requirements")
    pure_req_us_df[["verb req", "aux req", "noun req", "propn req", "pron req", "adj req", "adv req", "sconj req", "part req", "org req"]] = pure_req_us_df["requirement"].apply(lambda x: extract_token_pos(x, nlp)).apply(pd.Series)

    nlp = spacy.load("en_core_web_sm")
    logger.debug("Extracting user story parts llama-4-maverick")

    pure_req_us_df[["user story (us llama-4-maverick)", "user story parts (us llama-4-maverick)"]] = (
    pure_req_us_df["user story llama-4-maverick"].apply(extract_user_story_parts).apply(pd.Series))

    logger.debug("Extracting pos tags from user story llama-4-maverick")
    pure_req_us_df[["verb us llama-4-maverick", "aux us llama-4-maverick", "noun us llama-4-maverick",
                "propn us llama-4-maverick", "pron us llama-4-maverick", "adj us llama-4-maverick",
                "adv us llama-4-maverick", "sconj us llama-4-maverick", "part us llama-4-maverick",
                "org us llama-4-maverick"]] = pure_req_us_df["user story parts (us llama-4-maverick)"]\
    .apply(lambda story_list: extract_pos_from_user_story_list(story_list, nlp))


    logger.debug("Extracting user story parts llama-3-3-70b")
    pure_req_us_df[["user story (us llama-3-3-70b)", "user story parts (us llama-3-3-70b)"]] = (
    pure_req_us_df["user story llama-3-3-70b"].apply(extract_user_story_parts).apply(pd.Series))

    logger.debug("Extracting pos tags from user story llama-3-3-70b")
    pure_req_us_df[["verb us llama-3-3-70b", "aux us llama-3-3-70b", "noun us llama-3-3-70b",
                "propn us llama-3-3-70b", "pron us llama-3-3-70b", "adj us llama-3-3-70b",
                "adv us llama-3-3-70b", "sconj us llama-3-3-70b", "part us llama-3-3-70b",
                "org us llama-3-3-70b"]] = pure_req_us_df["user story parts (us llama-3-3-70b)"]\
    .apply(lambda story_list: extract_pos_from_user_story_list(story_list, nlp))


    logger.debug("Extracting subkeys from the keys column")
    pure_req_us_df[["keys_0", "keys_1", "keys_2"]] = pure_req_us_df["keys"].apply(extract_subkeys).apply(pd.Series)

    logger.debug("Reordering columns")
    columns_order = ["filename", "keys", "keys_0", "keys_1", "keys_2", "requirement", "verb req", "aux req", "noun req", "propn req", "pron req", "adj req", "adv req", "sconj req", "part req", "org req",
    "user story llama-4-maverick", "user story (us llama-4-maverick)", "user story parts (us llama-4-maverick)", "verb us llama-4-maverick", "aux us llama-4-maverick", "noun us llama-4-maverick", "propn us llama-4-maverick", "pron us llama-4-maverick", "adj us llama-4-maverick", "adv us llama-4-maverick", "sconj us llama-4-maverick", "part us llama-4-maverick", "org us llama-4-maverick", "user story llama-3-3-70b", "user story (us llama-3-3-70b)", "user story parts (us llama-3-3-70b)", "verb us llama-3-3-70b", "aux us llama-3-3-70b", "noun us llama-3-3-70b", "propn us llama-3-3-70b", "pron us llama-3-3-70b", "adj us llama-3-3-70b", "adv us llama-3-3-70b", "sconj us llama-3-3-70b", "part us llama-3-3-70b", "org us llama-3-3-70b"]

    pure_req_us_df = pure_req_us_df[columns_order]
    logger.debug("Saving the dataset")
    pure_req_us_df.to_csv(os.path.join("data", "pure_req_user_stories_annotateSS.csv"), index=False)