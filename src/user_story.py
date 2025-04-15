import regex as re
import spacy
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), 'src')))
from logger import setup_logger

logger = setup_logger()

class UserStory:
    def __init__(self, id, text, nlp = spacy.load("en_core_web_sm")):
        self.id = id
        self.text = text
        self.type_of_user, self.goal, self.reason = self.extract_user_story_parts().values() 
        self.nlp = nlp 
        self.doc = self.nlp(self.text)
        self.verb, self.aux, self.noun, self.propn, self.pron, self.adj, self.adv, self.sconj, self.part, self.org = self.__extract_token_pos()

    def __extract_user_story_parts(self):
        logger.debug("Starting the user story extraction process")
        pattern = r"As a (.*?), I want (.*?) so that (.*?)(?:\.|$)"
        match = re.findall(pattern, self.text, re.IGNORECASE)[0]
        result = {
                "type_of_user": match[0],
                "goal": match[1],
                "reason": match[2]
            }
        logger.debug(f"Extracted user story: {result}")
        return result
    
    def __extract_token_pos(self):
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
        for token in self.doc:
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