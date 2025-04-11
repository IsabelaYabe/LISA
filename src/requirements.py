import spacy 
from src.logger import setup_logger

logger = setup_logger()

class Requirement:
    def __init__(self, id, text, nlp = spacy.load("en_core_web_sm")):
        self.id = id
        self.text = text
        self.nlp = nlp
        self.doc = self.nlp(self.text)
        self.verb, self.aux, self.noun, self.propn, self.pron, self.adj, self.adv, self.sconj, self.part, self.org = self.extract_token_pos(self.text, nlp)

    def extract_token_pos(self):
        logger.debug("Starting the tokenization and POS tagging process")   
        doc = self.nlp(self.text)
        tokens = [(token.self.text, token.pos_) for token in doc]

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
        logger.debug("Token classification completed")
        return verb, aux, noun, propn, pron, adj, adv, sconj, part, org
