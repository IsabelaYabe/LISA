import spacy 
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "src")))
from logger import setup_logger

logger = setup_logger()

class Requirement:
    def __init__(self, id, key, text, nlp = spacy.load("en_core_web_sm")):
        self.id = id
        self._key = key
        self._text = text
        self._nlp = nlp
        self._doc = self._nlp(self._text)
        self._verb, self._aux, self._noun, self._propn, self._pron, self._adj, self._adv, self._sconj, self._part, self._org = self.__extract_token_pos()

    def __extract_token_pos(self):
        logger.debug("Starting the tokenization and POS tagging process")   

        verb = [] # Verbo (should provkeye, enable, etc.)
        aux = [] # Auxiliar (should, must, can, is, are, etc.)
        noun = [] # Substantivo (system, user, report, etc.)
        propn = [] # Substantivo próprio (Nome da empresa, nome de funcionário, etc.)
        pron = [] # Pronome (user, it, they, he, she, etc.)
        adj = [] # Adjetivo (detailed, new, old, etc.)
        adv = [] # Advérbio (automatically, only, quickly, slowly, etc.)
        sconj = [] # Conjunção subordinada (so, that, when, if, because, etc.)
        part = [] # Particípio (to provkeye, to allow, should be, should have, etc.)
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
    
    @property
    def key(self):
        return self._key
    
    @key.setter
    def key(self, key):
        self._key = key

    @property
    def text(self):
        return self._text
    
    @text.setter
    def text(self, text):
        self._text = text
        self._doc = self._nlp(self._text)
        self._verb, self._aux, self._noun, self._propn, self._pron, self._adj, self._adv, self._sconj, self._part, self._org = self.__extract_token_pos()

    @property
    def nlp(self):
        return self._nlp
    
    @nlp.setter
    def nlp(self, nlp):
        self._nlp = nlp
        self._doc = self._nlp(self._text)
        self._verb, self._aux, self._noun, self._propn, self._pron, self._adj, self._adv, self._sconj, self._part, self._org = self.__extract_token_pos()

    @property
    def doc(self):
        return self._doc
    
    @property
    def verb(self):
        return self._verb
    
    @property
    def aux(self):
        return self._aux
    
    @property
    def noun(self):
        return self._noun
    
    @property
    def propn(self):
        return self._propn
    
    @property
    def pron(self):
        return self._pron
    
    @property
    def adj(self):
        return self._adj
    
    @property
    def adv(self):
        return self._adv
    
    @property
    def sconj(self):
        return self._sconj
    
    @property
    def part(self):
        return self._part
    
    @property
    def org(self):
        return self._org