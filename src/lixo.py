def _extract_token_pos(text):
        doc = nlp(self.text)
        dict_to_pos_tag = {"VERB": [], "AUX": [], "NOUN": [], "PROPN": [], "PRON": [], "ADJ": [], "ADV": [], "SCONJ": [], "PART": [], "ORG": []}
   
        for token in doc:
            if token.pos_ in dict_to_pos_tag.keys():
                dict_to_pos_tag[token.pos_].append(token.text)
        
        return dict_to_pos_tag