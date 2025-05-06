def _extract_token_pos(text):
        doc = nlp(self.text)
        dict_to_pos_tag = {"VERB": [], "AUX": [], "NOUN": [], "PROPN": [], "PRON": [], "ADJ": [], "ADV": [], "SCONJ": [], "PART": [], "ORG": []}
   
        for token in doc:
            if token.pos_ in dict_to_pos_tag.keys():
                dict_to_pos_tag[token.pos_].append(token.text)
        
        return dict_to_pos_tag
    
    
def _dir_interable(self, dir, **kwargs):
    results = {key: [] for key in kwargs}
    
    for file in os.listdir(dir):
        for key, function in kwargs.items():
            try:
                results[key].append(function(dir, file))
            except Exception as e:
                logger.error(f"Error in {key} for file {file}: {e}")
                results[key].append(None)
    return results