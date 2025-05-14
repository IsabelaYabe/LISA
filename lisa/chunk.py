import pandas as pd
import nltk
import re
nltk.download("punkt")
from nltk.tokenize import sent_tokenize
from lisa.logger import logger

class Chunker:
    def __init__(self, requirement_documentation_dataframe: pd.DataFrame, metadata_dataframe: pd.DataFrame):
        self.requirement_documentation_dataframe = requirement_documentation_dataframe
        self.metadata_dataframe = metadata_dataframe
        
    def find_section(self, text_document, section):
        
    def chunk_por_section(self):    
        stop = True    
        stop0 = True
        chunks = []
        
        for i, row in self.requirement_documentation_dataframe.iterrows():
            id_document = row["id"]
            filename_document = row["filename"]
            text_document = row["text"]    
        
            metadatas = self.metadata_dataframe[self.metadata_dataframe["req_doc_id"] == id_document]
            sections = metadatas.to_dict()
        
            if not sections:
                raise ValueError(f"Id not found: {id}")
            
            section_starts = []
            ids_section = list(sections["id"].values())
            sections = list(sections["text"].values())
            
            for i, section in enumerate(sections):
                fullname_section = f"{ids_section[i].id}. {section}"
                logger.debug(f"fullname_section: {fullname_section}")
                
                pattern = r"^\s*" + re.escape(fullname_section.strip()) + r"\s*$"

                matches = list(re.finditer(pattern, text_document, re.MULTILINE))
                len_matches = len(matches)
                
                if len_matches == 0:
                    pattern = r"^" + re.escape(section.strip()) + r"$"
                    matches = list(re.finditer(pattern, text_document))
                    len_matches = len(matches)
                    if len_matches == 0:
                        logger.error(f"Section not found: {section}")
                        raise ValueError(f"Section not found: {section}")
                
                #matches = list(re.finditer(r"\b" + re.escape(section) + r"\b", text_document))
                len_matches = len(matches)
                
                if len_matches == 0:
                    logger.error(f"Section not found: {section}")
                    raise ValueError(f"Section not found: {section}")
                elif len_matches == 2:
                    logger.debug(f"id_document: {id_document}")
                    logger.debug([m.group() for m in matches])
                    second_start = matches[1].start()
                    logger.debug(f"second_start: {second_start}")
                elif len_matches > 2:
                    logger.debug(f"id_document: {id_document}")
                    logger.debug(f"section: {section}")
                    logger.error(f"Multiple matches found for section: {section}")
                    raise ValueError(f"Multiple matches found for section: {section}")
                else:    
                    start_section = text_document.find(section)
                
                #if start_section != -1:
                #    section_starts.append(start_section)
                #else:
                    
                #    logger.debug(f"id_document: {id_document}")
                    
                #    logger.warning(f"Section not found: {section}")
                    
                #    raise ValueError(f"Section not found: {section}")

            #section_starts.sort(key=lambda x: x)
            
            #for i, section in enumerate(section_starts):
            #    if i == 0:
            #        new_chunk = text_document[:start_section]
            #        chunks.append(new_chunk)
            #    
            #    if section != section_starts[-1]:
            #        next_section = section_starts[i+1] 
            #        new_chunk = text_document[start_section:next_section]
            #        chunks.append(new_chunk)                   
            #        text_document = text_document[next_section:]
            #    else:
            #        new_chunk = text_document[start_section:]
            #        chunks.append(new_chunk)
                      
                              
        #for doc in self.requirement_documentation_dataframe:
        #    separadores = []
        #    doc_id = doc.id
        #    for metadata in self.metadata_dataframe:
        #        if doc_id == metadata.id:
        #            separadores.append(metadata.separador)

if __name__ == "__main__":
    from lisa.data_prepare import RequirementDocumentation, Metadata
    import os
    raw_text_doc_dir_path = os.path.join("data", "df", "df_req_docs.pkl")      
    doc_struct_dir_path = os.path.join("data", "df", "df_metadatas.pkl")   
    
    df_req_docs = pd.read_pickle(raw_text_doc_dir_path)
    df_metadatas = pd.read_pickle(doc_struct_dir_path)
    chunker = Chunker(df_req_docs, df_metadatas)
    chunker.chunk_por_section()        