import re
from pathlib import Path
import pandas as pd
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
from typing import Dict, Any
from typing import List
from data_prepare import RequirementDocumentation, Metadata
        
def carregar_documento(caminho_txt):
    with open(caminho_txt, 'r', encoding='utf-8') as f:
        return f.read()

def chunk_por_secao(texto):
    # Regex que identifica seções numeradas (ex: 1.1, 1.2.3 etc.)
    padrao_secao = r'(?=\n?\d+(\.\d+)*\s+[A-Z])'
    chunks = re.split(padrao_secao, texto)
    chunks = [c.strip() for c in chunks if len(c.strip()) > 100]  # evita pedaços muito curtos
    return chunks

def chunk_por_tamanho(texto, max_tokens=300):
    # Alternativa: chunking por tamanho usando sentenças
    sentencas = sent_tokenize(texto)
    chunks, chunk_atual = [], []
    total_tokens = 0

    for s in sentencas:
        tokens = len(s.split())
        if total_tokens + tokens > max_tokens:
            chunks.append(" ".join(chunk_atual))
            chunk_atual = [s]
            total_tokens = tokens
        else:
            chunk_atual.append(s)
            total_tokens += tokens

    if chunk_atual:
        chunks.append(" ".join(chunk_atual))

    return chunks

# Exemplo de uso:
caminho = "lps_srs.txt"  # substitua com seu caminho
texto = carregar_documento(caminho)

chunks_secao = chunk_por_secao(texto)
chunks_tamanho = chunk_por_tamanho(texto, max_tokens=300)

# Salvar para inspeção
df = pd.DataFrame({"chunk_secao": chunks_secao})
df.to_csv("chunks_por_secao.csv", index=False)
