import re
import unicodedata

def preprocess_text(text: str) -> str:
    """
    Normaliza o texto para melhorar a qualidade dos embeddings e da recuperação.
    Inclui: minúsculas, remoção de acentos, remoção de quebras de linha e múltiplos espaços.
    """
    if not isinstance(text, str): # Garante que a entrada é uma string
        return ""

    # 1. Converter para minúsculas
    text = text.lower()

    # 2. Remover acentos (normalização Unicode)
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')

    # 3. Remover quebras de linha e tabulações, substituindo por espaço
    text = text.replace('\n', ' ').replace('\t', ' ')

    # 4. Remover múltiplos espaços, substituindo por um único espaço
    text = re.sub(r'\s+', ' ', text).strip()

    # 5. Opcional: Remover pontuação (manter para documentos legais pode ser útil)
    # text = re.sub(r'[^a-z0-9\s]', '', text)

    return text


