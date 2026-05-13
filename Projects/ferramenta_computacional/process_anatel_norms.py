import fitz  # PyMuPDF
import pandas as pd
import re
import os
from typing import List, Dict, Any
from text_preprocessing import preprocess_text # Importa a função de pré-processamento

# --- Funções Auxiliares de Análise ---

def get_hierarchical_level(text: str) -> int:
    """
    Determina o nível hierárquico de um item com base na sua numeração.
    Ex: "1." -> 1, "1.1." -> 2, "1.1.1." -> 3. Retorna 0 se não for um item.
    """
    match = re.match(r"^(\\d+(\\.\\d+)*)\\s", text.strip())
    if match:
        # Conta o número de pontos para determinar o nível
        return match.group(1).count(".") + 1
    return 0

def get_item_identifier(text: str) -> str:
    """Retorna o identificador de um item, como \'1.\' ou \'1.1.\'."""
    match = re.match(r"^(\\d+(\\.\\d+)*)\\s", text.strip())
    return match.group(1) if match else ""

# --- Funções Principais de Processamento ---

def extract_tables_as_chunks(page: fitz.Page, page_num: int, metadata_base: Dict) -> List[Dict]:
    """Extrai tabelas da página e as formata como chunks completos."""
    table_chunks = []
    tables = page.find_tables()
    if not tables:
        return []
        
    for i, table in enumerate(tables):
        try:
            df = table.to_pandas().dropna(how=\'all\', axis=0).dropna(how=\'all\', axis=1)
            if df.empty:
                continue
            
            table_markdown = df.to_markdown(index=False)
            
            # Aplica pré-processamento ao texto da tabela (Markdown)
            processed_table_text = preprocess_text(table_markdown)
            
            chunk_meta = metadata_base.copy()
            chunk_meta.update({
                "pagina": page_num,
                "tipo_conteudo": "tabela",
                "bbox": table.bbox
            })
            table_chunks.append({"texto": processed_table_text, "metadados": chunk_meta})
        except Exception as e:
            print(f"Aviso: Falha ao processar tabela {i+1} na página {page_num}. Erro: {e}")
            
    return table_chunks

def analyze_and_tag_blocks(doc: fitz.Document) -> List[Dict]:
    """
    Passada 1: Analisa todos os blocos de texto, extraindo conteúdo,
    tipo, nível hierárquico e metadados básicos.
    """
    tagged_blocks = []
    for page_num, page in enumerate(doc, 1):
        blocks = page.get_text("blocks", sort=True)
        for block in blocks:
            block_text = block[4].strip()
            if not block_text:
                continue
            
            # Aplica pré-processamento ao texto do bloco antes de analisar e marcar
            processed_block_text = preprocess_text(block_text)

            level = get_hierarchical_level(processed_block_text)
            identifier = get_item_identifier(processed_block_text) if level > 0 else ""
            
            tagged_blocks.append({
                "texto": processed_block_text,
                "pagina": page_num,
                "level": level,
                "identifier": identifier
            })
    return tagged_blocks

def group_and_create_chunks(tagged_blocks: List[Dict], metadata_base: Dict) -> List[Dict]:
    """
    Passada 2: Agrupa os blocos marcados em chunks hierárquicos.
    """
    final_chunks = []
    if not tagged_blocks:
        return []

    current_chunk_blocks = []
    
    for block in tagged_blocks:
        # Se encontrarmos um item de nível 1, finalizamos o chunk anterior e começamos um novo.
        if block["level"] == 1:
            if current_chunk_blocks:
                # Agrupa o texto e os metadados do chunk finalizado
                chunk_text = "\n\n".join([b["texto"] for b in current_chunk_blocks])
                start_page = current_chunk_blocks[0]["pagina"]
                end_page = current_chunk_blocks[-1]["pagina"]
                
                chunk_meta = metadata_base.copy()
                chunk_meta.update({
                    "tipo_conteudo": "seccao_texto",
                    "item_pai": current_chunk_blocks[0]["identifier"],
                    "pagina_inicio": start_page,
                    "pagina_fim": end_page
                })
                final_chunks.append({"texto": chunk_text, "metadados": chunk_meta})
            
            # Inicia o novo chunk
            current_chunk_blocks = [block]
        else:
            # Se não for um item de nível 1, apenas anexa ao chunk atual
            current_chunk_blocks.append(block)
            
    # Não se esqueça de adicionar o último chunk que estava sendo construído
    if current_chunk_blocks:
        chunk_text = "\n\n".join([b["texto"] for b in current_chunk_blocks])
        start_page = current_chunk_blocks[0]["pagina"]
        end_page = current_chunk_blocks[-1]["pagina"]
        
        chunk_meta = metadata_base.copy()
        chunk_meta.update({
            "tipo_conteudo": "seccao_texto",
            "item_pai": current_chunk_blocks[0]["identifier"],
            "pagina_inicio": start_page,
            "pagina_fim": end_page
        })
        final_chunks.append({"texto": chunk_text, "metadados": chunk_meta})
        
    return final_chunks

def process_document_hierarchically(pdf_path: str) -> List[Dict]:
    """
    Orquestra o processo completo de chunking hierárquico para um PDF.
    """
    doc = fitz.open(pdf_path)
    filename = os.path.basename(pdf_path)
    norma_id_match = re.search(r"Ato_(\\d+)", filename, re.IGNORECASE)
    norma_id = f"Ato {norma_id_match.group(1)}" if norma_id_match else filename.replace(\".pdf\", \"\")

    metadata_base = {"fonte_arquivo": filename, "norma_id": norma_id}
    
    # 1. Extrair todas as tabelas como chunks separados primeiro
    all_table_chunks = []
    for page_num, page in enumerate(doc, 1):
        all_table_chunks.extend(extract_tables_as_chunks(page, page_num, metadata_base))
        
    # 2. Analisar e marcar todos os blocos de texto
    tagged_text_blocks = analyze_and_tag_blocks(doc)
    
    # 3. Agrupar os blocos de texto em chunks de seção hierárquicos
    hierarchical_text_chunks = group_and_create_chunks(tagged_text_blocks, metadata_base)
    
    doc.close()
    
    # 4. Combinar os chunks de tabela e de texto
    return all_table_chunks + hierarchical_text_chunks

# --- Execução ---

if not os.path.exists("normas_pdf"):
    os.makedirs("normas_pdf")
    print("Pasta \'normas_pdf\' criada. Por favor, adicione seus PDFs nela e rode o script novamente.")
else:
    # Processa o primeiro PDF encontrado na pasta para demonstração
    pdf_files = [f for f f in os.listdir("normas_pdf") if f.lower().endswith(".pdf")]
    if not pdf_files:
        print("Nenhum PDF encontrado na pasta \'normas_pdf\'.")
    else:
        target_pdf = os.path.join("normas_pdf", pdf_files[0])
        print(f"--- Processando: {pdf_files[0]} com Agrupamento Hierárquico ---")
        
        final_chunks = process_document_hierarchically(target_pdf)
        
        print(f"\nProcessamento concluído. Total de chunks gerados: {len(final_chunks)}")
        
        # Exemplo de visualização dos resultados
        print("\n--- AMOSTRA DE CHUNKS GERADOS ---")
        
        # Mostra uma tabela e uma seção de texto como exemplo
        tabela_exemplo = next((c for c in final_chunks if c["metadados"]["tipo_conteudo"] == "tabela"), None)
        seccao_exemplo = next((c for c in final_chunks if c["metadados"]["tipo_conteudo"] == "seccao_texto"), None)

        if tabela_exemplo:
            print("\n[EXEMPLO DE CHUNK DE TABELA]")
            print(f"  METADADOS: {tabela_exemplo[\"metadados\"]}")
            print(f"  TEXTO (Markdown):\n{tabela_exemplo[\"texto\"]}")
        
        if seccao_exemplo:
            print("\n[EXEMPLO DE CHUNK DE SEÇÃO DE TEXTO]")
            print(f"  METADADOS: {seccao_exemplo[\"metadados\"]}")
            print(f"  TEXTO (início): {seccao_exemplo[\"texto\"][:500].strip()}...")


