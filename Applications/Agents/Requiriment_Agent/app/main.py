# app.py
import streamlit as st
import pandas as pd
import json
from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
from tqdm import tqdm
from fpdf import FPDF

from spire.doc import *
from spire.doc.common import *
from glob import glob
# Configuração inicial do Streamlit
st.title("Análise de Requisitos")
st.write("Carregue sua planilha de requisitos e obtenha avaliações automáticas.")

# Função para carregar os requisitos da planilha
def load_requirements(file):
    """
    Carrega os requisitos de uma planilha Excel.
    
    Args:
        file (str): Caminho para o arquivo Excel.
    
    Returns:
        pd.DataFrame: DataFrame contendo os requisitos carregados.
    
    Raises:
        ValueError: Se nenhuma aba contendo requisitos for encontrada ou se ocorrer um erro ao carregar os dados.
    """
    xls = pd.ExcelFile(file)
    try:
        # Carregar abas específicas
        palavra_chave = "requisito"
        resultado = [item for idx, item in enumerate(xls.sheet_names) if palavra_chave in item.lower()]
        requisito_columns = resultado[0]
        try:
            requisitos = pd.read_excel(xls, sheet_name=requisito_columns, skiprows=len(xls.sheet_names)+1)
            requisitos = requisitos.dropna(subset=['Requisito', 'Categoria'])
        except:
                requisitos = pd.read_excel(xls, sheet_name=requisito_columns, skiprows=len(xls.sheet_names))
                requisitos = requisitos.dropna(subset=['Requisito', 'Categoria'])

    except FileNotFoundError:
        raise FileNotFoundError(f"O arquivo '{file}' não foi encontrado.")
    except pd.errors.EmptyDataError:
        raise ValueError("O arquivo está vazio ou não contém dados válidos.")
    except Exception as e:
        raise ValueError(f"Erro ao processar o arquivo: {e}")
    return requisitos

# Função para avaliar requisitos individualmente com streaming
def avaliar_requisito_stream(requisito, req_id, llm, analysis_placeholder, historico_analise, prompt):
   
    prompt_template = PromptTemplate(input_variables=["requisito"], template=prompt)
    chain = prompt_template | llm
    
    # Streaming da resposta
    streamed_text = ""
    for chunk in chain.stream({"requisito": requisito}):
        streamed_text += chunk
        
        analysis_placeholder.markdown(f"{historico_analise} **ID:** {req_id} | **Requisito:** {requisito}\n\n{streamed_text}")
    
    # Adicionar ao histórico
    historico_analise += f"**ID:** {req_id} | **Requisito:** {requisito}\n\n{streamed_text}\n\n---\n\n"
    return streamed_text, historico_analise

def analisar_relacoes(docs_json, llm):
    prompt = """
        Analise os seguintes requisitos como um todo:
        {requisitos}
        Identifique:
        1. Redundâncias: Requisitos que se sobrepõem ou repetem informações.
        2. Dependências: Requisitos que dependem de outros para serem compreendidos ou implementados.
        3. Inconsistências: Requisitos que se contradizem ou usam terminologia inconsistente.
        Forneça uma análise detalhada e obrigatoriamente responda em português.
        """
    prompt_template = PromptTemplate(input_variables=["requisitos"], template=prompt)
    formatted_prompt = prompt_template.format(requisitos=docs_json)
    analise = llm(formatted_prompt)
    return analise

# Função para análise global com streaming
def analisar_relacoes_stream(docs_json, llm, analysis_placeholder, historico_analise):
    prompt = """
        Analise os seguintes requisitos como um todo:
        {requisitos}
        Identifique:
        1. Redundâncias: Requisitos que se sobrepõem ou repetem informações.
        2. Dependências: Requisitos que dependem de outros para serem compreendidos ou implementados.
        3. Inconsistências: Requisitos que se contradizem ou usam terminologia inconsistente.
        Forneça uma análise detalhada e obrigatoriamente responda em português.
        """
    prompt_template = PromptTemplate(input_variables=["requisitos"], template=prompt)
    chain = prompt_template | llm
    
    # Streaming da resposta
    streamed_text = ""
    for chunk in chain.stream({"requisitos": docs_json}):
        streamed_text += chunk
        
        # Atualizar o placeholder com o histórico completo
        analysis_placeholder.markdown(f"{historico_analise} ### Análise Global:\n{streamed_text}")
        # Atualizar o histórico com a análise global parcial
    historico_analise += f"### Análise Global:\n{streamed_text}"

    
    return streamed_text, historico_analise

# Função para gerar resumo estruturado e tabelas
def gerar_resumo_estruturado(analises_individuais, analise_global, llm):
    # Criar prompt para resumo estruturado
    prompt_resumo = """
    Com base nas análises individuais e na análise global fornecidas abaixo, gere um resumo estruturado e organize as informações em tabelas.
    
    Análises Individuais:
    {analises_individuais}
    
    Análise Global:
    {analise_global}
    
    Instruções:
    1. Crie uma tabela com os seguintes campos para cada requisito:
       - ID: Identificador do requisito.
       - Requisito: Descrição do requisito.
       - Clareza: Sim/Não.
       - Sentenças Curtas: Sim/Não.
       - Individualidade: Sim/Não.
       - Independência: Sim/Não.
       - Consistência: Sim/Não.
       - Completude: Sim/Não.
       - Verificabilidade: Sim/Não.
       - Repetição: Sim/Não.
       - Testável: Sim/Não (inclua o teste, se aplicável).
    
    2. Forneça um resumo geral destacando:
       - Principais problemas identificados.
       - Sugestões para melhorias.
       - Conclusões gerais sobre a qualidade dos requisitos.
    
    Resposta esperada:
    - Uma tabela com os campos mencionados acima.
    - Um resumo geral em texto.
    """
    prompt_template = PromptTemplate(input_variables=["analises_individuais", "analise_global"], template=prompt_resumo)
    formatted_prompt = prompt_template.format(analises_individuais=analises_individuais, analise_global=analise_global)
    
    # Gerar resumo estruturado
    resumo_estruturado = llm(formatted_prompt)
    return resumo_estruturado

# Função para converter Markdown em PDF/DOCX usando Spire.Doc
def convert_markdown_to_file(markdown_content, output_format="pdf", file_name="relatorio_analise"):
    # Salvar o conteúdo Markdown em um arquivo temporário
    temp_md_file = f"{file_name}.md"
    with open(temp_md_file, "w", encoding="utf-8") as file:
        file.write(markdown_content)
    
    # Criar um objeto Document
    document = Document()
    
    try:
        # Carregar o arquivo Markdown
        document.LoadFromFile(temp_md_file)
        
        # Definir o nome do arquivo de saída
        if output_format == "pdf":
            output_file = f"{file_name}.pdf"
            document.SaveToFile(output_file, FileFormat.PDF)
        elif output_format == "docx":
            output_file = f"{file_name}.docx"
            document.SaveToFile(output_file, FileFormat.Docx2016)
        else:
            raise ValueError("Formato de saída inválido. Escolha 'pdf' ou 'docx'.")
        
        # Liberar recursos
        document.Dispose()
        
        # Remover o arquivo Markdown temporário
        os.remove(temp_md_file)
        
        return output_file
    except Exception as e:
        st.error(f"Erro ao converter o arquivo: {e}")
        return None


# Interface principal
uploaded_file = st.file_uploader("Carregue sua planilha de requisitos (.xlsx)", type=["xlsx"])
if uploaded_file is not None:
    # Carregar os requisitos
    requisitos = load_requirements(uploaded_file)
    st.write("Requisitos carregados:")
    st.dataframe(requisitos)
    
    # Converter requisitos para JSON
    docs_list = []
    for _, row in requisitos.iterrows():
        doc = {
            "id": row['Req ID'],
            "requisito": row['Requisito']
        }
        docs_list.append(doc)
    docs_json = json.dumps(docs_list, indent=4, ensure_ascii=False)
    
    
        # Criar lista suspensa com os IDs dos requisitos
    requisito_ids = requisitos['Req ID'].tolist()
        
    prompt_review_modes = glob("prompts/*")
    st.session_state.review_mode = st.selectbox(
        "Selecione o modo de revisão:", prompt_review_modes
    )
    
    with open(st.session_state.review_mode, 'r') as file:
        prompt = file.read()
    
    # Usar st.session_state para armazenar o requisito selecionado
    if 'requisito_selecionado' not in st.session_state:
        st.session_state.requisito_selecionado = None

    # Lista suspensa para selecionar o requisito
    st.session_state.requisito_selecionado = st.selectbox(
        "Selecione um requisito para análise específica:", requisito_ids
    )
    
    analise_escolhida=""
    # Botões para iniciar análises
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Iniciar Análise Completa"):
            analise_escolhida="geral"
    with col2:
            if st.button("Analisar Requisito Específico"):
                
                requisito_selecionado = st.session_state.requisito_selecionado
                if requisito_selecionado:
                    analise_escolhida="individual"

        
    if analise_escolhida == "geral":
        with st.spinner("Analisando requisitos..."):
            # Placeholder para streaming
            analysis_placeholder = st.empty()

            # Variável para acumular o histórico de análises
            historico_analise = "### Análise Individual de cada Requisito:\n"

            # Carregar o modelo LLM
            llm_llama31 = Ollama(model="llama3.3:latest", temperature=0.1)
            #llm_llama31 = Ollama(model="llama3.1:8b", temperature=0.1)

            # Análise individual
            resultados_individual = []
            for req in tqdm(json.loads(docs_json)):
                analise, historico_analise = avaliar_requisito_stream(
                    req['requisito'], req['id'], llm_llama31, analysis_placeholder, historico_analise, prompt
                )
                resultados_individual.append(analise)

            # Análise global com streaming
            analise_global, historico_analise = analisar_relacoes_stream(
                docs_json, llm_llama31, analysis_placeholder, historico_analise
            )

            # Análise global
            #analise_global = analisar_relacoes(docs_json, llm_llama31)
            #historico_analise += f"### Análise Global:\n{analise_global}"
            #analysis_placeholder.markdown(historico_analise)
                            # Gerar resumo estruturado e tabelas
        resumo_estruturado = gerar_resumo_estruturado("\n".join(resultados_individual), analise_global, llm_llama31)

        # Exibir resultados finais
        st.subheader("Resumo Estruturado e Tabelas")
        st.markdown(resumo_estruturado)

        # Exibir resultados finais
        #st.subheader("Resultados da Análise")

        #st.write("### Análise Individual:")
        #for i, analise in enumerate(resultados_individual):
        #    id_req = json.loads(docs_json)[i]['id']
        #    requisito = json.loads(docs_json)[i]['requisito']
        #    st.write(f"**ID:** {id_req} | **Requisito:** {requisito}")
        #    st.write(analise)
        #    st.write("---")

        #st.write("### Análise Global:")
        #st.write(analise_global)

        # Opção para baixar os resultados
        resultado_texto = ""
        resultado_texto += "### Análise Individual:\n"
        for i, analise in enumerate(resultados_individual):
            id_req = json.loads(docs_json)[i]['id']
            requisito = json.loads(docs_json)[i]['requisito']
            resultado_texto += f"ID: {id_req} | Requisito: {requisito}\n{analise}\n\n"
        resultado_texto += "### Análise Global:\n"
        resultado_texto += analise_global

        st.download_button(
            label="Baixar Resultados em TXT",
            data=resultado_texto,
            file_name="analise_requisitos.txt",
            mime="text/plain"
        )

        # Opção para baixar os resultados em DOCX
        docx_file = convert_markdown_to_file(historico_analise, output_format="docx", file_name="relatorio_analise")
        if docx_file:
            with open(docx_file, "rb") as file:
                st.download_button(
                    label="Baixar Relatório em DOCX",
                    data=file,
                    file_name="relatorio_analise.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                )

    elif analise_escolhida == "individual": 
        with st.spinner(f"Analisando requisito {requisito_selecionado}..."):
            # Placeholder para streaming
            analysis_placeholder = st.empty()

            # Variável para acumular o histórico de análises
            historico_analise = ""

            # Carregar o modelo LLM
            llm_llama31 = Ollama(model="llama3.3:latest", temperature=0.1)
            #llm_llama31 = Ollama(model="llama3.1:8b", temperature=0.1)

            # Filtrar o requisito selecionado
            requisito_filtrado = requisitos[requisitos['Req ID'] == requisito_selecionado]['Requisito'].iloc[0]
            req_id = requisitos[requisitos['Req ID'] == requisito_selecionado]['Req ID'].iloc[0]
            print(requisito_filtrado)

            # Análise do requisito específico
            analise, historico_analise = avaliar_requisito_stream(
                requisito_filtrado, req_id, llm_llama31, analysis_placeholder, historico_analise, prompt
            )