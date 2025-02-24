#!/usr/bin/env python
# coding: utf-8

# ### Libraries

# In[ ]:


import pandas as pd
import os, sys 


# In[2]:


from langchain_ollama.llms import OllamaLLM
from langchain.llms import HuggingFacePipeline
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document

from langchain_community.embeddings.ollama import OllamaEmbeddings

from langchain.memory import ConversationBufferMemory
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate


# In[3]:


from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import FAISS


# ### Load  requirements

# In[4]:


# Carregar o arquivo Excel
file_path = "Template - Requisito de Software v02.xlsx"
xls = pd.ExcelFile(file_path)

# Carregar abas específicas
sobre_documento = pd.read_excel(xls, sheet_name="Sobre o documento")
glossario = pd.read_excel(xls, sheet_name="1-Glossário")
requisitos = pd.read_excel(xls, sheet_name='2-Requisitos de Produto', skiprows= len(xls.sheet_names)+1)
lists = pd.read_excel(xls, sheet_name="3-Lists")
versoes = pd.read_excel(xls, sheet_name="4-Versões")
guia = pd.read_excel(xls, sheet_name="5-Guia")

requisitos= requisitos.dropna(subset=['Requisito','Categoria'])
# Exibir as primeiras linhas da aba de requisitos
#print(requisitos.head())

requisitos_texto = requisitos.to_string(index=False)


# In[5]:


requisitos.head()


# ### Format requirements in docs

# In[6]:


from langchain.docstore.document import Document

# Criar lista de documentos
docs = []
for _, row in requisitos.iterrows():
    content = f"ID: {row['Req ID']} | Categoria: {row['Categoria']} | Descrição Alto Nível: {row['Descrição Alto Nível']} | Requisito: {row['Requisito']}"
    docs.append(Document(page_content=content, metadata={"ID": row["Req ID"]}))


# In[7]:


import json
docs_list =[]  
for _, row in requisitos.iterrows():
    doc = {
        "id": row['Req ID'],
        "requisito": row['Requisito']
    }
    docs_list.append(doc)

# Convertendo para JSON
docs_json = json.dumps(docs_list, indent=4, ensure_ascii=False)


# ### LLM

# In[165]:


from langchain.llms import Ollama

llm_llama31 = Ollama(model="llama3.1:8b", temperature=0.1)
llm_llama3 = Ollama(model="llama3.3:latest", temperature=0.1)


# ###  Global relations analysis

# In[141]:


from langchain.prompts import PromptTemplate
import json
docs_list = []
def analisar_relacoes(docs_json, llm_reqs_glob):
    prompt = """
        Analise os seguintes requisitos como um todo:
        {requisitos}

        Identifique:
        1. Redundâncias: Requisitos que se sobrepõem ou repetem informações.
        2. Dependências: Requisitos que dependem de outros para serem compreendidos ou implementados.
        3. Inconsistências: Requisitos que se contradizem ou usam terminologia inconsistente.

        Forneça uma análise detalhada e obrigatoriamente responda em português.
        """

    prompt_template = PromptTemplate(
        input_variables=["requisitos"], template=prompt)

    prompt  = prompt_template.format(requisitos=docs_json)
    analise = llm_reqs_glob(prompt)
    return analise


# In[188]:


analise_global = analisar_relacoes(docs_json, llm_llama3)
print(analise_global)


# ### Individual Analysis

# In[167]:


def avaliar_requisito(requisito, llm_reqs_indiv):
    prompt = """
    Avalie o seguinte requisito com base nas boas práticas descritas abaixo, responda em português:
    
    Boas Práticas para Requisitos:
    1. Os requisitos devem ser claros, precisos e não genéricos.
    2. Os requisitos devem ter sentenças curtas.
    3. Os requisitos devem ser individuais.
    4. Os requisitos devem ser independentes.
    5. Os requisitos devem ser consistentes.
    6. Os requisitos devem ser completos.
    7. Os requisitos devem ser verificáveis.
    8. Os requisitos não devem ser repetidos ou redundantes.
    9. Os requisitos devem ser coerentes quanto ao conteúdo.
    10. Os requisitos não devem repetir várias vezes a mesma palavra.
    
    Exemplo 1:
    Requisito: "O sistema deve permitir que o usuário faça login usando seu e-mail e senha."
    
    Avaliação:
    Clareza: Sim, o requisito é claro e específico.
    Sentenças Curtas: Sim, a sentença é curta e direta.
    Individualidade: Sim, trata-se de um único requisito.
    Independência: Sim, este requisito não depende de outro.
    Consistência: Sim, o requisito é consistente com as boas práticas.
    Completude: Sim, o requisito descreve completamente a funcionalidade.
    Verificabilidade: Sim, é possível verificar se o login funciona com e-mail e senha.
    Repetição: Não há repetição desnecessária de palavras.
    
    Testável: Sim. Teste: Verificar se o usuário consegue fazer login inserindo e-mail e senha válidos.
    Classificação: Sistema (é uma funcionalidade do sistema).
    
    Exemplo 2:
    Requisito: "O sistema deve ser rápido."
    
    Avaliação:
    Clareza: Não, o requisito é vago e genérico. O que significa "rápido"?
    Sentenças Curtas: Sim, a sentença é curta.
    Individualidade: Sim, trata-se de um único requisito.
    Independência: Sim, este requisito não depende de outro.
    Consistência: Não, o requisito é inconsistente porque "rápido" não é definido.
    Completude: Não, o requisito não descreve completamente a funcionalidade.
    Verificabilidade: Não, não é possível verificar se o sistema é "rápido" sem uma definição clara.
    Repetição: Não há repetição desnecessária de palavras.
    
    Testável: Não. Não há como testar algo tão subjetivo.
    Classificação: Sistema (é uma característica geral do sistema).
    
    Exemplo 3:
    Requisito: "O sistema deve enviar uma notificação ao usuário quando houver uma nova mensagem."
    
    Avaliação:
    Clareza: Sim, o requisito é claro e específico.
    Sentenças Curtas: Sim, a sentença é curta e direta.
    Individualidade: Sim, trata-se de um único requisito.
    Independência: Sim, este requisito não depende de outro.
    Consistência: Sim, o requisito é consistente com as boas práticas.
    Completude: Sim, o requisito descreve completamente a funcionalidade.
    Verificabilidade: Sim, é possível verificar se a notificação é enviada.
    Repetição: Não há repetição desnecessária de palavras.
    
    Testável: Sim. Teste: Verificar se o usuário recebe uma notificação ao receber uma nova mensagem.
    Classificação: Sistema (é uma funcionalidade do sistema).
    
    Exemplo 4:
    Requisito: "O sistema deve ser fácil de usar."
    
    Avaliação:
    Clareza: Não, o requisito é vago e subjetivo. O que significa "fácil de usar"?
    Sentenças Curtas: Sim, a sentença é curta.
    Individualidade: Sim, trata-se de um único requisito.
    Independência: Sim, este requisito não depende de outro.
    Consistência: Não, o requisito é inconsistente porque "fácil de usar" não é definido.
    Completude: Não, o requisito não descreve completamente a funcionalidade.
    Verificabilidade: Não, não é possível verificar algo tão subjetivo.
    Repetição: Não há repetição desnecessária de palavras.
    
    Testável: Não. Não há como testar algo tão subjetivo.
    Classificação: Produto (é uma característica geral do produto).
    
    Agora, avalie o seguinte requisito:
    
    Requisito: {requisito}
    
    Sua resposta deve seguir o mesmo formato acima:
    
    Avaliação:
    Clareza: 
    Sentenças Curtas: 
    Individualidade: 
    Independência: 
    Consistência: 
    Completude: 
    Verificabilidade: 
    Repetição: 
    
    Testável (Identifique se o Requisito é Testável, se sim indique o teste):
    """
    prompt_template = PromptTemplate(
    input_variables=["requisito"], template=prompt)

    prompt  = prompt_template.format(requisito=requisito)
    analise = llm_reqs_indiv(prompt)
    return analise


# In[174]:


from tqdm import tqdm

resultados_individual = []

for req in tqdm(json.loads(docs_json)):
    analise = avaliar_requisito(req['requisito'], llm_llama31)
    resultados_individual.append(analise)

# Adicionar resultados ao DataFrame
requisitos['Análise Individual'] = resultados_individual


# In[176]:


resultados_individual[0]


# In[189]:


from tqdm import tqdm

resultados_individual = []

for req in tqdm(json.loads(docs_json)):
    analise = avaliar_requisito2(str(req))
    resultados_individual.append(analise[analise.find('\n</think>\n')+9:])

# Adicionar resultados ao DataFrame
requisitos['Análise Individual'] = resultados_individual


# In[190]:


with open('analise_completa.txt', 'w') as f:
    # Análise Individual
    f.write("### Análise Individual:\n")
    for i, analise in enumerate(resultados_individual):
        Id = json.loads(docs_json)[i]['id']
        requisito =  json.loads(docs_json)[i]['requisito']
        f.write(f"ID: {id} | Requisito {requisito}: \n{analise}\n\n")

    # Análise Global
    f.write("### Análise Global:\n")
    f.write(analise_global)

