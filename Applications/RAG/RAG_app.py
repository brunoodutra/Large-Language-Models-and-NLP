import streamlit as st
import os
import torch
from transformers import AutoTokenizer
from langchain_ollama.llms import OllamaLLM
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings.ollama import OllamaEmbeddings

from langchain_community.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
import pdfplumber
import time

# Initialize session state for chat history and document processing flag
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'document_processed' not in st.session_state:
    st.session_state.document_processed = False

# Configuração Inicial
#st.set_page_config(page_title="Chat com PDF", layout="wide")
st.title("Ultrasound Question Answering")

# Verificação de GPU
if torch.cuda.is_available():
    st.write(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    st.write("Using CPU")

# Funções
def read_pdf(file):
    with pdfplumber.open(file) as pdf:
        text = "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())
    return text

def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    return splitter.split_text(text)

def get_conversation_chain(vectorstore, llm, prompt, K=4):
    
        memory1 = ConversationBufferMemory(
            # Set params from input variables list
            memory_key="history",
            return_messages=True,
            input_key="question",
        )
        memory2 = ConversationBufferMemory(
            memory_key="history",
            return_messages=True,
            output_key="result",
        )
        
        conversation_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type='stuff',
            retriever=vectorstore.as_retriever(k=K),
            return_source_documents=True,
            chain_type_kwargs={
                "verbose": True,
                "prompt": prompt,
                "memory": memory1,
            },
            memory = memory2
        )

        return conversation_chain
    
def query_with_history(chain, question):
    chat_history = chain.memory.load_memory_variables({})    
    formatted_history = "\n".join([
        f"{msg.type.capitalize()}: {msg.content}" for msg in chat_history['history']
    ])
    

    if chat_history['history'] !=[]:
        complete_prompt = f"Contexto do histórico:\n{formatted_history}\n\nPergunta atual: {question}"
    else:
        complete_prompt = question
        
    response = chain({"query": complete_prompt})
    
    return response


# Interface do Usuário
#------------------------------------------------------------------
# Listar Bancos Existentes
db_directory = "db"
if not os.path.exists(db_directory):
    os.makedirs(db_directory)

bancos_disponiveis = [d for d in os.listdir(db_directory) if os.path.isdir(os.path.join(db_directory, d))]

# Opção do Usuário
st.subheader("Escolha a Fonte de Dados")
opcao = st.radio(
    "Deseja usar um banco existente ou criar um novo?",
    options=["Usar banco existente", "Criar novo banco com um arquivo"]
)

vectorstore = None

# Se o usuário escolher um banco existente
if opcao == "Usar banco existente":
    if bancos_disponiveis:
        banco_selecionado = st.selectbox("Selecione o banco:", options=bancos_disponiveis)
        if st.button("Carregar banco selecionado"):
            with st.status(f"Carregando banco '{banco_selecionado}'..."):
                st.session_state.vectorstore = Chroma(
                    persist_directory=os.path.join(db_directory, banco_selecionado),
                    embedding_function=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
                )
                st.success(f"Banco '{banco_selecionado}' carregado com sucesso!")
                
                st.session_state.document_processed = True
    else:
        st.warning("Nenhum banco existente encontrado. Faça o upload de um novo arquivo para criar um banco.")

# Se o usuário optar por criar um novo banco
elif opcao == "Criar novo banco com um arquivo":
    uploaded_file = st.file_uploader("Upload your PDF", type="pdf")
    if uploaded_file is not None:
        novo_banco_nome = st.text_input("Nome do novo banco:", value=uploaded_file.name.split(".")[0])
        if st.button("Criar banco"):
            with st.status("Processando documento e criando banco..."):
                pdf_text = read_pdf(uploaded_file)
                chunks = get_text_chunks(pdf_text)
                documents = [Document(page_content=chunk) for chunk in chunks]

                embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                novo_banco_path = os.path.join(db_directory, novo_banco_nome)
                st.session_state.vectorstore = Chroma.from_documents(documents=documents, embedding=embeddings, persist_directory=novo_banco_path)
                st.session_state.vectorstore.persist()

                st.success(f"Banco '{novo_banco_nome}' criado com sucesso e salvo em '{novo_banco_path}'!")
                st.session_state.document_processed = True
                
#-----------------------------------------------------------------

    
# Load or create conversation chain
if 'qa_chain' not in st.session_state and 'vectorstore' in st.session_state:

    llm = OllamaLLM(model="llama3-chatqa:70b", temperature=0.2)
    prompt_template = PromptTemplate(
        input_variables=["history", "context", "question"],
        template="""
        Use the following parts of the Context and History to answer the user's question.
        Be sure to cite the page number or refernce in your answer, e.g., [pag.34], [section 5 ].
        Do your response based in the Language used in the "User:" and  in the "History:".

        Context: {context}
        History: {history}
        User: {question}
        Chatbot:
        """
    )

    st.session_state.qa_chain=get_conversation_chain(st.session_state.vectorstore, llm, prompt_template, K=4)


# Exibir o histórico completo da conversa
for chat in st.session_state.chat_history:
    with st.chat_message(chat["role"]):
        st.markdown(chat["message"])
# Input do usuário
if user_input := st.chat_input("Ask a question about the document"):
    # Adicionar mensagem do usuário ao histórico
    user_message = {"role": "user", "message": user_input}
    st.session_state.chat_history.append(user_message)
    
    with st.chat_message("user"):
        st.markdown(user_input)

    # Resposta do assistente
    with st.chat_message("assistant"):
        with st.spinner("Assistant is typing..."):
            response = st.session_state.qa_chain({"query": user_input})
        message_placeholder = st.empty()
        full_response = ""
        for chunk in response['result'].split():
            full_response += chunk + " "
            time.sleep(0.05)  # Efeito de digitação
            message_placeholder.markdown(full_response + "▌")  # Cursor piscando
        message_placeholder.markdown(full_response)  # Resposta final

    # Salvar resposta do assistente no histórico
    assistant_message = {"role": "assistant", "message": response['result']}
    st.session_state.chat_history.append(assistant_message)

