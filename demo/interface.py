import sys, os
sys.dont_write_bytecode = True

from dotenv import load_dotenv
import time
import streamlit as st
import pandas as pd
import os
import io
from pypdf import PdfReader

# Ajouter le répertoire courant au chemin Python
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from pdf_utils import process_pdf_files
import pandas as pd

# Ajouter le répertoire courant au chemin Python
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from pdf_utils import process_pdf_files
import pandas as pd
import streamlit as st
import openai
from streamlit_modal import Modal

from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.faiss import DistanceStrategy
from langchain_community.embeddings import HuggingFaceEmbeddings

from llm_agent import ChatBot
from ingest_data import ingest
from retriever import SelfQueryRetriever
import chatbot_verbosity as chatbot_verbosity

load_dotenv()

DATA_PATH = os.getenv("DATA_PATH")
FAISS_PATH = os.getenv("FAISS_PATH")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")

print(DATA_PATH)
print(FAISS_PATH)

st.set_page_config(page_title="Resume Screening GPT", layout="wide")

# Afficher le logo en haut de la page
col1, col2, col3 = st.columns([1, 2, 1])
with col2:  # Centrer le logo
    try:
        logo_path = os.path.join(os.path.dirname(__file__), "assets/logo.png")
        st.image(logo_path, use_column_width=True, output_format="PNG")
    except Exception as e:
        st.warning(f"Failed to load logo. Error: {str(e)}")


if "chat_history" not in st.session_state:
  st.session_state.chat_history = []

if "df" not in st.session_state:
  st.session_state.df = pd.read_csv(DATA_PATH)

if "embedding_model" not in st.session_state:
  st.session_state.embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, model_kwargs={"device": "cpu"})

if "rag_pipeline" not in st.session_state:
  vectordb = FAISS.load_local(FAISS_PATH, st.session_state.embedding_model, distance_strategy=DistanceStrategy.COSINE, allow_dangerous_deserialization=True)
  st.session_state.rag_pipeline = SelfQueryRetriever(vectordb, st.session_state.df)

if "resume_list" not in st.session_state:
  st.session_state.resume_list = []

def display_dataframe():
    st.subheader("")
    if not st.session_state.df.empty:
        st.write(f"Nombre total de CVs chargs : {len(st.session_state.df)}")
        st.dataframe(st.session_state.df.head(), use_container_width=True)
        
        # Afficher des statistiques de base
        st.subheader("")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("", len(st.session_state.df))
        with col2:
            st.metric("", ", ".join(st.session_state.df.columns.tolist()))


# Les fonctions de traitement PDF ont été déplacées dans pdf_utils.py

def clear_uploaded_files():
    """Reset the list of uploaded files"""
    st.session_state.uploaded_file = None
    st.session_state.df = pd.read_csv(DATA_PATH)
    vectordb = FAISS.load_local(FAISS_PATH, st.session_state.embedding_model, 
                              distance_strategy=DistanceStrategy.COSINE, 
                              allow_dangerous_deserialization=True)
    st.session_state.rag_pipeline = SelfQueryRetriever(vectordb, st.session_state.df)

def upload_file():
    modal = Modal(key="File Error Modal", title="File Error", max_width=500)
    
    try:
        # Vérifier si aucun fichier n'est téléchargé ou si la liste est vide
        if not hasattr(st.session_state, 'uploaded_file') or st.session_state.uploaded_file is None or \
           (isinstance(st.session_state.uploaded_file, list) and len(st.session_state.uploaded_file) == 0):
            st.session_state.df = pd.read_csv(DATA_PATH)
            vectordb = FAISS.load_local(FAISS_PATH, st.session_state.embedding_model, 
                                      distance_strategy=DistanceStrategy.COSINE, 
                                      allow_dangerous_deserialization=True)
            st.session_state.rag_pipeline = SelfQueryRetriever(vectordb, st.session_state.df)
            return
        
        # Vérifier si c'est un fichier unique ou multiple
        uploaded_files = st.session_state.uploaded_file
        if not isinstance(uploaded_files, list):
            uploaded_files = [uploaded_files]
        
        # Filtrer les fichiers None (peut arriver lors de la suppression)
        uploaded_files = [f for f in uploaded_files if f is not None]
        
        # Si plus aucun fichier valide après filtrage, réinitialiser
        if not uploaded_files:
            st.session_state.df = pd.read_csv(DATA_PATH)
            vectordb = FAISS.load_local(FAISS_PATH, st.session_state.embedding_model, 
                                      distance_strategy=DistanceStrategy.COSINE, 
                                      allow_dangerous_deserialization=True)
            st.session_state.rag_pipeline = SelfQueryRetriever(vectordb, st.session_state.df)
            return
        
        data = []
        
        # Séparer les fichiers PDF et CSV
        pdf_files = [f for f in uploaded_files if f is not None and hasattr(f, 'name') and f.name.lower().endswith('.pdf')]
        csv_files = [f for f in uploaded_files if f is not None and hasattr(f, 'name') and f.name.lower().endswith('.csv')]
        
        # Traiter d'abord tous les fichiers PDF en une seule fois
        if pdf_files:
            try:
                csv_data = process_pdf_files(pdf_files)
                if csv_data:  # Vérifier que des données ont été retournées
                    df = pd.read_csv(io.StringIO(csv_data))
                    data.extend(df.to_dict('records'))
            except Exception as e:
                with modal.container():
                    st.error(f"Erreur lors du traitement des fichiers PDF : {str(e)}")
                return
        
        # Puis traiter les fichiers CSV
        for file in csv_files:
            try:
                # Lire le fichier CSV
                df = pd.read_csv(file)
                if "Resume" not in df.columns or "ID" not in df.columns:
                    raise ValueError("Le fichier CSV doit contenir les colonnes 'ID' et 'Resume'.")
                data.extend(df.to_dict('records'))
            except Exception as e:
                with modal.container():
                    st.error(f"Error reading file {getattr(file, 'name', 'unknown')}: {str(e)}")
                continue
        
        # Si aucune donnée valide n'a été traitée
        if not data:
            st.session_state.df = pd.read_csv(DATA_PATH)
            vectordb = FAISS.load_local(FAISS_PATH, st.session_state.embedding_model, 
                                      distance_strategy=DistanceStrategy.COSINE, 
                                      allow_dangerous_deserialization=True)
            st.session_state.rag_pipeline = SelfQueryRetriever(vectordb, st.session_state.df)
            return
            
        # Créer un DataFrame avec les données extraites
        df_load = pd.DataFrame(data)
        st.session_state.df = df_load
        
        # Créer les embeddings
        try:
            vectordb = ingest(st.session_state.df, "Resume", st.session_state.embedding_model)
            # Configurer le pipeline de recherche
            st.session_state.rag_pipeline = SelfQueryRetriever(vectordb, st.session_state.df)
        except Exception as e:
            with modal.container():
                st.error(f"Error creating embeddings: {str(e)}")
            
    except Exception as error:
        with modal.container():
            st.error(f"Une erreur est survenue lors du traitement du fichier : {str(error)}")


def check_openai_api_key(api_key: str):
  openai.api_key = api_key
  try:
    _ = openai.chat.completions.create(
      model="gpt-4o-mini",  # Use a model you have access to
      messages=[{"role": "user", "content": "Hello!"}],
      max_tokens=3
    )
    return True
  except openai.AuthenticationError as e:
    return False
  else:
    return True
  
  
def check_model_name(model_name: str, api_key: str):
  openai.api_key = api_key
  model_list = [model.id for model in openai.models.list()]
  return True if model_name in model_list else False


def clear_message():
  st.session_state.resume_list = []
  st.session_state.chat_history = []


user_query = st.chat_input("Type your message here...")


with st.sidebar:
  st.markdown("# Control Panel")

  st.text_input("OpenAI's API Key", type="password", key="api_key")
  st.selectbox("RAG Mode", ["Generic RAG", "RAG Fusion"], placeholder="Generic RAG", key="rag_selection")
  st.text_input("GPT Model", "gpt-4o-mini", key="gpt_selection")
  
  # File type dropdown
  file_type = st.selectbox(
      "Select file type:",
      ["PDF", "CSV"],
      index=0,  # Default to PDF
      key="file_type"
  )
  
  # Update file_uploader based on selected file type
  if st.session_state.file_type == "PDF":
      uploaded_files = st.file_uploader(
          "Upload one or more PDF files",
          type=["pdf"],
          key="uploaded_file",
          on_change=upload_file,
          accept_multiple_files=True
      )
  else:
      uploaded_files = st.file_uploader(
          "Upload a CSV file",
          type=["csv"],
          key="uploaded_file",
          on_change=upload_file
      )
  
  # Selected files are managed automatically by Streamlit
  # User can remove files directly via the file_uploader interface
  
  # File deletion is handled directly by the file_uploader widget
  # using the 'x' next to the file name
  # Clear conversation button
  st.button("Clear conversation", on_click=clear_message, key="clear_chat_button")
  with st.expander("View data"):
    # Display only the table without the header
    if not st.session_state.df.empty:
        st.dataframe(st.session_state.df, use_container_width=True, hide_index=True)
        
        # Simplified download button
        csv = st.session_state.df.to_csv(index=False, encoding='utf-8-sig')
        st.download_button(
            label="Download CVs as CSV",
            data=csv,
            file_name="cv_analysis.csv",
            mime="text/csv"
        )


for message in st.session_state.chat_history:
  if isinstance(message, AIMessage):
    with st.chat_message("AI"):
      st.write(message.content)
  elif isinstance(message, HumanMessage):
    with st.chat_message("Human"):
      st.write(message.content)
  else:
    with st.chat_message("AI"):
      message[0].render(*message[1:])


if not st.session_state.api_key:
  st.stop()

if not check_openai_api_key(st.session_state.api_key):
  st.error("The API key is incorrect. Please set a valid OpenAI API key to continue. Learn more about [API keys](https://platform.openai.com/api-keys).")
  st.stop()

if not check_model_name(st.session_state.gpt_selection, st.session_state.api_key):
  st.error("The model you specified does not exist. Learn more about [OpenAI models](https://platform.openai.com/docs/models).")
  st.stop()


retriever = st.session_state.rag_pipeline

llm = ChatBot(
  api_key=st.session_state.api_key,
  model=st.session_state.gpt_selection,
)

if user_query is not None and user_query != "":
  with st.chat_message("Human"):
    st.markdown(user_query)
    st.session_state.chat_history.append(HumanMessage(content=user_query))

  with st.chat_message("AI"):
    start = time.time()
    with st.spinner("Generating answers..."):
      document_list = retriever.retrieve_docs(user_query, llm, st.session_state.rag_selection)
      query_type = retriever.meta_data["query_type"]
      st.session_state.resume_list = document_list
      stream_message = llm.generate_message_stream(user_query, document_list, [], query_type)
    end = time.time()

    response = st.write_stream(stream_message)
    
    retriever_message = chatbot_verbosity
    retriever_message.render(document_list, retriever.meta_data, end-start)

    st.session_state.chat_history.append(AIMessage(content=response))
    st.session_state.chat_history.append((retriever_message, document_list, retriever.meta_data, end-start))