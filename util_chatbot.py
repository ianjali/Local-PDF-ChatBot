from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from pinecone import ServerlessSpec
import os

#***Split the Data into Text Chunks****
def text_split(extracted_data):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    text_chunks=text_splitter.split_documents(extracted_data)
    return text_chunks

#***Extract Data From the PDF File***
def load_pdf_file(data):
    loader= DirectoryLoader(data,
                            glob="*.pdf",
                            loader_cls=PyPDFLoader)
    documents=loader.load()
    return documents

def download_hugging_face_embeddings():
    embeddings=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    return embeddings

def get_index_and_knowledge_base(text_chunks,pc_index_name,embeddings): 
    pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))  
    idx_list = [json_idx['name'] for json_idx in pc.list_indexes()]
    print(idx_list)
    #print(pc.list_indexes().['name'])
    if pc_index_name not in idx_list:
        pc.create_index(
            name=pc_index_name,
            dimension=384, 
            metric="cosine", 
            spec=ServerlessSpec(
                cloud="aws", 
                region="us-east-1"
            ) 
        ) 
        knowledge = PineconeVectorStore.from_documents(
            text_chunks,
            index_name=pc_index_name,
            embedding=embeddings
        )
    else:
        knowledge = PineconeVectorStore.from_existing_index(
        index_name=pc_index_name,
        embedding=embeddings#OllamaEmbeddings()
    )
    return pc,knowledge