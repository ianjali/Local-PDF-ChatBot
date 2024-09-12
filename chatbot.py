#%%
# from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
#! pip install langchain-pinecone
#https://docs.pinecone.io/guides/get-started/build-a-rag-chatbot
from langchain_community.llms import Ollama
# from langchain_community.embeddings import OllamaEmbeddings
#https://docs.pinecone.io/integrations/langchain
# from langchain.chains import RetrievalQAWithSourcesChain  
from langchain_community.llms import Ollama
from util_chatbot import *
import warnings
warnings.filterwarnings("ignore")
import os
import sys
load_dotenv()

#%%
#MODEL = "llama2"
MODEL = "llama3.1"
PINECONE_API_KEY= os.getenv("PINECONE_API_KEY")
PINECONE_API_ENV= os.getenv("PINECONE_ENV")
pc_index_name = 'pinecone-doc-384'
DIMENSIONS = 384

embeddings=download_hugging_face_embeddings()
extracted_data=load_pdf_file(data='data/')
text_chunks=text_split(extracted_data)
print("Length of Text Chunks", len(text_chunks))

pc,knowledge = get_index_and_knowledge_base(text_chunks,pc_index_name,embeddings)

if MODEL.startswith("gpt"):
    pass
    # model = ChatOpenAI(api_key=OPENAI_API_KEY, model=MODEL)
    # embeddings = OpenAIEmbeddings()
else:
    model_llm = Ollama(model=MODEL)
    #embeddings = OllamaEmbeddings()
# qa = RetrievalQA.from_chain_type(
#     llm=model_llm,
#     chain_type="stuff",
#     retriever=knowledge.as_retriever()
# )

#conversational AI
from langchain.chains import ConversationalRetrievalChain
qa = ConversationalRetrievalChain.from_llm(model_llm, retriever=knowledge.as_retriever())
while True:
    chat_history = []
    query = "Who is teaching Machine Learning?"
    query = input(f"Input Prompt: ")
    if query == 'exit':
        print('Exiting')
        sys.exit()
    if query == '':
        continue
    result = qa({"question":query, "chat_history":chat_history})
    print("Response: ", result['answer'])





#chat bot question answering
# query = "Welcome"
# while query:
#     query = input('Ask question about your PDF Document:')
#     if query:
#         print(f'' + query)
#         docs = knowledge.similarity_search(query, k=5)
#         print(f"Genrenating response.\n")
#         qa = RetrievalQA.from_chain_type(
#             llm=model_llm,
#             chain_type="stuff",
#             retriever=knowledge.as_retriever()
#         )
#         response = qa.invoke(query).get("result")
#         print(f'Hello, ' + response)
# print(f"Bye")     
# pc.delete_index(pc_index_name)
# print(f"index deleted")  