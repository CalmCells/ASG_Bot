from flask import Flask, render_template, request, Response
import requests
import openai
import requests
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain import OpenAI, VectorDBQA #VectorDBQA is deprecated
from langchain.document_loaders import DirectoryLoader
from langchain.chains import RetrievalQA #Using instead of VDBQA
from langchain import PromptTemplate
import os
import nltk
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.tools import BaseTool
from langchain.llms import OpenAI
from langchain import LLMMathChain, SerpAPIWrapper
from langchain.memory import ConversationBufferMemory
from langchain.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)


os.environ['OPENAI_API_KEY'] = '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'
embeddings = HuggingFaceEmbeddings()

#Load the text documents for text retrieval
loader = DirectoryLoader('testdata', glob='**/*.txt')
documents = loader.load()

# Split the text documents into smaller chunks for processing
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

#load it into Chroma
db = Chroma.from_documents(texts, embeddings)

#save to disk
db2 = Chroma.from_documents(texts, embeddings, persist_directory="./chroma_db")

# load from disk
db3 = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)


Ashneer1 = RetrievalQA.from_chain_type(
        llm=OpenAI(), chain_type="stuff", retriever=db3.as_retriever()
    )

#Defining the tools for the agent
tools = [Tool(
    
    name = "AG_BOT",
    func = Ashneer1.run,
    description = "Answers questions as Ashneer"

)]


memory = ConversationBufferMemory(memory_key="chat_history")

#Initializing Agent
agent = initialize_agent(
    tools, llm=OpenAI(), agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True, memory=memory
)

#Preparing Query
query1 = "Where did you study?"
template = """ You are Ashneer Grover. Answer the {question} as Ashneer Grover"""
prompt = PromptTemplate(
    input_variables=["question"],
    template=template,
)
query = prompt.format(question=query1)  


#DIY Stlizing Model

def chat_with_model(message):
    context = "You are chatting with a bot which acts and responds like Ashneer Grover from India"
    prompt = f"Chat:\n{context}\nUser: {message}\n"

    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=50
    )

    reply = response.choices[0].text.strip()
    return reply

    

reply = chat_with_model(agent.run(query))