# Import Statements
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain import OpenAI, VectorDBQA  # VectorDBQA is deprecated
from langchain.document_loaders import DirectoryLoader
from langchain.chains import RetrievalQA  # Using instead of VDBQA
import os
import nltk
from langchain import PromptTemplate
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.tools import BaseTool
from langchain.llms import OpenAI
from langchain import LLMMathChain, SerpAPIWrapper
from langchain.memory import ConversationBufferMemory

# Setting API Key
os.environ['OPENAI_API_KEY'] = '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'

# Loading the Documents and Splitting the test
loader = DirectoryLoader('../AG_BOT_V1/', glob='**/*.txt')
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

# Creating the docsearch and Embeddings
embeddings = OpenAIEmbeddings(openai_api_key=os.environ['OPENAI_API_KEY'])
docsearch = Chroma.from_documents(
    texts, embeddings, collection_name="Ashneer1")

# Creating the prompt template
template = """
Imagine You are Ashneer Grover. 
Answer this question {question} as if you are Ashneer Grover.
"""

prompt = PromptTemplate(
    input_variables=["question"],
    template=template,
)

# Creating the agent chain
Ashneer1 = RetrievalQA.from_chain_type(
    llm=OpenAI(), chain_type="stuff", retriever=docsearch.as_retriever()
)

# Creating the toolset for the agent
tools = [Tool(

    name="AG_BOT",
    func=Ashneer1.run,
    description="Answers questions as Ashneer"

)]

# Buffer memory for the agent
memory = ConversationBufferMemory(memory_key="chat_history")

# Initialising the agent
agent = initialize_agent(
    tools, llm=OpenAI(), agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True, memory=memory
)

# Defining a function


def ask_ashneer(user_query):

    query = prompt.format(question=user_query)
    agent.run(query)


if __name__ == "__main__":

    print("Hello Main")  # Debug Statement

    flag = True

    while (flag == True):
        print("Ask your question: \n")
        userQuestion = input()
        ask_ashneer(userQuestion)

        print("Any other questions? : \n")
        answer = input()
        if (answer == "no" or answer == "No"):
            flag = False
        else:
            continue
