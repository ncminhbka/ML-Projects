import warnings
warnings.filterwarnings("ignore")


#VECTOR STORE
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import DeepLake
embeddings = OllamaEmbeddings(model="gemma3:1b")
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA



import os
from dotenv import load_dotenv

# Load biến môi trường từ file .env
load_dotenv()

# Lấy key
activeloop_key = os.getenv("ACTIVELOOP_TOKEN")

from langchain_community.llms import Ollama
llm = Ollama(model="gemma3:1b", temperature=0)
embeddings = OllamaEmbeddings(model="nomic-embed-text")

text = [
    "Napoleon Bonaparte was born on August 15, 1769, in Corsica.",
    "Louis XIV was born in 5 September 1638",
]

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.create_documents(text)


# create Deep Lake dataset
# TODO: use your organization id here. (by default, org id is your username)
my_activeloop_org_id = "ncminhbka" 
my_activeloop_dataset_name = "langchain_course_from_zero_to_hero"
dataset_path = f"hub://{my_activeloop_org_id}/{my_activeloop_dataset_name}"
db = DeepLake(dataset_path=dataset_path, embedding_function=embeddings)

# add documents to our Deep Lake dataset
#db.add_documents(docs)

#create RetrievalQA chain
retrieval_qa = RetrievalQA.from_chain_type(
	llm=llm,
	chain_type="stuff",
	retriever=db.as_retriever()
)

#create an agent that uses the retrieval_qa chain
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType

tools = [
    Tool(
        name="Retrieval QA System",
        func=retrieval_qa.run,
        description="Useful for answering questions."
    ),
]

agent = initialize_agent(
	tools,
	llm,
	agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
	verbose=True
)

response = agent.invoke("When was Napoleone born?")
print(response)


# create new documents
texts = [
    "Lady Gaga was born in 28 March 1986",
    "Michael Jeffrey Jordan was born in 17 February 1963"
]
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.create_documents(texts)

# add documents to our Deep Lake dataset
db.add_documents(docs)

# ask questions again
response = agent.invoke("When was Lady Gaga born?")
print(response)