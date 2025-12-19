from langchain_community.document_loaders import TextLoader

# text to write to a local file
# taken from https://www.theverge.com/2023/3/14/23639313/google-ai-language-model-palm-api-challenge-openai
text = """Google opens up its AI language model PaLM to challenge OpenAI and GPT-3
Google is offering developers access to one of its most advanced AI language models: PaLM.
The search giant is launching an API for PaLM alongside a number of AI enterprise tools
it says will help businesses “generate text, images, code, videos, audio, and more from
simple natural language prompts.”

PaLM is a large language model, or LLM, similar to the GPT series created by OpenAI or
Meta’s LLaMA family of models. Google first announced PaLM in April 2022. Like other LLMs,
PaLM is a flexible system that can potentially carry out all sorts of text generation and
editing tasks. You could train PaLM to be a conversational chatbot like ChatGPT, for
example, or you could use it for tasks like summarizing text or even writing code.
(It’s similar to features Google also announced today for its Workspace apps like Google
Docs and Gmail.)
"""

# write text to local file
#with open("my_file.txt", "w") as file:
    #file.write(text)

# use TextLoader to load text from local file
loader = TextLoader("my_file.txt")
docs_from_file = loader.load()

from langchain.text_splitter import CharacterTextSplitter

# create a text splitter
text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=20)

# split documents into chunks
docs = text_splitter.split_documents(docs_from_file)

#print(docs[0].page_content)
#print(docs[1].page_content)

from langchain_community.embeddings import HuggingFaceEmbeddings
# create embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

from langchain.vectorstores import DeepLake
import os
from dotenv import load_dotenv
load_dotenv()
os.environ["ACTIVELOOP_TOKEN"] = os.getenv("ACTIVELOOP_TOKEN")
my_activeloop_org_id = "ncminhbka"
my_activeloop_dataset_name = "langchain_course_indexers_retrievers"
dataset_path = f"hub://{my_activeloop_org_id}/{my_activeloop_dataset_name}"
db = DeepLake(dataset_path=dataset_path, embedding_function=embeddings)

# add documents to our Deep Lake dataset
db.add_documents(docs)

from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM
retriever = db.as_retriever()
# create a retrieval chain
qa_chain = RetrievalQA.from_chain_type(
	llm=OllamaLLM(model="gemma3:1b"),
	chain_type="stuff",
	retriever=retriever
)

query = "How Google plans to challenge OpenAI?"
response = qa_chain.invoke(query)
print(response)