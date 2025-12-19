from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.vectorstores import DeepLake
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import SeleniumURLLoader
from langchain.prompts import PromptTemplate
# URLs để crawl
urls = [
    "https://beebom.com/what-is-nft-explained/",
    "https://beebom.com/how-delete-spotify-account/",
    "https://beebom.com/how-download-gif-twitter/",
    "https://beebom.com/how-use-chatgpt-linux-terminal/",
    "https://beebom.com/how-delete-spotify-account/",
    "https://beebom.com/how-save-instagram-story-with-music/",
    "https://beebom.com/how-install-pip-windows/",
    "https://beebom.com/how-check-disk-usage-linux/"
]

# Load tài liệu bằng SeleniumURLLoader
loader = SeleniumURLLoader(urls=urls)
docs_not_splitted = loader.load()

# Chia nhỏ tài liệu
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(docs_not_splitted)

# Tạo embeddings bằng Ollama
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# DeepLake config
import os
from dotenv import load_dotenv
load_dotenv()
os.environ["ACTIVELOOP_TOKEN"] = os.getenv("ACTIVELOOP_TOKEN")
my_activeloop_org_id = "ncminhbka"
my_activeloop_dataset_name = "langchain_course_customer_support"
dataset_path = f"hub://{my_activeloop_org_id}/{my_activeloop_dataset_name}"

# Khởi tạo DeepLake vectorstore
db = DeepLake(dataset_path=dataset_path, embedding=embeddings)

# Add documents vào DB
#db.add_documents(docs)


# let's write a prompt for a customer support chatbot that
# answer questions using information extracted from our db
template = """You are an exceptional customer support chatbot that gently answer questions.

You know the following context information.

{chunks_formatted}

Answer to the following question from a customer. Use only information from the previous context information. Do not invent stuff.

Question: {query}

Answer:"""

prompt = PromptTemplate(
    input_variables=["chunks_formatted", "query"],
    template=template,
)

# user question
query = "How to check disk usage in linux?"

# retrieve relevant chunks
docs = db.similarity_search(query)
retrieved_chunks = [doc.page_content for doc in docs]

# format the prompt
chunks_formatted = "\n\n".join(retrieved_chunks)
prompt_formatted = prompt.format(chunks_formatted=chunks_formatted, query=query)

# generate answer
llm = OllamaLLM(model="gemma3:1b", temperature=0)
answer = llm(prompt_formatted)
print(answer)