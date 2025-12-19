from langchain.embeddings import OllamaEmbeddings
import faiss
from langchain.vectorstores import FAISS
from langchain.docstore import InMemoryDocstore

# Định nghĩa mô hình embedding
embeddings_model = OpenAIEmbeddings(model="text-embedding-ada-002")

# Khởi tạo vectorstore
embedding_size = 1536
index = faiss.IndexFlatL2(embedding_size)
vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})


from langchain_experimental.autonomous_agents import BabyAGI
from langchain_ollama import OllamaLLM
# Đặt mục tiêu
goal = "Plan a trip to the Grand Canyon"

# Tạo tác nhân BabyAGI
baby_agi = BabyAGI.from_llm(
    llm=OllamaLLM(model="gemma3:1b", temperature=0),
    vectorstore=vectorstore,
    verbose=False,
    max_iterations=3
)
response = baby_agi({"objective": goal})