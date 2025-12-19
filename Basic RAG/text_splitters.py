from langchain_community.document_loaders import PyPDFLoader
loader = PyPDFLoader("The One Page Linux Manual.pdf")
pages = loader.load_and_split()

from langchain.text_splitter import CharacterTextSplitter

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
texts = text_splitter.split_documents(pages)

print (f"You have {len(texts)} documents")

from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=50,
    chunk_overlap=10,
    length_function=len,
)

docs = text_splitter.split_documents(pages)
print (f"You have {len(docs)} documents")

from langchain.text_splitter import TokenTextSplitter



# Initialize the TokenTextSplitter with desired chunk size and overlap
text_splitter = TokenTextSplitter(chunk_size=100, chunk_overlap=50)

# Split into smaller chunks
texts = text_splitter.split_text(pages)
print(texts[0])