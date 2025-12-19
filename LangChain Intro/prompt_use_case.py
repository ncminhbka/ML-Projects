from langchain_ollama import OllamaLLM
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate

llm = OllamaLLM(model='gemma3:1b', temperature=0)
template = "You are a helpful assistant that helps users find information about movies."
system_message_prompt = SystemMessagePromptTemplate.from_template(template)

human_template = "Find information about the movie: {movie_name}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
#response = llm.invoke(chat_prompt.format_prompt(movie_name="Inception").to_messages())

#print(response)


#summarize a PDF file
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import PyPDFLoader

summarize_chain = load_summarize_chain(llm)

document_loader = PyPDFLoader("The One Page Linux Manual.pdf")
#document = document_loader.load()

#summary = summarize_chain.invoke(document)
#print(summary['output_text'])

from langchain.prompts import PromptTemplate
prompt = PromptTemplate(template = "Question: {question}\nAnswer:", input_variables=["question"])

chain = prompt | llm
#response = chain.invoke({"question": "what is the current prime minister of UK?"})
#print(response)

#cần phân biệt chatmodel và LLM, llm sinh văn bản thuần túy, chatmodel sinh văn bản theo cấu trúc chat (có role) có khả năng duy trì ngữ cảnh
from langchain.schema import HumanMessage, SystemMessage, AIMessage
chat_model = OllamaLLM(model='gemma3:1b', temperature=0)
llm = OllamaLLM(model='llama3', temperature=0)
messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="What is the capital of France?"),
    AIMessage(content="The capital of France is Paris."),
    HumanMessage(content="I'd like to know about the city you just mentioned.")
]

response = llm.invoke(messages)
print(response)

