from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
'''
# LLM interface
llm = OllamaLLM(model="gemma3:1b")

# Tạo PromptTemplate
prompt = PromptTemplate.from_template("Translate English to French: {text}")

# Format prompt
final_prompt = prompt.format(text="Hello, how are you?")

# Gọi model
response = llm.invoke(final_prompt)
print(response)
'''

from langchain.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
# Chat interface
chat = ChatOllama(model="gemma3:1b")

# Tạo ChatPromptTemplate
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "Translate English to French: {text}")
])

# Format messages
messages = chat_prompt.format_messages(text="Hello, how are you?")

# Gọi model
response = chat.invoke(messages)
print(response.content)

'''
Tôi đã hiểu, LLM nhận text sinh text, còn ChatModel nhận messages sinh messages. PromptTemplate sinh text, còn ChatPromptTemplate sinh messages.
phải dùng một cách tương ứng. nếu cố tình dùng ChatPromptTemplate với LLM, thì response chỉ là text, không có trường content.
hoặc cũng có thể gây lỗi vì ChatPromptTemplate không cho ra 1 string, cho nên không cho được vào LLM
Một điều nữa là OpenAI là LLM, còn ChatOpenAI là ChatModel. OllamaLLM là LLM, còn ChatOllama là ChatModel.
'''