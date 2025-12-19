
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate

template = 'you are a helpful assistant that can translate {input_language} to {output_language}.'
system_message = SystemMessagePromptTemplate.from_template(template)

human_template = "{text}"
human_message = HumanMessagePromptTemplate.from_template(human_template)

chat_prompt = ChatPromptTemplate.from_messages([system_message, human_message])
chat_prompt = chat_prompt.format_prompt(input_language="English", output_language="French", text="I love programming.").to_messages()
print(chat_prompt)
from langchain_ollama import OllamaLLM

chat = OllamaLLM(model="llama3", temperature=0)
response = chat.invoke(chat_prompt)
print(response)