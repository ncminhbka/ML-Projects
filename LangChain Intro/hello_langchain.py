from langchain_community.llms import Ollama
#testing ollama

llm = Ollama(model="gemma3:1b", temperature=0.5)

#resp = llm.invoke("chào bạn")
#print(resp)

#CHAINS
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

prompt = PromptTemplate(
    input_variables=["product"], #input_variables=["product", "audience"],
    template = "What is a good name for a company that makes {product}?" #template="Suggest a creative name for a company that makes {product}, targeting {audience}.",
)

chain = LLMChain(llm=llm, prompt=prompt) #Nghĩa là bạn đã đóng gói một workflow: điền giá trị → gửi model → nhận output.
print(chain.run("eco-friendly water bottles")) #print(chain.run({"product": "eco-friendly water bottles", "audience": "college students"}))

#MEMORY
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
#Khác với LLMChain chỉ chạy một lần với prompt template, thì ConversationChain giữ ngữ cảnh giữa nhiều lần gọi.
conversation = ConversationChain(
    llm=llm,
    verbose=True, # in ra chi tiết các bước thực hiện
    memory=ConversationBufferMemory() #Mỗi khi bạn gọi .predict(...), LangChain sẽ nối các câu trước đó vào prompt, để model nhớ được bối cảnh.
)

# Start the conversation
conversation.predict(input="Tell me about yourself.")

# Continue the conversation
conversation.predict(input="What can you do?")
conversation.predict(input="How can you help me with data analysis?")

# Display the conversation
print(conversation)







