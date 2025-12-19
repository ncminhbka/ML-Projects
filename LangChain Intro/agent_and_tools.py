import os
from dotenv import load_dotenv
# Load biến môi trường từ file .env
load_dotenv()
# Lấy key
google_api_key = os.getenv("GOOGLE_API_KEY")
google_cse_id = os.getenv("GOOGLE_CSE_ID")


from langchain.agents import AgentType

from langchain.agents import initialize_agent

from langchain.agents import Tool
from langchain.utilities import GoogleSearchAPIWrapper

from langchain_ollama import OllamaLLM
llm = OllamaLLM(
    model="gemma3:1b",
    temperature=0,
)


search = GoogleSearchAPIWrapper()

#define tools
tools = [
    Tool(
        name = "google-search",
        func=search.run,
        description="useful for when you need to search google to answer questions about current events",
        
    )
]

#create the agent
agent = initialize_agent(tools, 
                         llm, 
                         agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
                         verbose=True,
                         max_iterations=6,
                         handle_parsing_errors=True)
response = agent("What's the latest news about the Mars rover?")
print(response['output'])

#TOOLS

from langchain.agents import Tool
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.agents import initialize_agent, AgentType


prompt = PromptTemplate(
    input_variables=["query"],
    template="Write a summary of the following text: {query}"
)

summarize_chain = LLMChain(llm=llm, prompt=prompt)

search = GoogleSearchAPIWrapper()

tools = [
    Tool(
        name="Search",
        func=search.run,
        description="useful for finding information about recent events"
    ),
    Tool(
       name='Summarizer',
       func=summarize_chain.run,
       description='useful for summarizing texts'
    )
]