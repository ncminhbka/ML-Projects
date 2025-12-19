import json
import requests
from newspaper import Article

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.82 Safari/537.36'
}

article_url = "https://www.artificialintelligence-news.com/2022/01/25/meta-claims-new-ai-supercomputer-will-set-records/"

session = requests.Session()

try:
    response = session.get(article_url, headers=headers, timeout=10)
    
    if response.status_code == 200:
        article = Article(article_url)
        article.download()
        article.parse()
        
       # print(f"Title: {article.title}")
       # print(f"Text: {article.text}")
        
    else:
        print(f"Failed to fetch article at {article_url}")
except Exception as e:
    print(f"Error occurred while fetching article at {article_url}: {e}")


from langchain_core.messages import HumanMessage, SystemMessage

# we get the article data from the scraping part
article_title = article.title
article_text = article.text

# prepare template for prompt
template = """
As an advanced AI, you've been tasked to summarize online articles into bulleted points. Here are a few examples of how you've done this in the past:

Example 1:
Original Article: 'The Effects of Climate Change
Summary:
- Climate change is causing a rise in global temperatures.
- This leads to melting ice caps and rising sea levels.
- Resulting in more frequent and severe weather conditions.

Example 2:
Original Article: 'The Evolution of Artificial Intelligence
Summary:
- Artificial Intelligence (AI) has developed significantly over the past decade.
- AI is now used in multiple fields such as healthcare, finance, and transportation.
- The future of AI is promising but requires careful regulation.

Now, here's the article you need to summarize:

==================
Title: {article_title}

{article_text}
==================

Please provide a summarized version of the article in a bulleted list format.
"""

prompt = template.format(article_title=article.title, article_text=article.text)

messages = [HumanMessage(content=prompt)]

from langchain_ollama import OllamaLLM
chat = OllamaLLM(model='llama3', temperature=0)

#summary = chat.invoke(messages)
#print(summary)

from langchain.output_parsers import PydanticOutputParser

from pydantic import BaseModel, Field, field_validator
from typing import List

# create output parser class
class ArticleSummary(BaseModel):
    title: str = Field(description="Title of the article")
    summary: List[str] = Field(description="Bulleted list summary of the article")

    # validating whether the generated summary has at least three lines
    @field_validator('summary')
    def has_three_or_more_lines(cls, list_of_lines):
        if len(list_of_lines) < 3:
            raise ValueError("Generated summary has less than three bullet points!")
        return list_of_lines

# set up output parser
parser = PydanticOutputParser(pydantic_object=ArticleSummary)

from langchain_core.prompts.prompt import PromptTemplate
# create prompt template
# notice that we are specifying the "partial_variables" parameter
template = """
You are a very good assistant that summarizes online articles.

Here's the article you want to summarize.

==================
Title: {article_title}

{article_text}
==================

{format_instructions}
"""

prompt = PromptTemplate(
    template=template,
    input_variables=["article_title", "article_text"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

# Format the prompt using the article title and text obtained from scraping
formatted_prompt = prompt.format_prompt(article_title=article_title, article_text=article_text)

output = chat.invoke(formatted_prompt.to_messages())
parsed_output = parser.parse(output.split("\"]}")[0] + "\"]}")
print(parsed_output)