from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
#nhớ là llm đi cùng với prompttemplate, chatmodel đi cùng với chatprompttemplate

llm = OllamaLLM(model="gemma3:1b")
'''
template = '''
#As a futuristic robot band conductor, I need you to help me come up with a song title.
#What's a cool song title for a song about {theme} in the year {year}? 

'''

prompt = PromptTemplate(
    input_variables=["theme", "year"],
    template=template
)

input = {
    'theme': "interstella travel",
    'year': "3030"
}

chain = prompt | llm

response = chain.invoke(input)

print("Theme: interstellar travel")
print("Year: 3030")
print("AI-generated song title:", response)

'''
## FEWSHOT PROMPTING
from langchain.prompts import FewShotPromptTemplate
examples = [
    {"color": "red", "emotion": "passion"},
    {"color": "blue", "emotion": "serenity"},
    {"color": "green", "emotion": "tranquility"},
]

example_formatter_template = """
Color: {color}
Emotion: {emotion}\n
"""
example_prompt = PromptTemplate(
    input_variables=["color", "emotion"],
    template=example_formatter_template,
)

few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix="Here are some examples of colors and the emotions associated with them:\n\n",
    suffix="\n\nNow, given a new color, identify the emotion associated with it:\n\nColor: {input}\nEmotion:",
    input_variables=["input"],
    example_separator="\n",
)

formatted_prompt = few_shot_prompt.format(input="purple")

prompt=PromptTemplate(template=formatted_prompt, input_variables=[])
chain = prompt | llm

# Run the Runnable to get the AI-generated emotion associated with the input color
response = chain.invoke({})

print("Color: purple")
print(response)

#ta có thể thấy ở ví dụ 1 đã bị comment, câu trả lời khá miên man nhưng với fewshot câu trả lời đúng và đủ