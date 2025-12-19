from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
from pydantic import BaseModel, Field, field_validator
from typing import List
from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaLLM

# ------------------------
# 1. Định nghĩa schema
# ------------------------
class Suggestions(BaseModel):
    words: List[str] = Field(description="list of substitute words based on context")

    @field_validator("words")
    def not_start_with_number(cls, field):
        for item in field:
            if item[0].isnumeric():
                raise ValueError("The word can not start with numbers!")
        return field

# Parser gốc
base_parser = PydanticOutputParser(pydantic_object=Suggestions)

# Parser có khả năng sửa lỗi output
parser = OutputFixingParser.from_llm(
    parser=base_parser,
    llm=OllamaLLM(model="llama3", temperature=0.0)
)

# ------------------------
# 2. Prompt ép format JSON
# ------------------------
template = """
Offer a list of suggestions to substitute the specified target_word based on the presented context.
{format_instructions}

target_word={target_word}
context={context}
"""

prompt = PromptTemplate(
    template=template,
    input_variables=["target_word", "context"],
    partial_variables={"format_instructions": base_parser.get_format_instructions()}
)

model_input = prompt.format_prompt(
    target_word="behaviour",
    context="The behaviour of the students in the classroom was disruptive and made it difficult for the teacher to conduct the lesson."
)

# ------------------------
# 3. Gọi model
# ------------------------
llm = OllamaLLM(model="llama3", temperature=0.0)

output = llm.invoke(model_input.to_string())

# ------------------------
# 4. Parse và validate
# ------------------------
try:
    parsed = parser.parse(output)
    print("✅ AI-generated suggestions:", parsed.words)
except Exception as e:
    print("❌ Parse error:", e)
    print("Raw output from LLM:\n", output)
