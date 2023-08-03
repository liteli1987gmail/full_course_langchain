from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
openai_api_key=""

llm = OpenAI(openai_api_key=openai_api_key)
prompt = PromptTemplate(input_variables=["product"],template="给这个{product}，取名字")
chain = LLMChain(llm=llm,prompt=prompt)
answer = chain.run("袜子")
print(answer)