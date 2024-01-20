from langchain.chat_models import ChatOpenAI
from langchain.schema import (
AIMessage,
HumanMessage ,
SystemMessage)

chat = ChatOpenAI(openai_api_key="你的密钥",temperature=0)
result = chat([
HumanMessage(
content="请把下面的话翻译成英语 :敬个礼，握握手你是我的好朋友。")])
print(result)