
# 10分钟 Langchain 中文教程

### Introduction and overview

LangChain是什么？为什么要使用它？它是如何工作的？让我们来看一下。

![](https://pica.zhimg.com/80/v2-db727dddbb4c33e4b58d85a75e4fe8a7_1440w.png)

LangChain是一个开源框架，允许开发人员在与人工智能（AI）一起工作时将大型语言模型（如GPT4）与外部计算和数据源相结合。该框架目前提供Python和Js两种版本。

在这个教程中，我们将开始解析Python框架，并探讨为什么该框架的流行度正在迅速增长，特别是在2023年3月引入gpt4之后。
![](https://pic1.zhimg.com/80/v2-5820ba8ca2a204964bf1f42433f3fda7_1440w.png)

### Why Langchain?

为了理解LangChain填补了什么需求，让我们看一个实际的例子。

到目前为止，我们都知道ChatGPT（GPT4）拥有令人印象深刻的广泛知识，我们几乎可以询问任何问题，都能得到相当不错的答案。

假设您想从自己的数据、自己的文档中获取特定的信息，这可能是一本书、一个PDF文件、一个带有专有信息的数据库，LangChain允许您连接像GPT4这样的大型语言模型到您自己的数据源中。

这里我们不是指将文本文档的片段粘贴到ChatGPT的提示中，而是指引用一个完整的填满您自己数据的数据库。

而且，一旦获得所需的信息，您可以让LangChain帮助您执行想要的操作，例如发送带有特定信息的电子邮件。
![](https://picx.zhimg.com/80/v2-17010bf02b05e64738f218c7eeee796e_1440w.png)

您可以通过选择您希望您的语言模型引用的文档，将其切分成较小的块，并将这些块存储在矢量数据库中。这些块被存储为嵌入，即它们是文本的矢量表示。
![](https://pic1.zhimg.com/80/v2-a5f502cc551d998c38ec82291bfa5a67_1440w.png)

这使您能够构建遵循一般流程的语言模型应用程序：用户提出初始问题，然后将该问题发送到语言模型，使用该问题的矢量表示在矢量数据库中进行相似性搜索，从而允许我们从矢量数据库中获取相关的信息块，并将其提供给语言模型。

现在，语言模型既有初始问题，又有来自矢量数据库的相关信息，因此能够提供答案或执行操作。


### Langchain的价值地位 The value proposition of Langchain

LangChain有助于构建遵循这种流程的应用程序，这些应用程序既能够意识到数据（我们可以在矢量存储中引用自己的数据），又可以执行操作，而不仅仅是提供问题的答案。

![](https://pica.zhimg.com/80/v2-f4c43f7282de47906c59a02af10ffafa_1440w.png)

这两个能力,为无数的实际用例打开了大门，任何涉及个人辅助的事物都将是巨大的突破。

您可以让大型语言模型预订航班、转账、缴税，现在想象一下学习和掌握新事物的影响，您可以让大型语言模型引用整个教学大纲，并帮助您尽快学习材料。这将对编码、数据分析和数据科学产生影响。
![](https://picx.zhimg.com/80/v2-3301fdd1dccdd0e5ae39d976d5521da8_1440w.png)

我最兴奋的应用之一是将大型语言模型与现有公司数据（例如客户数据、市场营销数据等）相连接。

我认为我们将看到数据分析和数据科学取得指数级的进展，我们将能够将大型语言模型连接到Meta API或Google的API等高级API。
![](https://pic1.zhimg.com/80/v2-927856a2000a3068e622895bc1a03ab9_1440w.png)
### 拆解 Langchain Unpacking Langchain

因此，LangChain的主要价值主张可以分为三个主要概念：

* 我们有llm包装器，可以连接到像GPT4或hugging face的大型语言模型；
* 提示模板允许我们避免硬编码文本，这是llm的输入；
* 索引，可以从llm中提取相关信息；
* 链可以将多个组件组合在一起以解决特定任务，并构建整个llm应用；
* 代理，允许llm与外部API进行交互。

![](https://picx.zhimg.com/80/v2-b0ff795984a64cb5756da57c9f53d960_1440w.png)

LangChain有很多要解释的内容，每天都在添加新的内容，但从高层次上来看，这就是该框架的样子：我们有模型或模型的包装器；我们有问题；我们有链；我们有嵌入和矢量存储，即索引；然后我们有代理。

现在，我要开始逐个解释这些元素，通过编写代码来解析它们。

在这个教程中，我将保持框架层次的介绍，只是为了对该框架有一个概览并对不同的元素有所了解。

我们首先要做的是安装三个库：python（用于管理包含密码的环境文件）、LangChain和Pinecone客户端。

```
python-dotenv==1.0.0
langchain==0.0.137
pinecone-client==2.2.1

```

在环境文件中，我们需要OpenAI的API密钥、Pinecone环境和Pinecone API密钥。一旦您注册了Pinecone账户（免费），很容易找到API密钥和环境名称，对于OpenAI也是一样，只需访问：platform.orgmaili.com/account/API keys。

让我们开始吧！当您将密钥放入环境文件后，您只需获取密钥，然后我们就可以开始了。

```
from dotenv import load_dotenv

# Load the .env file (先要在当前目录下新建一个.env.local文件，内容写：OPENAI_API_TOKEN="YOU_OPENAI_API_TOKEN")
load_dotenv('.env.local')

# Get the value of the variable
openaikey_env = os.getenv('OPENAI_API_TOKEN')

os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_TOKEN')

```


### LLM Wrappers

我们将从llms（或llms的包装器）开始，然后我将导入OpenAI的包装器，并实例化text DaVinci 003完成模型并要求它解释大型语言模型是什么，这与直接调用open AI API时非常相似。

```
# Run basic query with OpenAI wrapper

from langchain.llms import OpenAI
llm = OpenAI(model_name="text-davinci-003")
llm("explain large language models in one sentence")
```

接下来，我们将转向聊天模型，即GPT 3.5和GPT4。

为了通过LangChain与聊天模型交互，我们将导入一个由三个部分组成的模式：一个AI消息`AIMessage`，一个人类消息`HumanMessage`和一个系统消息`SystemMessage`。然后，我们将导入ChatOpenAI。

```
# import schema for chat messages and ChatOpenAI in order to query chatmodels GPT-3.5-turbo or GPT-4

from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from langchain.chat_models import ChatOpenAI
```

系统消息是您在使用模型时用于配置系统的消息，而人类消息是用户消息。感谢您使用聊天模型，您可以将系统消息和人类消息组合成一个列表，然后将其用作聊天模型的输入。

这里我使用的是GPT 3.5 Turbo，您也可以使用gpt4，但由于open AI服务目前有些受限，所以我没有使用。这样就可以正常工作。

```
chat = ChatOpenAI(model_name="gpt-3.5-turbo",temperature=0.3)
messages = [
    SystemMessage(content="You are an expert data scientist"),
    HumanMessage(content="Write a Python script that trains a neural network on simulated data ")
]
response=chat(messages)

print(response.content,end='\n')
```

###  提示模板 Prompts and Prompt Templates
让我们继续下一个概念，即提示模板。

提示是我们将发送给语言模型的内容，但大多数情况下，这些提示都不是静态的，它们将在应用程序中使用，并且为此，LangChain有一个叫做提示模板 *`PromptTemplate`*。

这使我们能够将用户输入插入到文本中，并且可以使用用户输入格式化提示，然后将其提供给语言模型。

这是最基本的示例，但它使我们能够动态更改带有用户输入的提示。

```
# Import prompt and define PromptTemplate

from langchain import PromptTemplate

template = """
You are an expert data scientist with an expertise in building deep learning models. 
Explain the concept of {concept} in a couple of lines
"""

prompt = PromptTemplate(
    input_variables=["concept"],
    template=template,
)
```

```
# Run LLM with PromptTemplate

llm(prompt.format(concept="autoencoder"))
```

### 链 Chains

链将语言模型和提示模板结合到一个接口中，该接口接受用户的输入并从语言模型中输出答案，类似于一个复合函数，其中内部函数是提示模板，外部函数是语言模型。

我们还可以构建顺序链，其中一个链返回一个输出，然后第二个链将第一个链的输出作为输入。

所以这里我们有第一个链，它接受一个机器学习概念并给出一个简要的解释。

```
# Import LLMChain and define chain with language model and prompt as arguments.

from langchain.chains import LLMChain
chain = LLMChain(llm=llm, prompt=prompt)

# Run the chain only specifying the input variable.
print(chain.run("autoencoder"))
```

然后第二个链接受第一个概念的描述，并用我五岁时的语言解释给我听。

```
# Define a second prompt 

second_prompt = PromptTemplate(
    input_variables=["ml_concept"],
    template="Turn the concept description of {ml_concept} and explain it to me like I'm five in 500 words",
)
chain_two = LLMChain(llm=llm, prompt=second_prompt)
```

然后我们简单地将两个链，即第一个名为chain的链和第二个名为chain two的链，组合成一个总的链，并运行该链，我们看到总的链返回了概念的第一个描述和用五岁时的语言解释的概念。

```
# Define a sequential chain using the two chains above: the second chain takes the output of the first chain as input

from langchain.chains import SimpleSequentialChain
overall_chain = SimpleSequentialChain(chains=[chain, chain_two], verbose=True)

# Run the chain specifying only the input variable for the first chain.
explanation = overall_chain.run("autoencoder")
print(explanation)
```

### 嵌入和向量存储 Embeddings and VectorStores

让我们继续到嵌入和向量存储，但在此之前，让我将解释器切换为“用五岁的语言解释给我”的提示，这样我们将得到更多的词。我将选择500个词。好的，这是一个稍长一点的五岁儿童解释。

现在我要检查这个文本，将其分成块，因为我们想将其存储在Pinecone中的向量存储中，而LangChain有一个名为文本切割器 `langchain.text_splitter` 的工具可以做到这一点。所以我要导入递归字符文本切割器，然后我要根据我们在教程开始时讨论的方式将文本分成块。

```
# Import utility for splitting up texts and split up the explanation given above into document chunks

from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 100,
    chunk_overlap  = 0,
)

texts = text_splitter.create_documents([explanation])
```
我们可以使用page_content提取列表中各个元素的纯文本。
```
# Individual text chunks can be accessed with "page_content"

texts[0].page_content
```

现在，我们要将其转换为嵌入，嵌入只是这个文本的向量表示，我们可以使用OpenAI的嵌入模型Ada来进行嵌入。

```
# Import and instantiate OpenAI embeddings

from langchain.embeddings import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model_name="ada")
```

我们可以在从文档块中提取的原始文本上调用嵌入查询，然后我们就可以得到该文本的向量表示或嵌入。

```
# Turn the first text chunk into a vector with the embedding

query_result = embeddings.embed_query(texts[0].page_content)
print(query_result)

```


然后，我们要检查解释文档的块，并将向量表示存储在Pinecone中。所以我们将导入Pinecone的Python客户端，并从LangChain Vector存储中导入Pinecone。

我们用环境文件中的密钥和环境初始化Pinecone客户端

```
# Import and initialize Pinecone client

import os
import pinecone
from langchain.vectorstores import Pinecone


pinecone.init(
    api_key=os.getenv('PINECONE_API_KEY'),  
    environment=os.getenv('PINECONE_ENV')  
)
```

然后我们取名为 `index_name` 的变量，这个变量包含我们要存储的所有数据块，我们取出嵌入模型和索引名称，并将这些块加载到Pinecone的嵌入中。

```
# Upload vectors to Pinecone

index_name = "langchain-quickstart"
search = Pinecone.from_documents(texts, embeddings, index_name=index_name)
```


一旦我们在Pinecone中存储了向量，我们可以提出关于存储数据的问题，例如一个自动编码器有什么神奇之处，然后我们可以在Pinecone中进行相似性搜索，以获得答案或提取所有相关的块。

```
# Do a simple vector similarity search

query = "What is magical about an autoencoder?"
result = search.similarity_search(query)

print(result)
```

如果我们转到Pinecone，我们可以看到索引在这里，我们可以点击它并检查它，检查索引信息，我们在向量存储中一共有13个向量。

### 代理 Langchain Agent

现在，如果您转到open AI chat GPT插件页面，您会看到他们展示了一个Python代码解释器。实际上，我们可以在Langchain中做类似的事情。在这里，我导入了create_python_agent，以及来自Langchain的PythonREPLTool工具和create_python_agent。

然后，我们使用OpenAI语言模型实例化一个python代理执行器，这样我们就可以让语言模型运行Python代码。

```

# Import Python REPL tool and instantiate Python agent

from langchain.agents.agent_toolkits import create_python_agent
from langchain.tools.python.tool import PythonREPLTool
from langchain.python import PythonREPL
from langchain.llms.openai import OpenAI

agent_executor = create_python_agent(
    llm=OpenAI(temperature=0, max_tokens=1000),
    tool=PythonREPLTool(),
    verbose=True
)
```
在这里，我想找到一个二次函数的根，我们看到代理执行器正在使用numpy roots来找到这个二次函数的根。
```
# Execute the Python agent

agent_executor.run("Find the roots (zeros) if the quadratic function 3 * x**2 + 2*x -1")
```
好的，这个教程旨在向您简要介绍langchain的核心概念。
