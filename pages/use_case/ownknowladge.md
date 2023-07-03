# LangChain 教程 - 使用 LangChain，将 GPT 连接到自己的知识库

> 向您展示如何将自己的数据集成到 ChatGPT 中。本教程需要一个 OpenAI API 密钥。如果没有，您可以像那样按照教程进行操作。
如何学习提示，索引，记忆和链，嵌入等概念 - 理论和实践。


嗨，在这个文章中，我将向您展示如何使用 LangChain 来增强像 gpt4 这样的大型语言模型。例如，使用 LangChain，您可以将 GPT 连接到自己的知识库中，而这些知识库并未经过模型训练。

## 理论可行性

首先，我将解释为什么要这样做以及如何在理论上实现。然后，我们将通过实际的代码来更容易地理解。

让我们以一个虚构的例子为例。假设我们有 100 个文本的知识库，我们希望将其提供给其他人，并通过 gpt4 生成类似人类的自然语言信息，而不仅仅是从这些文本文件中查询事实。

然而，问题是，您给机器人的提示(Prompts)或消息只能有一定数量的单词或标记，因此您必须预先过滤大型数据集。这通常使用语义过滤来完成，而向量数据库就派上了用场。它们会检查问题与数据库中的数据的相似性，然后将相似的数据作为输出返回。然后，将这个减少的输出再次作为提示(Prompts)发送。

现在，让我们从开发者的角度来看。

作为开发者，我们最初只有这些文本文件，但当然我们也可以有其他的知识来源，比如 PDF 或 CSV 文件。对于文本文件，提取数据相对容易。但对于 PDF 文件，情况就比较困难了。一旦我们提取了数据，我们不能直接将其馈送到数据库中，而是需要将其拆分成小块，即所谓的“块”。例如，如果我们有 10,000 个字，我们将它们分成 100 个字的一组，最后得到 100 个块。

现在我们有了很多块，这些块必须转换为所谓的“嵌入（embedding）”。

### 什么是嵌入（embedding）？

嵌入（embedding）模型会将输入文本转换为一个向量，您可以看到这个向量只是一组数字，或者说是一个数字数组或数字列表。
这些向量存储在数据库中，您可以在这里看到这些向量具有意义，例如，与动物相关的任何内容都存储在一个向量空间中，与运动员相关的任何内容都存储在另一个空间中。

然后，您发出一个查询，例如“什么是大象”，大象也会被转换为一个向量，然后从数据库中检索类似的向量。这就是语义过滤。

现在，假设您从数据库中获得了前四个向量，并将这些向量转换回文本。然后，您提出一个问题，并将从向量数据库中检索到的文本作为提示(Prompts)输入到 GPT 中进行检查。

好了，这就是理论部分。

现在我们将进入代码部分，我将为您介绍代码，您可以了解如何创建一个向量数据库以及如何在其中运行查询。


## 代码实现

### 安装包

首先，我要安装 LangChain，这是一个包，可以让我们很容易地使用向量数据库以及 openAI 来运行查询。
这是非常好的和轻量级的包，我将在这里展示基本用法。您还需要安装 openAI 并拥有一个 openAI 的 API 密钥，这是唯一的先决条件，否则将无法工作，因为我使用 openAI 作为我的嵌入（embedding）模型。我还需要安装 pickle，这是用于反序列化和序列化我们的向量数据库的。我们将使用由 Facebook 构建的文件。我们还将安装 python-dotenv，因为我将我的 API 密钥存储在这个.env 文件中，并且我不希望向所有人展示。在初始化或安装了这些包之后，我会加载我的 API 密钥到变量 API key 中。

```bash
!pip install langchain
!pip install openAI
!pip install pickle
!pip install python-dotenv
```

### 加载 LangChain 的 `document_loaders`

好的，在加载了 API 密钥之后，我们现在可以看一下 LangChain 的第一个概念，那就是 `document_loaders`。
`document_loaders` 是一个类，允许您使用数据库或文本来源，并将其转换或放入数据库中。

```python
from langchain.document_loaders import DirectoryLoader, TextLoader

loader = DirectoryLoader('./FAQ', glob="**/*.txt", loader_cls=TextLoader, show_progress=True)
docs = loader.load()
```
有很多不同的 loader，比如 `TextLoader`、`PDFLoader` 以及这里的 `DirectoryLoader`，这里数据中包含了一些关于动物的有效信息。
有兔子和狗，这样做是为了确保我对数据库发出的查询不是由 GPT 生成的，而是从我的数据库中检索到的。

因此，要加载数据，首先我们必须导入，`import DirectoryLoader, TextLoader`。
我使用 `DirectoryLoader`，因为 FAQ 文本文件中的所有内容都是文本，我可以直接使用此路径，然后使用 `DirectoryLoader` 和 `TextLoader` 来加载它，并显示我的进度，然后将其加载到内存中。

### 原始数据切块

加载数据到内存中后，我们现在必须将其拆分成所谓的块。
LangChain 有非常好的文本拆分器，用于递归字符文本拆分。

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
)

documents = text_splitter.split_documents(docs)
# documents[0]
```
这里我将指定一个块大小为 500 个单词或标记，并且我还需要一个 100 个单词或标记的重叠。因为我不想丢失块的上下文。

在初始化或实例化了这个 `RecursiveCharacterTextSplitter` 类之后，我现在将拆分存储在这个 `docs` 变量中的文档(docs 变量保存了全部 txt 文件的数据）。

当运行这段代码时，您现在可以看到从文本文件中提取出来的内容，并且其块大小为 500。

### 向量存储

好了，现在我们加载了我们的块，我们现在需要使用我们的嵌入（embedding）。从 LangChain 中，您可以使用不同的嵌入（embedding）类，我将使用 openAI 的嵌入（embedding）。

```
from langchain.embeddings import openAIEmbeddings
embeddings = openAIEmbeddings(openAI_api_key=API_KEY)
```
我将在这里导入它，并使用 openAI 的 API 密钥实例化它，这个值是我从我的.env 文件中加载的 API 密钥。所以我有了一个实例化的类，然后我们需要将块加载到一个向量数据库中。

在 LangChain 中，有不同类型的向量数据库可以使用，我将在这里使用 FAISS，并导入这个文件类。

```
from langchain.vectorstores.faiss import FAISS
import pickle

vectorstore = FAISS.from_documents(documents, embeddings)

with open("vectorstore.pkl", "wb") as f:
    pickle.dump(vectorstore, f)
```
现在，从 FAISS 中，我有一个 `from_documents` 的类方法，我将使用这些 `documents` 和嵌入（embedding）实例 `embeddings`，这将创建一个向量存储，这个向量存储可以被 pickled 或 dumped，现在我在文件系统上有了一个向量存储。

现在，我有了一个向量存储，可以在其上运行查询和操作。现在我们只需要再次加载我们的向量存储，并将其放入内存中。

```
with open("vectorstore.pkl", "rb") as f:
    vectorstore = pickle.load(f)
```

### 输入提示(Prompts)

好了，现在我们可以向 GPT 发送一个问题，但您不能只是提出一个问题，您需要提供更多的内容，这个内容实际上是上下文，所以在问题中的上下文被称为所谓的提示(Prompts)。

您给机器人或 GPT 模型一个身份，比如说您是一名兽医，您帮助用户处理他们的动物问题，然后您在这里放置您的回答，所有这些放在一起被称为所谓的提示(Prompts)。

您可以从 LangChain 获得一个不错的提示(Prompts)模板，也就是代码中的 `PromptTemplate`, 并使用上下文变量和问题进行实例化。

```python
from langchain.prompts import PromptTemplate

prompt_template = """
你是一位兽医，帮助用户处理他们的宠物.

{context}

Question: {question}
Answer is:
"""
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
```

这是整个模板，这部分是固定的，这部分是变量，您将其放在这里，现在我们有了提示(Prompts)模板。
所以这个提示(Prompts)模板现在可以在所谓的“链(Chains)”中使用，您可以为不同的用例获得不同的链(Chains)，比如聊天或检索 QA。

### 链(Chains)

检索式问答链 `RetrievalQA` 将被用于我们的查询和数据库，然后将该输出放入模型中，这在检索式问答链 `RetrievalQA` 中得到了很好的实现。

```python
from langchain.llms import openAI
from langchain.chains import RetrievalQA

chain_type_kwargs = {"prompt": PROMPT}

llm = openAI(openAI_api_key=API_KEY)
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever(), chain_type_kwargs=chain_type_kwargs)

query = "一只鸡可以活多久？"
qa.run(query)
```
我们首先需要初始化或实例化我们的模型，这里我将使用 openAI。在这里放入 openAI API 密钥，现在我们有了我们的 LLM（大型语言模型），将其作为变量放入链(Chains)中。

现在您还需要在这里放入一个检索器(retriever)，将 **向量存储转换为检索器** `retriever=vectorstore.as_retriever()`。

如果您现在运行整个链(Chains)，您将得到将要使用的 LLM。

### 内存（Memory）


您还将在这里使用数据存储，因此我将运行 aquarium，了解动物的寿命，GPT 将为我回答。


首先，我当然必须运行模板，然后运行链(Chains)。

但是现在我们有一个小问题，或者至少有时候我们有一个小问题，那就是如果我们运行多个查询，我们总是必须在这里给出正确的问题，但是整个对话的上下文丢失了。这就是记忆非常有用的地方，记忆存储了整个对话并将其用于 Prompts 链(Chains)中的输入。

Langchain 提供了多个记忆类，例如对话缓冲区记忆 `ConversationBufferMemory`，您可以给它一个记忆键 `memory_key='chat_history'`，如聊天记录和一个输出键 `output_key`。您还可以指定您希望从记忆中返回的消息。

```python
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer')
```

上面创建这个对话缓冲区记忆的实例，并且我们使用不同的通道，因为检索式问答链(Chains)无法使用记忆，但是对话检索链 `ConversationalRetrievalChain` 可以使用它，所以我们做同样的事情，使用 `text-davinci-003` 模型来实例化它，这是一个非常适合文本生成的模型，并给温度调节为 0.7，表示模型的动态程度以及记忆作为输入参数。

```python
from langchain.chains import ConversationalRetrievalChain

qa = ConversationalRetrievalChain.from_llm(
    llm=OpenAI(model_name="text-davinci-003", temperature=0.7, openai_api_key=API_KEY),
    memory=memory,
    retriever=vectorstore.as_retriever(),
    combine_docs_chain_kwargs={'prompt': PROMPT}
)


query = "鸡能活多久?"
qa({"question": query})
qa({"question": "鸡每天吃多少?"})
```

如果您运行多个查询，例如这里的“动物寿命有多长”和“它吃多少”，在这种情况下，它从记忆中检索到或从记忆中解释出来，否则 GPT 将不知道它实际上是什么意思，因此它必须有记忆。

因此，如果我们运行这个查询，您可以看到它是狗或兔子的混合物，只需要吃一点食物，因此它被正确解释。

好了，就是这样，正如您所看到的，Langchain 非常易于使用且功能强大。

如果您喜欢我的文章，请不要忘记收藏和点赞，非常感谢，再见。

## 参考资料

GitHub 仓库：https://github.com/Coding-Crashkurse/LangChain-Basics/blob/main/basics.ipynb

## 学习交流
