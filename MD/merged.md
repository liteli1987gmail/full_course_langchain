

# 1.1.1 为什么需要 Langchain
# 1.1.2 环境和密钥配置
# 1.3.1 4 行代码开始
# 1.3.2 核心模块
# 1.3.3 解决大模型先天缺陷
# 1.4.1 Langchain 开发流程
# 1.4.2 创建你的第一个聊天机器人
# 1.4.3 减少 Prompt 输入
# 1.4.4 开始对话


# 模型输入输出

LangChain 打通了大语言模型应用开发的“最后一公里”。

任何大语言模型应用的核心元素是 —— 模型的输入 Input 和输出 Output (简称模型输入输出 (Model I/O),下面都用 I 表示模型的输入 Input，O 表示模型的输出 Output）。

最近一年左右，通过强化学习等方法对模型进行微调（Fine-tuning），以及在提示（Prompt）中使用各种方法，以从这些模型中获得最佳效果。一旦我们能够做到这一点，从这些大型语言模型中我们可以得到一些非常好的结果。问题是，它们仍然无法以接口方式访问现有数据和API,无法与真实的世界连接起来就很难应用,尤其是复杂的应用。

如果我们想要使用这些模型构建一个应用程序，我们需要某种方式在大型语言模型和传统软件之间进行接口交互。同时，越来越多的大语言模型涌现出来，模型输入输出变得异常重要。LangChain 为我们提供了与任何大型语言模型进行交互的基本组件。

LangChain 不仅可以方便地与最新、最强大的模型如GPT-4进行交互，还可以与本地私有化部署的语言模型，或是在 HuggingFace 上找到的开源模型进行交互。

只需几行代码，就可以实现与这些模型进行对话。如何使用 Langchain 的基础组件模型输入输出来访问大型语言模型？来看看下面这张示意图：

![ model_io 图示](/img/model_io.jpg)

Model I/O 组件提供了三个核心功能：

- [语言模型 models ](/docs/modules/model_io/models/): 通过接口调用语言模型，即 Model。
- [提示 prompts ](/docs/modules/model_io/prompts/): 将模型输入模板化、动态选择和管理，即 Model I。
- [输出解析器 output_parsers ](/docs/modules/model_io/output_parsers/): 从模型输出中提取信息 ，即 Model O。


我们将深入了解如何使用 Langchain 的模型输入输出组件来访问大语言模型。我们先认识三大模型类别。


####   语言模型的分类

LangChain 模型输入输出组件的目前有三大模型类型。分别是通用大型语言模型 LLMs， 专用聊天模型 Chat Models 和文本嵌入模型 Text Embedding Models。

在 LangChain 中使用的不同类型的完成大语言模型输入和输出。在这一章中，我们将对模型类型进行分类和认识。

####  通用大型语言模型

首先介绍的是大型语言模型（ LLMs ）。这些模型以文本字符串作为输入，并返回文本字符串作为输出。

####  专用聊天模型

第二种是专用聊天模型。这些模型通常由语言模型支持，但其 API 更加结构化。具体来说，这些模型以聊天消息列表作为输入，并返回聊天消息。

####  文本嵌入模型

文本嵌入模型是以文本作为输入，并返回浮点数列表，也就是向量维度表示的列表。

这三种类型中，通用大型语言模型和专用聊天模型很容易被人误解没有必要。实际上，通用大型语言模型和专用聊天模型有微妙但重要的区别。

####   通用大型语言模型和专用聊天模型的区别

LangChain 中的 通用大型语言模型是指纯文本补全模型， 也就是Text To Text。
它们包装的 API 接受字符串提示作为输入，并输出字符串补全部分。OpenAI 的 GPT-3 就是 LLM 的实现典型。

专用聊天模型通常由 LLMs 支持，但专门用于进行对话。

专用聊天模型的不同之处在于：提供以 "聊天消息" 作为输入和输出的接口。它们的输入不是单个字符串，而是聊天消息的列表。

通常，这些消息带有发言者身份（LangChain 目前支持的消息类型有“AIMessage”，“HumanMessage”，“SystemMessage”）。它们返回一个（"AI"）聊天消息作为输出。GPT-4 和 Anthropic 的 Claude 都是作为专用聊天模型实现的。


####   通用大型语言模型和专用聊天模型的的学习路径


在Langchain的发展迭代过程中，每个模块都精细地划分出通用大型语言模型（LLM）和专用聊天模型两种类型，紧跟OpenAI的前沿技术潮流，以更好地适配新的专用聊天模型。

这一区分已经形成了一种趋势，也为我们学习Langchain提供了线索。每当我们看到一个针对通用模型的Langchain类或方法，我们便能预期到对应的聊天模型类或方法也必然存在。

然而，Langchain这样的做法有时候会让人感到困惑，因为它每次都会单独声明通用大型语言模型和专用聊天模型的不同代码。但是，如果你从模型的输入输出开始理解，下面的学习过程会变得轻松很多。例如，如果你刚刚学到通用模型，你就预期会有一个聊天模型，你就能更好地做出预测。


####   模型输出的通用接口

无论是通用大模型还是专用聊天模型，Langchain 都暴露了predict方法，而且包装进了对话的链或者代理上，使用非常方便。




# 2.1.1 通用大型语言模型（LLMs）

大型语言模型（LLMs）是 LangChain 的核心组件。LangChain 不提供自己的 LLMs，而是提供了与许多不同的 LLMs 进行交互的标准接口。

我们开始逐个解释这些元素，通过编写代码来解析它们。

有许多 LLM 提供商（OpenAI、Cohere、HuggingFace 等）—— `langchain.llms` 类旨在为所有提供商提供标准接口。

使用 OpenAI LLM 包装器，这些方法对于所有 LLM 类型都是通用的。以下是示例代码：

####  安装和设置密钥

首先，我们需要安装 OpenAI Python 包：

```bash
pip install openai langchain
```

然后设置好密钥：

```python
from langchain.llms import OpenAI
OpenAI.openai_api_key = "YOUR_OPENAI_API_TOKEN"
```


####  使用 LLM 的最简单方法: 字符串 in -> 字符串 out


```python
# Run basic query with OpenAI wrapper
llm = OpenAI()
llm("Tell me a joke")
```
运行后的结果：

```
    'Why did the chicken cross the road?\n\nTo get to the other side.'
```

说明： 这里的运行结果是随机的，而不是唯一固定的回答。

接下来我们转入第二种专用聊天模型。

# 2.1.2 专用聊天模型

当前最大的应用场景便是 “聊天 Chat” ，就像OpenAI的热门应用ChatGPT一样。为了紧跟用户需求，LangChain推出了专门专用聊天模型Chat Models，以便我们能与各种聊天模型进行无缝交互。

在上一节我们提到专用聊天模型的不同之处在于：提供以 "聊天消息" 作为输入和输出的接口。它们的输入不是单个字符串，而是聊天消息的列表。

这一节，我们通过代码，看看输入聊天消息列表的专用聊天模型, 究竟与通用 LLMs 有什么区别？


以下是示例代码：

####  安装和设置密钥

首先，我们需要安装 OpenAI Python 包：

```bash
pip install openai langchain
```

然后设置好密钥：

```python
import os

os.environ['OPENAI_API_KEY'] = ''
```

####  使用 Chat Models 的最简单方法

为了通过LangChain与聊天模型交互，我们将导入一个由三个部分组成的模式：一个AI消息 `AIMessage`，一个人类消息`HumanMessage`和一个系统消息`SystemMessage`。然后，我们将导入`ChatOpenAI`。


```python
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from langchain.chat_models import ChatOpenAI
```

`SystemMessage` 是您在使用模型时用于配置系统的消息，而 `HumanMessage` 是用户消息。

我们现在将系统消息和人类消息组合成一个聊天信息列表，然后输入到 Chat Models 聊天模型。

这里我使用的是模型名称是：gpt-3.5-turbo。如果你有gpt4，也可以使用gpt4。

```python
chat = ChatOpenAI(model_name="gpt-3.5-turbo",temperature=0.3)
messages = [
    SystemMessage(content="你是个取名大师，你擅长为创业公司取名字。"),
    HumanMessage(content="帮我给新公司取个名字，要包含AI的")
]
response=chat(messages)

print(response.content,end='\n')
```

运行后获得 `AIMessage` 的结果是：

```
当然！以下是一些包含"AI"的创业公司名称的建议：

1. AIgenius
2. AItech
3. AIvision
4. AIpros
5. AIlink
6. AIsense
7. AIsolutions
8. AIwave
9. AInova
10. AIboost

希望这些名称能够给你一些启发！如果你有其他要求或者想要更多的建议，请随时告诉我。

```

本节中，我们着重研究了如何运用专用聊天模型。在LangChain的持续迭代过程中，我们目睹了模型和LangChain都在不断进化。

# 2.1.3 文本嵌入模型

想要造航母，仅仅一个大语言模型是不够的。

假设我们想要创造一个《红楼梦》聊天机器人应用，并且询问关于这本书的知识。一个大语言模型就够了吗？

还有我们上传专业的论文，长达几十页的临床医学实验报告，或者使用搜索或外部数据的东西，跟真实的数据连接起来，大语言模型能处理吗？

虽然最新的模型一直在突破极限，但是仍然不能满足我们想要大模型阅读大型文档、书籍的需求。

面对这些挑战，我们可能需要总结或寻找其他方法来应对。因为大型语言模型可以传递的标记数量是有限的，大多数模型最多可以传递1024到16K个标记，尽管一些新的模型可以处理更多的标记。这就意味着，我们需要思考如何有效地利用这些有限的标记，以达到我们的应用需求。

此外，理解模型的类型和其各自的特点也同样重要。模型在高层次上可以分为两种不同类型：语言模型和文本嵌入模型。嵌入模型将文本转换为向量标记，让我们可以将文本标记为向量特征，显示在向量空间中。了解这两种模型的特性和差异，能帮助我们更好地选择和使用模型，进一步优化我们的应用。

####   嵌入模型将文本转换为向量标记的例子

在自然语言处理（NLP）中，嵌入模型是将文本转换为向量的重要工具，这样我们可以在向量空间中表示文本，以便进行后续的机器学习或深度学习任务。举个例子，我们可以用 Word2Vec，一个常见的嵌入模型，来演示这个过程。

假设我们有以下的语料库：

```
1. I love learning.
2. I like reading books.
3. Books are great for learning.
```

首先，我们需要建立一个词汇库，也就是列出所有在语料库中出现过的独立单词：

```
"I", "love", "learning", "like", "reading", "books", "are", "great", "for"
```

接下来，我们用 Word2Vec 来训练这个语料库。Word2Vec 是一个预测模型：它试图从上下文预测目标单词，或者从目标单词预测上下文。训练过程中，Word2Vec 会学习到一个单词和它的上下文单词之间的关系，并将这些信息编码到向量中。

训练完成后，每个单词都会被赋予一个向量。例如，“books”的向量可能是：

```
[0.1, -0.2, 0.3, 0.5, -0.1]
```

而“learning”的向量可能是：

```
[-0.1, 0.3, -0.2, 0.1, 0.4]
```

这些向量现在就代表了对应的单词在向量空间中的位置，且具有一些有趣的属性。比如，语义上相似的单词在向量空间中的位置会比较接近，我们可以通过计算两个向量之间的余弦相似度来衡量这种接近程度。

以上就是嵌入模型将文本转换为向量标记的一个基本例子。

####   嵌入模型的原理


![](https://pic3.zhimg.com/80/v2-df6b821706891153068f5ccc4fab7afa_1440w.jpeg)

在上面这个图像中，我们可以看到在一个二维空间中，“man”是“king”，“woman”是“queen”，它们代表不同的事物，但我们可以看到一种模式。这个模式就是可以在向量空间中寻找最相似的文本片段，实现语义搜索。

例如，OpenAI 的文本嵌入模型可以精确地嵌入大段文本，具体而言，8100 个标记，根据它们的词对标记比例 0.75，大约可以处理 6143 个单词。它输出 1536 维的向量。

我们可以使用 LangChain 与多个嵌入提供者进行接口交互，例如 OpenAI 和 Cohere 的 API，但我们也可以通过使用 Hugging Faces 的开源嵌入在本地运行，以达到 免费和数据隐私 的目的。

现在，您可以使用仅 4 行代码在自己的计算机上创建自己的嵌入。但是，维度数量可能会有所不同，嵌入的质量可能会较低，这可能会导致检索不太准确。

####   快速入门

####  安装和设置密钥

我们需要安装 Langchain Python 包：

```bash
pip install openai langchain
```

####  使用 Text Embedding Models 的最简单方法

导入 OpenAIEmbeddings：

```python
from langchain.embeddings import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(openai_api_key = "YOUR_OPENAI_API_TOKEN")
result = embeddings.embed_query("This is an apple.")

```

打印 result ，我们就可以看到 "This is an apple." 这句话的向量空间表达数组。

```
[0.012008527293801308,
 -0.001835253438912332,
 0.00145026296377182,
 -0.0030227703973650932,
 -0.005661344155669212,
 0.005086636636406183,...]
```

打印出的数字并不能直观看到这些词在向量空间中的位置，但如果你对向量可视化感兴趣，可以访问OpenAI的官方网站，那里有许多关于词嵌入（embedding）可视化的实践场所（playground）供你探索和学习。
# 提示

一切都始于Prompts 。

事实上,Langchain 的一切都是围绕Prompts 构建的,Prompts 对于您使用 Langchain 所做的一切至关重要。

让我们稍微谈谈 Prompts 是什么以及在 Langchain 中的Prompts 模板的概念。

我们跟聊天机器人沟通的唯一方式是输入 Prompts。

什么是 Prompts ？Prompts 指的是模型的输入,　我们发给大语言模型的内容。

在 Langchain 中,Prompts 基本上是让模型在某些输入文本的**条件限制**下生成文本的过程。这些模型通常只是继续预测下一个标记,通常是下一个单词,然后通过这种方式组装文本。

Prompts 本身对输出结果有很大的影响,以至于这导致了一个全新的领域,人们现在称之为 Prompts 工程。

早期的Prompts 非常简单,您可能会问一些简单的问题,比如乾隆是什么时候皇帝? 希望大型语言模型在其权重中学到足够的知识来回答这个问题,并生成一个输出结果。

随着模型的不断改进, Prompts 从这些简单的形式转变为更加复杂的形式,现在我们看到人们给出了包含整个上下文和各种信息的Prompts 。

从GPT论文中,我们可以看到其中有不同类型的Prompts ,论文中包含了许多不同的Prompts 。

您还可以看到这个示例中我们有所谓的样本（few shot）模板,即我们不仅仅告诉Prompts 一些内容,还会给它一些示例,然后让它生成一个新的答案或示例。

在 Langchain 中,所有这些都由Prompts 模板来处理,因此了解Prompts 的不同部分非常重要,因为您将在Prompts 中构建不同的部分。

在 Langchain 中,您将构建一个Prompts ,并在其中注入不同的内容。例如,这是一个简单的Prompts ,以一个上下文开始,告诉模型您是一个起名字助手,可以帮助用户根据所提供的业务描述创建业务名称,然后给出一些示例,以特殊的方式给出这些示例,这样模型就可以学习生成与示例相似的内容。然后,我们可以传递问题或任务,并为输出结果进行预处理。

不仅 Langchain 中经常使用, 也在许多其他Prompts 系统中使用。这些 Prompts 系统, 基本上是在占位符注入一些用户输入的信息,这是用于回复的一些要点,再次注入一些信息。把 Prompts 发给语言模型,语言模型给我们一个回答。

这样可以针对许多不同的任务进行操作,可以用于分类,也可以用于其他各种不同的任务。

在实际的应用开发中, 开发者一定是想要动态插入提示,而不是硬编码的。还想要有多个灵活的值,而不只有一句提示。LangChain 提供了 PromptTemplate 基本类对象来构建使用多个值的提示。

LangChain 还提供了多个类和函数,构建了很多不同实例的 PromptTemplate, 让开发者使用提示更加容易,满足这些实际开发的多样需求。

这一章的重要概念包括提示模板、输出解析器、示例选择器和控制提示长度。

我们首先来理解提示模板 PromptTemplate。


# 2.2 提示

新的大语言模型的编程方式是通过提示（prompt）进行的。"提示"是 Langchain 与大模型交互的最重要组件。Langchain 中的提示输入通常由多个部分构成。LangChain 提供了类和函数，以方便构造和操作提示。


####   提示的定义

什么是提示？提示指的是模型的输入,　我们发给大语言模型的内容。

在 Langchain 中,提示是让模型在输入的提示文本**条件限制**下生成文本的过程。这些模型通常是继续预测下一个标记,通常是下一个单词,然后通过这种方式组装文本输出。

####   你的第一个提示

在编程世界里，我们常说"Hello World"是每个程序员的第一步。现在，对于使用大语言模型编程的我们，"你好", 一个emoj 表情或者任何一句话，也是我们的第一步。因为无论我们的任务多么复杂，一切都始于一个prompt。

还记得你第一次使用一个聊天机器人时，你敲打键盘的第一句话是什么吗？是不是 “你好” ？

![图 2-1](/img/2.2-1.png)

事实上，这个看似简单的“你好”，在编程世界里就像完成了“Hello World”的编程那样具有重要意义。这是因为在这个场景中，“你好”是一个提示。

你的第一个提示代表你已进入 AI 编程的美丽新世界。

提示对大语言模型的输出结果有很大的影响,以至于这导致了一个全新的领域,人们现在称之为提示工程。

####   提示的演化

早期的提示非常简单,你可能会问一些简单的问题,比如乾隆是什么时候皇帝? 希望大型语言模型在它之前数据集学到足够的知识来回答这个问题,并生成一个输出结果。

随着模型的不断改进,提示从这些简单的形式转变为更加复杂的形式,现在我们看到人们给出了包含整个上下文和各种信息的提示。还有少数示例提示模板（Few-shot）,即我们不仅仅告诉提示一些内容,还会给它一些示例,然后让它生成一个新的答案或示例。

从GPT论文中,我们可以看到其中有不同类型的提示,论文中包含了许多不同的提示。

随着技术的不断进步和人工智能领域的深入研究，如今的提示已经越来越复杂和工程化。尤其是在参数化模型输入和样本示例这两个方面，都做出了巨大的努力来提高提示的质量，以实现更好地控制大型语言模型的目标。

####   模型对提示的影响

我们现在进入了模型开发的白热阶段，也就是“千模大战”, 未来可能会有更多模型提供商加入,那其他的模型对提示会有什么影响？

LangChain 构造和操作的提示适用于其他模型，提示对于其他模型的影响，取决于模型的规模和质量。如果你想使用免费的Hugging Face模型中的一个，我们可以用LangChain 做相同的事情。

以下是使用 Google T5-Flan-XL 模型的代码：

```python
from langchain.llms import HuggingFaceHub
```
在这里，我只是设置了一个Hugging Face版本的Google T5-Flan-XL模型。这不是最大的模型，最大的模型是XXL，但是这个模型的规模肯定没有 OpenAI 的`text-davinci-003`模型大，不过你会看到我们仍然可以得到一些连贯的回应。

```python
llm_hf = HuggingFaceHub(
    repo_id="google/flan-t5-xl",
    model_kwargs={"temperature":0.9 }
)
```

你会注意到，这个回应不同，如果我们再次运行它，我们仍然得到相同的东西，但每个模型都给出了不同的回应，但基本上做的事情都是一样的。


```python
text = "Why did the chicken cross the road?"

print(llm_hf(text))
```

```
It was hungry.
```

####   Langchain 里的提示

Langchain对于你开发应用的提示功能非常重要。通过它，你可以更有效地指导你的应用来实现预期的功能。而且，一旦模板构建好，你只需要提供关键的信息。让我们通过一个具体的实例来说明这一点。

假设你正在开发一个专门为电影爱好者设计的聊天机器人，该机器人的目标是提供详细的电影信息。此时，Langchain的提示功能就可以派上用场。你可以建立一个模板，像这样：

- 提示："""请告诉我《{电影名}》的导演是谁？"""
- 提示："""《{电影名}》的主要演员有谁？"""
- 提示："""《{电影名}》的剧情简介是什么？"""
- 提示："""《{电影名}》的评分是多少？"""
  
在模板建立好之后，用户只输入电影名，比如《阿凡达》或者《泰坦尼克号》。此时，模型会接收到Langchain提供的结构化提示输入，如“请告诉我《阿凡达》的导演是谁？”或“《泰坦尼克号》的主要演员有谁？”等。

然后，模型会根据这些提示，生成相应的输出。由于Langchain的提示非常具体和结构化，模型生成的答案质量通常较高，符合用户的预期。你可以将更多的精力集中在用户和业务上，而不必担心如何指导模型生成满足用户需求的答案。因为一旦模板建立好，Langchain的提示功能就会帮你做这件事。

在 Langchain 中,所有这些都由提示模板来处理,因此了解提示的组成部分非常重要,对于你使用 Langchain 所做的一切至关重要。

Langchain 里的提示模板主要2个功能是：

####  1. 参数化模型输入

参数化模型输入是通过将模型输入转化为可动态的参数形式，从而实现模型输入的灵活调整。具体来说，它允许我们根据特定的需求和环境条件，来动态地调整模型的输入。这样一来，我们可以根据特定的任务需求，通过调整输入参数，使模型生成满足我们需求的输出。例如，我们正在开发一个聊天机器人，并希望给它一个具有个性的名字，比如"小智"或"问答达人"等。我们可以将机器人的名字作为输入参数，通过Langchain 里的提示模板构建一个动态的提示。"小智"或"问答达人"就是我们输入的参数，Langchain 里的提示模板接受这个参数，构造成一个完整的提示输入给大语言模型,。


这是最基本的示例，但它使我们能够动态更改带有用户输入的提示。

这个代码中,`{botname}` 是要被替换的占位符。

```
# Import prompt and define PromptTemplate
from langchain import OpenAI
from langchain import PromptTemplate

template = """
从现在开始，你的名字是：{botname}。每次打招呼的时候，请回应：“你好，我是{botname}聊天机器人，有什么可以帮到你？”。
"""

prompt = PromptTemplate(
    input_variables=["botname"],
    template=template,
)
# Run LLM with PromptTemplate
llm = OpenAI(openai_api_key="YOUR_OPENAI_API_KEY")
llm(prompt.format(botname=""))
```
当我们传入 `botname="小智"` 时, PromptTemplate 方法,将 "小智" 注入到提示中，这个聊天机器人就以“小智”的身份回应你的问题。 


####  2. 动态选择包含在提示中的示例

"示例提示模板"（又称为 "少数示例学习" 或 "Few-shot learning"）已经成为当前提升大型语言模型性能的一个重要策略。让我们通过一个具体的例子来了解一下这种方法的工作原理。

假设我们正在开发一个聊天机器人，该机器人需要回答用户关于天气的各种问题。这时，我们可以使用示例提示模板或者少数示例学习的方法来训练和优化我们的机器人。

首先，我们可以为聊天机器人提供一些关于如何回答天气相关问题的示例。这些示例包括如下内容：

User：“今天北京的天气怎么样？” 
Chatbot：“我不确定，让我查一下。今天北京的天气预报是晴朗。”

User：“明天上海会下雨吗？”
Chatbot：“让我为你查看明天上海的天气预报。不，明天上海不会下雨。”

这些示例将帮助机器人理解用户可能会提出的天气相关问题，以及如何回答这些问题。然后，当用户实际提问时，机器人  (Chatbot) 就可以参考这些已经学习过的示例，来生成合适的回答。这就是少数示例学习的工作原理。

通过使用这种方法，我们可以大大提高聊天机器人的性能。因为它允许我们以一种非常直接而灵活的方式，为机器人提供具体的指导。这使得我们的机器人不仅可以更准确地回答用户的问题，而且还可以更好地适应各种不同的环境和情境。因此，示例提示模板已经成为当前提升大型语言模型性能的一个重要工具。

在实际开发中，我们通过将这两种方法结合起来，我们可以更好地控制大型语言模型的表现。这是因为我们不仅可以根据需求调整模型的输入，还可以选择最适合的训练数据作为输入。这就提供了一种强大的工具，可以帮助我们实现更精细、更有目标性的模型控制。



# 2.2.1 提示模板

实际上, 提示模板是一个 "未完成" 的输入示例。Langchain 提供了我们通过告诉它要替换占位符的值来创建格式化之后的提示。Langchain 内置的提示模板对象 (PromptTemplate), 使我们能够将用户输入插入到提示文本的占位符中，并且可以增加额外的知识,额外的上下文, 将用户输入进行格式化，最终提供给语言模型高质量的提示,从而获得更好的输出结果。

####  提示模板的定义

在Langchain中，提示模板是一种可复制生成提示的方式。它包含一个文本字符串（即"模板"），该模板可以接收用户的一系列参数，从而生成一个提示。比如说，我们要给公司的产品取一个好听的名字，用户输入的就是产品的品类名字，例如“袜子”或“毛巾”。然而，我们并不需要为每一个品类都编写一个提示，而是使用这个模板，根据用户输入的不同品类，生成对应的提示。

####  模板的三个组成部分

提示模板主要由三个部分组成：语言模型的说明，一组少数示例，以及一个问题或任务。这三部分中，语言模型的说明和示例是固定的常量，而问题或任务是可以由用户改变的变量。Langchain将这三个部分组合起来，并为输出结果进行格式化的处理，以生成一个完整的提示。

提示模板在Langchain中得到了广泛的应用，但也在许多其他的大语言模型提示系统中使用。我们常常能看到类似的情况。Langchain的提示模板在其他模型中是通用的。

提示模板有许多实际应用场景。比如，它在做摘要或分类时非常实用。以新闻摘要为例，我们可以构建一个模板，模型的任务就是生成新闻的摘要。用户的输入可能是复制黏贴了一篇完整的新闻文章，给语言模型的说明可能是“生成这篇新闻的摘要”，示例则可以是一些已经生成过的新闻摘要。然后，Langchain将这三个部分组合起来，生成一个完整的提示，如：“请为这篇新闻生成一个摘要。”  根据这个提示，模型就能生成一个符合要求的新闻摘要。这样，我们就可以利用提示模板，轻松地为任意一篇新闻生成摘要。

####   一个最简单的示例

这是最基本的示例,这个提示模板没有包含示例,但它使我们能够动态更改带有用户输入的提示。

```
from langchain import PromptTemplate

template = """
You are an expert data scientist with an expertise in building deep learning models. 
Explain the concept of {concept} in a couple of lines
"""

prompt = PromptTemplate(
    input_variables=["concept"],
    template=template,
)

prompt.format(concept="NLP")
```
当我们传入 `concept="NLP"` 时, PromptTemplate 方法,将 "NLP" 注入到模板中

我们得到的结果是:

```
'\nYou are an expert data scientist with an expertise in building deep learning models. \nExplain the concept of NLP in a couple of lines\n'
```

这个代码示例将用户输入变量 `input_variables`和开发者定义的 `template` 传参给 `PromptTemplate`对象, 然后调用 `format` 方法，格式化为语言模型可用的字符串提示。

####  构造提示模板的步骤

构造模板包含取出模板和格式化两个步骤。`from_template`即取出模板，`format`即格式化用户输入。Langchain, 内置了很多模板，比如聊天模型的提示模板，角色的提示模板。格式化可以是普通字符串用作通用语言模型输入，也可以是用作聊天模型输入的Message对象。

上面这个代码还可以简化为：

```
# 导入PromptTemplate和template定义同上

prompt = PromptTemplate.from_template(template).format(concept="NLP")
prompt

```

这段代码可以用自然语言描述为：取出内置模式给template使用，传参用户的输入后格式化输出完整提示。无论是哪一种提示模板，都是基于 PromptTemplate 对象，我们基于这个对象构造自己的提示模板。

内置提示模板，是Langchain简化开发的方式。即使你不指定 `input_variables` , Langchain 会自行推断，只要你告诉了它取出哪个模式。


####   专用聊天模型的提示模板

专用聊天模型的提示模板的不同之处，输入的是消息列表, 支持输出 Message 对象。Langchain 的优势在于提供了聊天提示模板 （ChatPromptTemplate）和角色消息提示模板。角色消息提示模板包含AIMessagePromptTemplate, SystemMessagePromptTemplate 和 HumanMessagePromptTemplate 三种角色消息提示模板。

无论看起来这些多么复杂，他们都遵循构造模板的步骤，取模式和格式化。聊天模型的提示模板取是消息提示模式，格式化多了消息列表的选择。

我们将上一个示例代码改造为聊天提示模板，下面是代码示例：

先导入消息提示模板：
```
from langchain.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
```

改造思路是生成人类和系统的消息提示，通过ChatPromptTemplate整合为消息列表提示：
先使用 from_template 取出消息模式，传入定义的template模板字符串，生成人类和系统两种模板。
```
from langchain import PromptTemplate

template = """
You are an expert data scientist with an expertise in building deep learning models. 
"""

system_message_prompt = SystemMessagePromptTemplate.from_template(template)
human_template="Explain the concept of {concept} in a couple of lines"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

```
然后转化为聊天提示模板类型：

```
chat_prompt = ChatPromptTemplate.from_messages(system_message_prompt, human_message_prompt])
```

最后指定输出为消息类型的格式, 字符串的输出则是这样：

```
chat_prompt.format(concept="NLP")
```
打印输出的是：

```
'System: \nYou are an expert data scientist with an expertise in building deep learning models. \n\nHuman: Explain the concept of NLP in a couple of lines'
```
指定格式化为消息对象类型则是这样：
```
chat_prompt.format_prompt(concept="NLP").to_messages()
```
打印输出的是：

```
[SystemMessage(content='\nYou are an expert data scientist with an expertise in building deep learning models. \n', additional_kwargs={}),
HumanMessage(content='Explain the concept of 强化学习 in a couple of lines', additional_kwargs={}, example=False)]
```

尽管上述示例并没有展示如何加入示例，但请不要忧虑，接下来我们即将进入下一章节：少样本提示模板。这个章节将全面展示如何利用示例来优化我们的模板。

事实上，所有的提示模板都基于示例选择这一概念。为了从大语言模型中获取更加完整和准确的结果，少量示例学习（Few Shot Learning）就显得尤为重要。下一章少样本提示模板，就是在构造提示时，我们提供一些示例来指导模型的输出。这些示例可以帮助模型更好地理解任务要求，从而生成更加贴合用户预期的答案。这也正是`FewShotPromptTemplate`的价值所在：通过使用少量示例学习，我们能够有效提升模型的输出质量，让模型给出的回答更加准确和具有针对性。

# 2.2.2 少样本提示模板

少样本提示模板 (FewShotPromptTemplate)已经被证明是很有用的。通常用户只要输入关键信息，就可以获得预期满意的回答。这些示例可以硬编码，但如果动态选择，会更强大。Langchain 提供了少样本提示模板接受用户输入，然后返回包含示例列表的提示。

在这篇文章中，我们将详细探讨如何通过调整和增强提示，特别是加入一些具体的例子，让模型通过学习这些例子来提升自身的输出质量。你会惊讶地发现，这种微小的调整会对模型输出的结果产生巨大的影响。

首先，我们需要明白一点，这与我们之前所做的工作并无太大区别，其核心仍然是制作一个优秀的提示模板。现在，让我们通过一个具体的例子来理解这个过程。

假设我们现在的任务是让模型进行反义词接龙。在这个任务中，我们会给模型一个词，然后期望模型返回这个词的反义词。因此，我们需要提供一些示例，例如 "happy" 对应的反义词是 "sad"，"tall" 的反义词是 "short"，以此类推。然后，我们通过LangChain来设置我们的提示模板。

首先，我们会设置一个前缀，这样可以方便我们逐步构建提示的各个部分，以确保它们能完整地结合在一起。然后，我们会为模型设置一些标准的示例，以帮助模型理解任务需求。接下来，我们会设置一个Few Shot提示模板，然后在这里传入我们的示例。我们还会为此设置一个前缀和一个后缀，这样可以帮助模型更好地理解任务。

然后，我们运行代码，看看模型能否正确地生成我们期望的结果。例如，如果我们输入"big"，模型就应该返回"small"。这就是我们期望看到的反义词。

实际上，我们并没有看到整个提示，因为在大多数情况下，我们并不希望向用户展示完整的提示。因此，这就是使用提示模板制作提示的一种简单方法。

在这个过程中，标准提示模板和Few Shot提示模板都发挥了重要的作用。要注意的是，Few Shot提示模板的关键在于示例，因此你可以在这里使用一系列示例。这个示例只需要展示输入是什么，以及你期望的输出是什么。

我们通过下面代码，创建一个包含几个示例的少样本提示模板,来解释 FewShotPromptTemplate 的作用。

```
pip install openai langchain
```


导入 PromptTemplate, FewShotPromptTemplate 对象

```
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts.prompt import PromptTemplate
```

创建一个包含几个示例的列表

```
examples = [
    {"word": "happy", "antonym": "sad"},
    {"word": "tall", "antonym": "short"},
]
```

实例化 PromptTemplate 对象

```
example_prompt_format = """Word: {word}
Antonym: {antonym}
"""
example_prompt = PromptTemplate(input_variables=["word","antonym"], template=example_prompt_format)

print(example_prompt.format(**examples[0]))
```

实例化 FewShotPromptTemplate 对象

我们创建 FewShotPromptTemplate 对象，传入示例、示例格式化器、前缀、命令和后缀，这些都在指导 LLM 的输出。

此外，我们还可以提供输入变量 examples, example_prompt 和分隔符 example_separator="\n"，用于将示例与前缀 prefix 和后缀 suffix 分开。

```
few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt, # 上一步实例化PromptTemplate对象
    prefix="Give the antonym of every input\n",
    suffix="Word: {input}\nAntonym: ",
    input_variables=["input"],
    example_separator="\n",
)
```

我们可以生成一个提示，它看起来像这样。

```
print(few_shot_prompt.format(input="big"))

Give the antonym of every input

Word: happy
Antonym: sad

Word: tall
Antonym: short

Word: big
Antonym: 
```

FewShotPromptTemplate 是一种非常有用的范例，可以控制 LLMs 的输出并引导其响应。

在实际的应用开发中,我们的示例相当复杂, 现实是我们在提示中还要加入大量的历史聊天信息,可是大型语言模型可以传递的标记数量是有限的，大多数模型最多可以传递1024到16000个标记。

Langchain 提供了控制提示长度的方法, 有兴趣的读者可以在官方文档中查看 LengthBasedExampleSelector 和 SemanticSimilarityExampleSelector 两种高阶的示例选择器, Langchain 还支持自定义示例选择器, 自己动手做一个示例选择器也是个不错的选择。
# 2.2.3 扩展提示模板

LangChain提供了极其灵活的提示模板方法和组合提示的方式，满足各种开发需求。在所有的这些方法中，基础模板和少样本提示模板是最基础的，其他所有的方法都在此基础上进行扩展。

LangChain提供了一套默认的提示模板，可以生成适用于各种任务的提示。然而，可能会出现默认提示模板无法满足你的需求的情况。例如，你可能需要创建一个带有特定动态指令的提示模板。在这种情况下，LangChain 支持你可以创建一个自定义的提示模板。

为了个性化LLM应用，你可能需要将LLM与特定用户的最新信息进行组合。特征库可以很好地保持这些数据的新鲜度，而LangChain提供了一种方便的方式，可以将这些数据与LLM进行组合。做法是从提示模板内部调用特征库，检索值，然后将这些值格式化为提示。

针对聊天模型需求，LangChain提供了不同类型的消息提示模板。最常用的是AIMessagePromptTemplate，SystemMessagePromptTemplate和HumanMessagePromptTemplate，它们分别创建一个AI消息、系统消息和人类消息。

此外，LangChain还支持"部分"提示模板，也就是说，传入一部分所需的值，以创建一个只期望剩余子集值的新提示模板。LangChain以两种方式支持这一点：部分格式化字符串值和部分格式化返回字符串值的函数。这两种不同的方式支持不同的用例。

我们可以通过PipelinePrompt来组合多个提示。这在你希望重用部分提示时非常有用。

通常，将提示存储为文件，而不是Python代码，更为方便。这样可以方便地共享、存储和版本控制提示。LangChain中支持的文件类型有JSON 和 YAML, 框架理论上 LangChain 想要支持一切的文件类型。



# 2.2.4 示例提示选择器

示例提示对模型输出结果产生的影响是非常显著的。这点在实际操作中的反馈是明确无误的。但问题在于，我们可能有大量这样的示例，我们不可能全部输入给模型。而且，尝试适应所有示例可能会很快变得非常昂贵，尤其是在计算资源和时间上。这就是示例选择器发挥作用的地方，它帮助我们选择最适合的示例来提示模型。

以金融财报的摘要为例，如果举例的摘要过长，成本会变得很昂贵。大量、冗长的例子可能会占用模型可处理的 Token 数量，这使得模型无法充分理解和处理用户的真正输入，从而影响输出质量。

幸运的是，LangChain 的示例提示选择器（Example Selector）提供了一套工具，来解决这个问题。这些工具能基于策略选择合适的例子，如根据例子的长度、输入与例子之间的n-gram重叠分数来评估其相似度打分、找到与输入具有最大余弦相似度的例子, 以及多样性等因素来选择例子, 从而保持提示成本的相对稳定。

根据长度选择示例，是很普遍和现实的需求，以下是根据长度选择示例的代码：

```python
from langchain.prompts import PromptTemplate
from langchain.prompts import FewShotPromptTemplate
from langchain.prompts.example_selector import LengthBasedExampleSelector

examples = [
    {"word": "happy", "antonym": "sad"},
    {"word": "tall", "antonym": "short"},
    {"word": "energetic", "antonym": "lethargic"},
    {"word": "sunny", "antonym": "gloomy"},
    {"word": "windy", "antonym": "calm"},
]

example_prompt = PromptTemplate(
    input_variables=["word", "antonym"],
    template="Word: {word}\nAntonym: {antonym}",
)

example_selector = LengthBasedExampleSelector(
    examples=examples, 
    example_prompt=example_prompt, 
    max_length=25,
)

dynamic_prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=example_prompt,
    prefix="Give the antonym of every input",
    suffix="Word: {adjective}\nAntonym:", 
    input_variables=["adjective"],
)

print(dynamic_prompt.format(adjective="big"))
```
todo: 返回的结果

# 2.3 输出解析器

在使用 GPT-4 或者类似的大型模型时，一个常见的挑战是如何将模型生成的输出转化为我们可以在代码中直接使用的格式。这里，我们会使用 LangChain 的 输出解析器（OutputParsers） 工具来解决这个问题。

虽然语言模型输出的文本信息可能非常有用，但开发的应用与真实的软件数据世界连接的时候，我们更希望得到的不仅仅是文本，而是更加结构化的数据。为了在应用程序中展示这些信息，我们需要将这些输出转换为某种常见的数据格式。我们可以编写一个函数来提取这个输出，但这并不理想。比如在提示结尾加上“请输出答案为JSON格式”，模型会返回字符串形式的JSON，我们还需要通过函数将其转化为JSON对象。但是在实践中，我们常常会遇到异常问题，例如返回的字符串JSON无法被正确解析。

处理生产环境中的数据时，我们更可能会遇到千奇百怪的输入，导致模型的响应无法解析，增加额外的补丁来进行处理异常。这就使得整个处理流程变得更为复杂。

结构化数据,如数组或JSON对象, 在软件开发中起着至关重要的作用, 它提高了数据处理的效率，便利了数据的存储和检索，支持了数据分析，并且有助于提高数据质量。

还有,大语言模型目前确实存在一些问题，例如机器幻觉，这是指模型在理解或生成文本时产生的错误或误解。另一个问题是为了显得“聪明”，模型有时候会加入不必要的冗长和华丽的语句，这可能会导致模型过度详细，显得“话痨”了。比如你提示的结尾是“你的答案是：”，模型就不会“话痨”了。

在真实的开发环境中，我们不仅希望获取模型的输出结果，而且还希望能够进行后处理，比如解析输出的结构化数据。

这就是为什么在大语言模型的开发中，结构化数据，如数组或JSON对象，显得尤为重要。他们可以帮助我们更好地理解和处理模型的输出结果，比如通过解析输出的JSON对象，我们可以得到模型的预测结果，而不仅仅是一个长长的文本字符串。我们也可以根据需要对这些结果进行进一步的处理，例如提取关键信息，进行数据分析等。这样，我们不仅可以得到模型的“直接回答”，而且可以根据自己的需求进行定制化的**后处理**， 比如传递给下一个任务函数，从而更好地利用大语言模型。

这就是输出解析器的用武之地。

输出解析器是一组工具，它们的主要功能是处理和格式化模型的输出。这个工具组包括了几个部分，对于我们的需求来说，最关键的部分是 `Structured Output Parser`。这个工具可以把我们之前作为字符串返回的模型输出，转化为可以在代码中直接使用的数据结构。


输出解析器是帮助结构化语言模型响应的类。一个输出解析器主要需要实现两种方法：

"get_format_instructions"：这是一种返回字符串的方法，该字符串包含如何格式化语言模型输出的指示。
"parse"：这是一种接受字符串（模型的响应）并将其解析为某种结构的方法。

本书中,我们选取最常用的列表和JSON格式的输出解析器,通过代码来解释输出解析器的必要性。Langchain 还提供了 `Datetime parser`, `Enum parser` , `Structured output parser`  等类型的输出解析器。 
# 2.3.1 输出列表格式

数组或者列表是程序世界最基本的数据格式, 这种格式在大语言模型开发中,也可以发挥不少作用.

你可以想象大语言模型就像一个非常聪明的机器人，它可以回答很多问题。但是，有时候这个机器人会说一些很复杂或者不正确的话，就像它在猜你想要知道什么，或者想要显得自己很聪明一样。

我们可以让这个机器人把它的回答写在一张“清单”上，就像你写购物清单一样。这样，我们就可以很容易地看到它的每个答案，而不需要去理解一大堆复杂的话。这个“清单”还可以帮助我们更好地使用这个机器人的答案，比如我们可以把这个名单给到其他机器人让它们帮忙处理。

使用“清单”这样的方式，可以让我们更好地使用这个聪明的机器人，让它变得更有用，更容易理解。

在软件世界中,最常见的“清单”就是列表格式。

我们通过快速入门的代码,来看看 Langchain 是如何将模型的回答输出为“清单”式的列表格式。

下面是最简单的列表输出解析器代码：

导入库,实例化列表输出解析器对象（CommaSeparatedListOutputParser）, 预期我们获得列表的结果:

```
from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI

output_parser = CommaSeparatedListOutputParser()
```

获取格式化指令:

```
format_instructions = output_parser.get_format_instructions()
prompt = PromptTemplate(
    template="List five {subject}.\n{format_instructions}",
    input_variables=["subject"],
    partial_variables={"format_instructions": format_instructions}
)
```

这里我们引入 OpenAI 模型:

```
model = OpenAI(temperature=0)
```
实例化一个提示模板:
```
_input = prompt.format(subject="ice cream flavors")
output = model(_input)
```

将提示传入OpenAI 模型:
```
_input = prompt.format(subject="ice cream flavors")
output = model(_input)
```
调用解析器的parse方法,解析数据为列表格式.

```
output_parser.parse(output)
```

最终的结果是:

```
    ['Vanilla',
     'Chocolate',
     'Strawberry',
     'Mint Chocolate Chip',
     'Cookies and Cream']
```

从快速入门代码,我们使用了输出解析器的2大方法: 格式化 `output_parser.get_format_instructions()` 和 解析 `output_parser.parse()`

接下来,我们进入到语言模型应用开发中，最常用的一种格式： JSON 对象。


# 2.3.2 Pydantic JSON解析器

JSON {} 对象，这种格式最大的特点是人和机器都看得懂。

你可以把JSON对象想象成一个大家都认识的“信息盒子”。在这个“信息盒子”里，我们可以存储各种各样的信息，比如你的名字、你的年龄、你最喜欢的食物，甚至是你所有玩具的列表等等。这些信息都被整齐地放在“信息盒子”里，每一种信息都有自己的标签，比如“名字”、“年龄”、“食物”、“玩具”。

在我们开发语言模型应用的时候，我们经常用到这个“信息盒子”。因为它可以帮我们更好地整理和使用机器人的答案。比如，机器人可能会给我们一个包含很多信息的答案，而我们可以用这个“信息盒子”来把这些信息整理得更清晰，更易于理解和使用。

所以，JSON对象就像一个非常有用的“信息盒子”，可以帮助我们更好地使用和理解语言模型的答案。

请记住，大语言模型是有“缺陷”的抽象！你需要使用一个具有足够能力的模型来生成格式良好的JSON。在OpenAI 模型家族中，DaVinci可以做到这一点，但Curie的能力已经大幅下降。Langchain 这种输出解析器可以指定一个任意的JSON结构，并向大语言模型查询，输出符合该架构的JSON。

你可以使用Pydantic来声明你的数据模型。Pydantic的BaseModel就像一个Python数据类，但它具有实际的类型检查和强制转换功能。

下面是最简单的Pydantic JSON解析器代码：

导入语言模型 OpenAI和Prompts模板。

```
from langchain.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
```

获取格式化指令 `PydanticOutputParser`:

```
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, validator
from typing import List
```

这里我们引入 OpenAI 的模型 `text-davinci-003`。

```
model_name = "text-davinci-003"
temperature = 0.0
model = OpenAI(model_name=model_name, temperature=temperature)
```
定义你想要的数据格式：
```
# Define your desired data structure.
class Joke(BaseModel):
    setup: str = Field(description="question to set up a joke")
    punchline: str = Field(description="answer to resolve the joke")

    # You can add custom validation logic easily with Pydantic.
    @validator("setup")
    def question_ends_with_question_mark(cls, field):
        if field[-1] != "?":
            raise ValueError("Badly formed question!")
        return field


# And a query intented to prompt a language model to populate the data structure.
joke_query = "Tell me a joke."

# Set up a parser + inject instructions into the prompt template.
parser = PydanticOutputParser(pydantic_object=Joke)

prompt = PromptTemplate(
    template="Answer the user query.\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

_input = prompt.format_prompt(query=joke_query)

output = model(_input.to_string())

parser.parse(output)
```

将提示传入OpenAI 模型:
```
_input = prompt.format(subject="ice cream flavors")
output = model(_input)
```
调用解析器的parse方法,解析数据为列表格式.

```
output_parser.parse(output)
```

最终的结果是:

```
    Joke(setup='Why did the chicken cross the road?', punchline='To get to the other side!')
```

细心的读者会注意到,我们这次仍旧是使用了输出解析器的2大方法: 格式化 `output_parser.get_format_instructions()` 和 解析 `output_parser.parse()`。

掌握这两种方法，我们就能掌握输出解析器的使用方法。
# 2.3.4 结构化输出解析器

`OutputParsers`是一组工具，其主要目标是处理和格式化模型的输出。它包含了多个部分，但对于我们实际的开发需求来说，其中最关键的部分是结构化输出解析器（StructuredOutputParser）。这个工具可以将模型原本返回的字符串形式的输出，转化为可以在代码中直接使用的数据结构。

使用结构化输出解析器时，我们首先需要定义我们所期望的输出格式。解析器将根据这个定义来生成模型的提示，从而引导模型产生我们所需的输出。例如，假设我们想要得到的输出是包含“Brand”，“Success Probability”，和“Reasoning”三个部分的 JSON 格式。我们可以将这个要求在解析器中进行定义，随后解析器就会自动为我们生成相应的 prompts。

以下是示例的代码：

```markdown
# 示例：Structured Output Parser
response_pattern = {
    "Brand": "The brand name is {brand_name}.",
    "Success Probability": "The success probability is {success_probability}.",
    "Reasoning": "The reasoning is {reasoning}."
}

output_parser = StructuredOutputParser(response_pattern)
```

在此例中，`output_parser`将会把模型的输出按照我们定义的样式进行格式化。当我们运行这个解析器时，我们可以看到它成功地生成了我们需要的格式。最终，模型的输出将被格式化为包含“Brand”，“Success Probability”，和“Reasoning”三个部分的 JSON 格式，我们便可以在代码中直接使用它了。

无论你的应用需要什么样的输出格式，`OutputParsers`都能够帮助你轻松地得到。只需要定义你希望的输出样式，模型便能为你生成适合的结果，使你能更快地构建应用程序，提供更优质的用户体验。这个输出解析器特别适用于你想返回多个字段的情况。
# 本章小结

本章，我们深入探讨了大语言模型的输入输出（Model I/O）流程, 三个核心功能。我们首先介绍了模型的基础知识，阐述了模型如何理解和处理输入。接着，我们讲解了prompt提示的概念和使用，揭示了其在引导模型生成期望输出中的重要性。最后，我们详细解读了输出解析器的角色，明确了其在转化模型输出为结构化数据中的关键作用。

总的来说，这一章我们走过了从模型输入到输出的全过程，对大语言模型的运作机制有了更深入的理解。这是开发应用的最基础，也是打好基本功的关键。只有深入理解这些基础知识，我们才能在大语言模型的应用开发中乘风破浪，开创新的可能。
# 数据概览

在当代，大量的大型语言模型（LLM）应用需要用户特定的数据，而这部分数据并不包含在模型的训练集中。因此，如何加载、转化、存储和查询这些数据，便成为我们需要解决的核心问题。针对这一需求，LangChain提供了一整套解决方案。它通过一系列的构建模块，包括文档加载器（Document loaders）、文档转换器（Document transformers）、文本嵌入模型（Text embedding models）、向量存储（Vector stores）以及检索器（Retrievers），让我们能够自由地与自己的数据和大型语言模型进行交互。

在以下的篇章中，我们将详细地探讨每一个构建模块的实现原理。

####   2. 文档加载器：灵活加载文档的利器

文档加载器能够从众多不同的来源加载文档。一个文档可以简单地理解为一段文字和其相关的元数据。例如，文档加载器可以从一个简单的.txt文件中，从任意网页的文本内容，甚至从YouTube视频的字幕中加载数据。

文档加载器提供一个"load"方法，用于从配置的数据源加载数据作为文档。它们还可选择性地实现"lazy load"，以便于懒加载数据到内存中。

####   3. 文档转换器：为你的应用精确定制文档

一旦文档被加载，我们往往需要对它们进行一定的转换，以便更好地适应我们的应用。最简单的例子是，我们可能需要将一个长文档分割成更小的片段，以便它们能够适应模型的上下文窗口。LangChain内置了多种文档转换器，使得分割、组合、过滤和其他形式的文档操作变得轻而易举。

####   4. 文本嵌入模型：将无结构文本转化为浮点数列表

Embeddings类是为接口化文本嵌入模型而设计的一个类。目前有许多嵌入模型提供商（如OpenAI, Cohere, Hugging Face等），而这个类就是为了提供一个所有这些供应商的标准接口。

Embeddings类将一段文本创建为一个向量表示。这是有用的，因为我们可以将文本在向量空间中进行思考，并做诸如语义搜索这样的操作，我们可以寻找在向量空间中最相似的文本片段。

LangChain中的基础Embeddings类公开了两种方法：一种用于嵌入文档，另一种用于嵌入查询。前者接受多个文本作为输入，而后者接受单个文本。之所以将这两种方法分开，是因为一些嵌入提供商对待待查询的文档与查询本身有着不同的嵌入方法。

####   5. 向量存储：为你的嵌入数据提供存储和搜索功能

最常见的存储和搜索无结构数据的方式之一是将其嵌入并存储生成的嵌入向量，然后在查询时嵌入无结构查询并检索那些与嵌入查询“最相似”的嵌入向量。向量存储负责存储嵌入数据并执行向量搜索。

####   6. 检索器：返回文档的接口

检索器是一个能够根据无结构查询返回文档的接口。它比向量存储的应用更为广泛。检索器不需要能够存储文档，只需返回（或检索）文档即可。向量存储可以作为检索器的支撑结构，但也存在其他类型的检索器。

以上就是LangChain通过这些技术准备数据，让我们使用自己的数据与大型语言模型交互的全过程。LangChain用这种简单而高效的方式，让我们可以轻松地处理各种数据，并使之与我们的大型语言模型交互。
# 数据概览

在当代，大量的大型语言模型（LLM）应用需要用户特定的数据，而这部分数据并不包含在模型的训练集中。因此，如何加载、转化、存储和查询这些数据，便成为我们需要解决的核心问题。针对这一需求，LangChain提供了一整套解决方案。它通过一系列的构建模块，包括文档加载器（Document loaders）、文档转换器（Document transformers）、文本嵌入模型（Text embedding models）、向量存储（Vector stores）以及检索器（Retrievers），让我们能够自由地与自己的数据和大型语言模型进行交互。

在以下的篇章中，我们将详细地探讨每一个构建模块的实现原理。

####   2. 文档加载器：灵活加载文档的利器

文档加载器能够从众多不同的来源加载文档。一个文档可以简单地理解为一段文字和其相关的元数据。例如，文档加载器可以从一个简单的.txt文件中，从任意网页的文本内容，甚至从YouTube视频的字幕中加载数据。

文档加载器提供一个"load"方法，用于从配置的数据源加载数据作为文档。它们还可选择性地实现"lazy load"，以便于懒加载数据到内存中。

####   3. 文档转换器：为你的应用精确定制文档

一旦文档被加载，我们往往需要对它们进行一定的转换，以便更好地适应我们的应用。最简单的例子是，我们可能需要将一个长文档分割成更小的片段，以便它们能够适应模型的上下文窗口。LangChain内置了多种文档转换器，使得分割、组合、过滤和其他形式的文档操作变得轻而易举。

####   4. 文本嵌入模型：将无结构文本转化为浮点数列表

Embeddings类是为接口化文本嵌入模型而设计的一个类。目前有许多嵌入模型提供商（如OpenAI, Cohere, Hugging Face等），而这个类就是为了提供一个所有这些供应商的标准接口。

Embeddings类将一段文本创建为一个向量表示。这是有用的，因为我们可以将文本在向量空间中进行思考，并做诸如语义搜索这样的操作，我们可以寻找在向量空间中最相似的文本片段。

LangChain中的基础Embeddings类公开了两种方法：一种用于嵌入文档，另一种用于嵌入查询。前者接受多个文本作为输入，而后者接受单个文本。之所以将这两种方法分开，是因为一些嵌入提供商对待待查询的文档与查询本身有着不同的嵌入方法。

####   5. 向量存储：为你的嵌入数据提供存储和搜索功能

最常见的存储和搜索无结构数据的方式之一是将其嵌入并存储生成的嵌入向量，然后在查询时嵌入无结构查询并检索那些与嵌入查询“最相似”的嵌入向量。向量存储负责存储嵌入数据并执行向量搜索。

####   6. 检索器：返回文档的接口

检索器是一个能够根据无结构查询返回文档的接口。它比向量存储的应用更为广泛。检索器不需要能够存储文档，只需返回（或检索）文档即可。向量存储可以作为检索器的支撑结构，但也存在其他类型的检索器。

以上就是LangChain通过这些技术准备数据，让我们使用自己的数据与大型语言模型交互的全过程。LangChain用这种简单而高效的方式，让我们可以轻松地处理各种数据，并使之与我们的大型语言模型交互。


# 3.1.1 加载器概览

在LangChain的数据处理流程中，Document Loaders起着至关重要的作用。它们被用来从各种来源加载数据，并将其转换为“文档”（Document）的形式。

一个“文档”可以理解为一段文本及其相关元数据。例如，我们可以有专门用于加载简单.txt文件的文档加载器，也可以有加载任何网页的文本内容的文档加载器，甚至还可以有加载YouTube视频转录文本的文档加载器。不同类型的文档加载器，使得LangChain可以从各种各样的数据源中抽取并处理数据。

这些文档加载器都会暴露出一个名为"load"的方法，用于从配置的数据源加载数据作为文档。这个"load"方法可以从指定的数据源中读取数据，并将其转换成一份或多份文档。这使得LangChain能够处理各种形式的输入数据，不仅仅限于文本文件，还可以是网页、视频字幕等等。

值得注意的是，文档加载器还可以选择性地实现一个名为"lazy load"的方法，这个方法的作用是实现数据的懒加载，即在需要时才将数据加载到内存中。这种方式可以有效减少内存占用，并提高数据处理的效率。

总的来说，通过Document Loaders，LangChain可以将各种各样的数据源无缝地转换为标准的文档形式，为后续的数据处理和分析提供了坚实的基础。

下面是最简单的文档加载器的代码示例：

加载简单.txt文件的文档加载器。

```
from langchain.document_loaders import TextLoader

loader = TextLoader("./index.md")
loader.load()

```
打印导入的结果。

```
[
    Document(page_content='---\nsidebar_position: 0\n---\n# Document loaders\n\nUse document loaders to load data from a source as `Document`\'s. A `Document` is a piece of text\nand associated metadata. For example, there are document loaders for loading a simple `.txt` file, for loading the text\ncontents of any web page, or even for loading a transcript of a YouTube video.\n\nEvery document loader exposes two methods:\n1. "Load": load documents from the configured source\n2. "Load and split": load documents from the configured source and split them using the passed in text splitter\n\nThey optionally implement:\n\n3. "Lazy load": load documents into memory lazily\n', metadata={'source': '../docs/docs_skeleton/docs/modules/data_connection/document_loaders/index.md'})
]
```
# 多元加载器

LangChain通过实现各种服务提供商的数据加载器（Loader），实现了对多元化数据源的处理。这些加载器的设计使得LangChain可以无缝地从各种服务提供商加载数据，进一步扩大了其在不同业务场景的应用范围。下面，我们将对这些加载器进行分类并简要描述。

####   1. 文件和目录加载器

这类加载器主要处理存储在本地或云端的文件和目录，例如`CSVLoader`，`DirectoryLoader`，`JSONLoader`，`S3DirectoryLoader`，`S3FileLoader`等。

####   2. 文本和标记语言文件加载器

这类加载器负责处理各种文本和标记语言格式的文件，如`BSHTMLLoader`，`MarkdownLoader`，`TextLoader`，`UnstructuredHTMLLoader`，`UnstructuredMarkdownLoader`等。

####   3. 文档和电子书加载器

处理电子书和文档的加载器包括`Docx2txtLoader`，`PDFMinerLoader`，`UnstructuredEPubLoader`，`UnstructuredPDFLoader`，`UnstructuredWordDocumentLoader`等。

####   4. 社交媒体和在线论坛加载器

这类加载器主要用于处理来自社交媒体和在线论坛的数据，如`FacebookChatLoader`，`GitHubIssuesLoader`，`RedditPostsLoader`，`TwitterTweetLoader`等。

####   5. 云服务提供商加载器

对于云服务提供商的数据源，LangChain提供了相应的加载器，如`AirbyteJSONLoader`，`AirtableLoader`，`AzureBlobStorageContainerLoader`，`AzureBlobStorageFileLoader`，`GoogleDriveLoader`，`OneDriveLoader`等。

####   6. 文本聊天和消息服务加载器

这类加载器主要处理各种聊天和消息服务的数据，如`DiscordChatLoader`，`SlackDirectoryLoader`，`TelegramChatApiLoader`，`TelegramChatFileLoader`，`WhatsAppChatLoader`等。

####   7. 网页和网站数据加载器

对于网页和网站的数据，如`SitemapLoader`，`UnstructuredURLLoader`，`WebBaseLoader`等加载器提供了处理方法。

####   8. 其他特殊类型的加载器

这类加载器主要处理一些特定的或者特殊的数据源，如`HuggingFaceDatasetLoader`，`UnstructuredImageLoader`，`WeatherDataLoader`，`YoutubeAudioLoader`等。

总的来说，通过实现这些加载器，LangChain可以从各种各样的服务提供商处加载数据，进一步提升了它的多样化数据处理能力。这对于需要处理大量、多样性数据的机器学习、自然语言处理等领域具有非常重要的意义。
# LangChain如何加载不同格式的数据

LangChain的数据加载能力并不限于单一的数据源或格式，它可以处理各种常见的数据格式，例如CSV、文件目录、HTML、JSON、Markdown以及PDF等。下面，我们将分别解析一下这些不同格式数据的加载方法。

####   1. CSV文件的加载

逗号分隔值（Comma-Separated Values，简称CSV）文件是一种使用逗号来分隔值的文本文件。每一行都是一条数据记录，每条记录包含一个或多个用逗号分隔的字段。LangChain可以加载CSV数据，其中每一行都被视为一个独立的文档。

####   2. 文件目录的加载

对于文件目录，LangChain提供了一种方法来加载目录中的所有文档。在底层，它默认使用UnstructuredLoader来实现这个功能。这意味着，只要将文档存放在同一目录下，无论数量多少，LangChain都能够将它们全部加载进来。

####   3. HTML文件的加载

HTML是用于设计在Web浏览器中显示的文档的标准标记语言。LangChain可以将HTML文档加载为我们后续使用的文档格式。这就意味着，我们可以直接从网页上提取并处理数据。

####   4. JSON文件的加载

JSON是一种使用人类可读文本来存储和传输数据对象的开放标准文件格式和数据交换格式，这些对象由属性-值对和数组（或其他可序列化值）组成。LangChain的JSONLoader使用指定的jq模式来解析JSON文件。jq是一种适用于Python的软件包。JSON文件的每一行都被视为一个独立的文档。

####   5. Markdown文件的加载

Markdown是一种使用纯文本编辑器创建格式化文本的轻量级标记语言。LangChain可以将Markdown文档加载为我们后续使用的文档格式。

####   6. PDF文件的加载

PDF（Portable Document Format）是Adobe在1992年开发的一种文件格式，用于以独立于应用软件、硬件和操作系统的方式呈现文档，包括文本格式化和图像。LangChain可以将PDF文档加载为我们后续使用的文档格式。

总的来说，通过对各种不同数据格式的加载能力，LangChain为大规模、多样性的数据处理提供了强大的支持。
# LangChain的文档转换器和文本分割器：工作原理与应用

LangChain为处理语言数据提供了一系列内置工具，包括文档加载器、文档转换器和文本分割器等。在您加载了文档之后，通常需要对其进行转换以更好地适应您的应用。这就需要用到LangChain的文档转换器和文本分割器。

####   1. 文档转换器

文档转换器可以轻松地将文档进行分割、合并、过滤和其他操作，以满足您的实际需求。例如，您可能希望将长文档分割成小块，以便适应您模型的上下文窗口。

####   2. 文本分割器

当处理长文本时，往往需要将文本分割成块。尽管这看起来简单，但实际上可能涉及很多复杂性。理想情况下，您会希望将语义相关的文本部分保持在一起。而"语义相关"的含义可能取决于文本的类型。下面将介绍几种实现这一目标的方法。

在高层次上，文本分割器的工作原理如下：

- 将文本分割成小的、语义上有意义的块（通常是句子）。
- 将这些小块开始组合成一个大的块，直到达到某个大小（通过某种函数进行测量）。
- 一旦达到该大小，将该块作为自己的文本片段，然后开始创建新的文本块，新的文本块和前一个文本块会有一些重叠（以保持块与块之间的上下文）。

这意味着，您可以沿着两个不同的轴来定制您的文本分割器：

- 文本如何被分割
- 块的大小如何被测量

####   3. 使用文本分割器

默认推荐的文本分割器是`RecursiveCharacterTextSplitter`。这个文本分割器接受一个字符列表，它尝试基于第一个字符进行分割，但如果任何块太大，它就会移动到下一个字符，依此类推。默认情况下，它尝试分割的字符是["\n\n", "\n", " ", ""]。

除了可以控制分割的字符，您还可以控制以下几点：

- `length_function`：如何计算块的长度。默认只计算字符数量，但是在此通常会传入一个标记计数器。
- `chunk_size`：您的块的最大大小（由长度函数测量）。
- `chunk_overlap`：块之间的最大重叠。有一些重叠可以在块之间保持连续性（例如采用滑动窗口的方式）。
- `add_start_index`：是否在元数据中包含每个块在原始文档中的起始位置。

通过以上内容，我们可以看到LangChain的文档转换器和文本分割器为处理和转换大规模文本提供了有效的工具，无论是文本的分割、组合、过滤还是其他操作，都能够得心应手。
# 文本分割方法

在处理大规模文本数据时，LangChain提供了多种文本分割方法，以满足各种类型的应用需求。本文将详细介绍这些方法的特性和工作原理。

####   1. 按字符分割

这是最简单的方法。它基于字符（默认为"\n\n"）进行分割，并通过字符数量来衡量块的大小。

- 文本如何分割：按单个字符
- 块大小如何测量：按字符数量

####   2. 代码分割

`CodeTextSplitter`允许你对多种语言的代码进行分割。导入枚举`Language`并指定语言即可。

####   3. Markdown标题文本分割器

这种分割器的动机源于许多聊天或问答应用需要在嵌入和向量存储之前将输入文档进行分块。Pinecone的这些注释提供了一些有用的提示。

####   4. 递归按字符分割

这是通用文本推荐的文本分割器。它由一系列字符参数化。直到块足够小，它会尝试按顺序分割它们。默认列表是["\n\n", "\n", " ", ""]。这样做的效果是尽可能地将所有段落（然后是句子，然后是单词）保持在一起，因为这些通常看起来是最强的语义相关的文本部分。

####   5. 按标记分割

语言模型有一个标记限制。你不应该超过这个标记限制。因此，当你将文本分割成块时，计算标记的数量是个好主意。有许多标记化器。当你在文本中计数标记时，应该使用与语言模型中使用的相同的标记化器。

通过以上内容，我们可以看到LangChain提供的各种文本分割方法能够满足不同类型的文本处理需求，无论是基于字符的分割，还是基于特定语言代码的分割，甚至是针对Markdown格式文本的分割，都具有各自的优点和特性，为处理和转换大规模文本提供了有效的工具。
# 优化文本处理的方法

在处理大规模文本数据时，我们可以利用Doctran库，基于OpenAI的函数调用特性，从文档中抽取具体的元数据。这种方法能够从多个方面助力我们处理文本信息。

####   1. Doctran抽取文档特性

我们可以从文档中抽取有用的特性，这对多种任务都很有帮助，包括：

- 分类：将文档分类到不同的类别
- 数据挖掘：提取可以用于数据分析的结构化数据
- 风格转换：改变文本的写作方式，使其更接近用户预期的输入，从而提高向量搜索的结果

####   2. Doctran询问文档

通常存储在向量库知识库中的文档以叙述或会话格式存储。然而，大多数用户查询都是以问题格式提出的。如果我们在将文档向量化之前将其转换为问答格式，我们可以增加检索到相关文档的可能性，减少检索到无关文档的可能性。

我们可以利用Doctran库，使用OpenAI的函数调用特性来“询问”文档。

####   3. Doctran翻译文档

通过嵌入比较文档具有跨多种语言工作的优点。"Harrison says hello" 和 "Harrison dice hola" 将在向量空间中占据相似的位置，因为它们在语义上有相同的含义。

然而，在将文档向量化之前，使用LLM将文档翻译成其他语言仍然可能是有用的。这在用户预期以不同的语言查询知识库，或者对于给定的语言没有可用的最先进的嵌入模型时，特别有帮助。

我们可以利用Doctran库，使用OpenAI的函数调用特性在语言之间翻译文档。

####   4. OpenAI函数元数据标签器

标记文档常常会很有用，这可以为文档添加如标题、调性或长度等结构化的元数据，以便后续进行更有针对性的相似性搜索。然而，对于大量文档，手动执行这个标记过程可能会很繁琐。

OpenAIMetadataTagger文档转换器通过根据提供的模式从每个提供的文档中提取元数据，自动化了这个过程。它在底层使用一个可配置的OpenAI Functions-powered链，所以如果你传递一个自定义的LLM实例，它必须是一个支持函数的OpenAI模型。

注意：这个文档转换器最适合处理完整的文档，所以在进行任何其他分割或处理之前，最好先使用整个文档运行它。

我们可以看到，通过使用Doctran库，我们不仅可以提取文档的重要特性，还可以对文档进行提问、翻译等操作，大大提升了文本处理的效率与准确性，无论是对于分类、数据挖掘，还是风格转换等任务，都具有重要的实际意义。同时，OpenAI的元数据标签器也为我们提供了一种自动化处理文档的有效方法，极大地简化了文本处理的过程。
# 文本嵌入模型概览

文本嵌入模型，如其名称所示，是用于处理文本信息的一种重要工具。具体而言，`Embeddings`类是专门设计用来与文本嵌入模型交互的类。目前，有许多嵌入模型提供商（如OpenAI、Cohere、Hugging Face等），而此类旨在为所有这些供应商提供一个标准的接口。

文本嵌入模型的核心工作就是为一段文本创建一个向量表示。这非常有用，因为我们可以在向量空间中思考文本，以及执行如语义搜索这样的操作，即在向量空间中寻找最相似的文本片段。

在LangChain中，基础的`Embeddings`类暴露了两种方法：一种用于嵌入文档，另一种用于嵌入查询。前者接收多个文本作为输入，而后者接收单个文本。之所以要将这两种方法区分开来，是因为一些嵌入供应商对于文档（要搜索的对象）与查询（搜索查询本身）有不同的嵌入方法。


# 文本嵌入类型

LangChain的文本嵌入类型丰富多样，无论面对何种文本处理需求或特定挑战，用户都有可能在其提供的嵌入类型列表中找到合适的解决方案。下面我们将对其支持的文本嵌入类型进行分类，并详细阐述各类的特点。


####  1. 自然语言模型（Natural Language Model）嵌入

此类嵌入主要利用诸如OpenAI、Hugging Face等自然语言处理（NLP）模型进行文本嵌入，特点是能够充分利用大规模预训练模型的语义理解能力。包括以下几种类型：

- "OpenAIEmbeddings"
- "HuggingFaceEmbeddings"
- "HuggingFaceHubEmbeddings"
- "HuggingFaceInstructEmbeddings"
- "SelfHostedHuggingFaceEmbeddings"
- "SelfHostedHuggingFaceInstructEmbeddings"

####  2. AI平台或云服务嵌入

此类嵌入主要依托AI平台或云服务的能力进行文本嵌入，典型的包括Elasticsearch、SagemakerEndpoint、DeepInfra等。这些嵌入方式主要特点是能够利用云计算的优势，处理大规模的文本数据。

- "ElasticsearchEmbeddings"
- "SagemakerEndpointEmbeddings"
- "DeepInfraEmbeddings"
- "VertexAIEmbeddings"

####  3. 专门的嵌入模型

此类嵌入主要是专门用于处理特定结构文本的嵌入模型，例如 AlephAlpha的AsymmetricSemanticEmbedding和SymmetricSemanticEmbedding，适用于结构不同或相似的文本。

- "AlephAlphaAsymmetricSemanticEmbedding"
- "AlephAlphaSymmetricSemanticEmbedding"
- "SentenceTransformerEmbeddings"
- "GooglePalmEmbeddings"

####  4. 自托管嵌入

这类嵌入一般适用于用户自行部署和管理的场景，如SelfHostedEmbeddings，给予用户更大的灵活性和控制权。

- "SelfHostedEmbeddings"

####  5. 仿真或测试用嵌入

FakeEmbeddings一般用于测试或模拟场景，不涉及实际的嵌入计算。

- "FakeEmbeddings"

####  6. 其他类型

此外，LangChain还支持一些其他类型的嵌入方式，如Cohere、LlamaCpp、ModelScope、TensorflowHub、MosaicMLInstructor、MiniMax、Bedrock、DashScope和Embaas等。这些嵌入方式各有特点，能够满足不同的文本处理需求。

在以上这些类型中，用户可以根据自己的具体需求，选择最合适的文本嵌入类型。同时，LangChain也将持续引入更多的嵌入类型，以进一步提升其处理文本的能力。
# 文本嵌入模型的应用

我们将通过以下步骤介绍如何使用这种文本嵌入模型。

####  1.环境设置

首先，我们需要安装OpenAI的Python包。

```
pip install openai
```

接着，我们需要获取一个API密钥来访问API，这可以通过创建一个账户并访问指定页面来获取。一旦我们拿到密钥，我们需要通过运行指定的代码将其设置为环境变量。

```
export OPENAI_API_KEY="..."
```

如果你不想设置环境变量，你也可以在初始化OpenAI LLM类时，通过名为openai_api_key的参数直接传入密钥。

```
from langchain.embeddings import OpenAIEmbeddings

embeddings_model = OpenAIEmbeddings(openai_api_key="...")
```
如果你设置环境变量的话，就不用传递参数。

```
from langchain.embeddings import OpenAIEmbeddings

embeddings_model = OpenAIEmbeddings()
```

####  2. 嵌入文本列表

这一步是将一组文本进行嵌入。你只需调用Embeddings类的相关方法，传入你要嵌入的文本列表即可。

```
embeddings = embeddings_model.embed_documents(
    [
        "Hi there!",
        "Oh, hello!",
        "What's your name?",
        "My friends call me World",
        "Hello World!"
    ]
)
len(embeddings), len(embeddings[0])
```

```
(5, 1536)
```

####  3. 嵌入单一查询

这一步是为了比较与其他嵌入文本的相似度，将单一的文本进行嵌入。同样的，你只需调用Embeddings类的相关方法，传入你要嵌入的文本即可。

```
embedded_query = embeddings_model.embed_query("What was the name mentioned in the conversation?")
embedded_query[:5]
```

```
[0.0053587136790156364,
 -0.0004999046213924885,
 0.038883671164512634,
 -0.003001077566295862,
 -0.00900818221271038]
```

以上便是使用LangChain进行文本嵌入的基本步骤和方法，相信通过以上的介绍，你已经对如何使用LangChain的文本嵌入模型有了基本的理解和掌握。
# 向量存储的原理和使用

####   向量存储的原理

在处理非结构化数据的存储和检索过程中，最常见的方式之一是将其嵌入，并存储生成的嵌入向量。然后在查询时，将非结构化查询嵌入，并检索与嵌入查询"最相似"的嵌入向量。向量存储就是用来存储嵌入数据和执行向量搜索的。

####   如何使用向量存储

下面我们将通过以下步骤介绍如何使用向量存储。

####  1.环境设置

本文展示了与向量存储相关的基本功能。使用向量存储的关键部分是创建要放入其中的向量，这通常是通过嵌入来创建的。因此，在深入了解这个内容之前，建议你先熟悉文本嵌入模型的接口。

本文使用的是FAISS向量数据库，该数据库使用了Facebook AI Similarity Search (FAISS)库。

```
pip install faiss-cpu
```
我们需要使用OpenAIEmbeddings，所以我们需要获取OpenAI API Key。

```
import os
import getpass

os.environ['OPENAI_API_KEY'] = getpass.getpass('OpenAI API Key:')
```

```
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS


raw_documents = TextLoader('../../../state_of_the_union.txt').load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
documents = text_splitter.split_documents(raw_documents)

embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(documents, embeddings)
```

####  2.相似度搜索

```
query = "What did the president say about Ketanji Brown Jackson"
docs = db.similarity_search(query)
print(docs[0].page_content)
```
```
    Tonight. I call on the Senate to: Pass the Freedom to Vote Act. Pass the John Lewis Voting Rights Act. And while you’re at it, pass the Disclose Act so Americans can know who is funding our elections.

    Tonight, I’d like to honor someone who has dedicated his life to serve this country: Justice Stephen Breyer—an Army veteran, Constitutional scholar, and retiring Justice of the United States Supreme Court. Justice Breyer, thank you for your service.

    One of the most serious constitutional responsibilities a President has is nominating someone to serve on the United States Supreme Court.

    And I did that 4 days ago, when I nominated Circuit Court of Appeals Judge Ketanji Brown Jackson. One of our nation’s top legal minds, who will continue Justice Breyer’s legacy of excellence.
```
通过向量进行相似度搜索：你也可以使用similarity_search_by_vector方法来搜索与给定的嵌入向量相似的文档，这个方法接受一个嵌入向量作为参数，而不是一个字符串。

```
embedding_vector = embeddings.embed_query(query)
docs = db.similarity_search_by_vector(embedding_vector)
```


####  3.异步操作

向量存储通常作为一个需要一些IO操作的单独服务运行，因此可能会异步调用。这有助于提升性能，因为你不需要浪费时间等待外部服务的响应。如果你使用的是异步框架，例如FastAPI，这可能非常重要。

Langchain支持在向量存储上进行异步操作。所有的方法都可以使用其异步对应方法调用，这些方法前缀为"a"，表示异步。

Qdrant是一个向量存储，支持所有的异步操作，因此将在本文中使用。

```
pip install qdrant-client
```
```
from langchain.vectorstores import Qdrant
```

####  4.异步创建向量存储

```
db = await Qdrant.afrom_documents(documents, embeddings, "http://localhost:6333")
```

####  5.相似度搜索

通过向量进行相似度搜索。

```
query = "What did the president say about Ketanji Brown Jackson"
docs = await db.asimilarity_search(query)
print(docs[0].page_content)
```

```
    Tonight. I call on the Senate to: Pass the Freedom to Vote Act. Pass the John Lewis Voting Rights Act. And while you’re at it, pass the Disclose Act so Americans can know who is funding our elections.

    Tonight, I’d like to honor someone who has dedicated his life to serve this country: Justice Stephen Breyer—an Army veteran, Constitutional scholar, and retiring Justice of the United States Supreme Court. Justice Breyer, thank you for your service.

    One of the most serious constitutional responsibilities a President has is nominating someone to serve on the United States Supreme Court.

    And I did that 4 days ago, when I nominated Circuit Court of Appeals Judge Ketanji Brown Jackson. One of our nation’s top legal minds, who will continue Justice Breyer’s legacy of excellence.
```    

####  6.最大边际相关性搜索（MMR）

最大边际相关性是为查询的相似性和所选文档的多样性进行优化。在异步API中也支持这个功能。

```
query = "What did the president say about Ketanji Brown Jackson"
found_docs = await qdrant.amax_marginal_relevance_search(query, k=2, fetch_k=10)
for i, doc in enumerate(found_docs):
    print(f"{i + 1}.", doc.page_content, "\n")
```

```
1. Tonight. I call on the Senate to: Pass the Freedom to Vote Act. Pass the John Lewis Voting Rights Act. And while you’re at it, pass the Disclose Act so Americans can know who is funding our elections.

Tonight, I’d like to honor someone who has dedicated his life to serve this country: Justice Stephen Breyer—an Army veteran, Constitutional scholar, and retiring Justice of the United States Supreme Court. Justice Breyer, thank you for your service.

One of the most serious constitutional responsibilities a President has is nominating someone to serve on the United States Supreme Court.

And I did that 4 days ago, when I nominated Circuit Court of Appeals Judge Ketanji Brown Jackson. One of our nation’s top legal minds, who will continue Justice Breyer’s legacy of excellence.

2. We can’t change how divided we’ve been. But we can change how we move forward—on COVID-19 and other issues we must face together.

I recently visited the New York City Police Department days after the funerals of Officer Wilbert Mora and his partner, Officer Jason Rivera.

They were responding to a 9-1-1 call when a man shot and killed them with a stolen gun.

Officer Mora was 27 years old.

Officer Rivera was 22.

Both Dominican Americans who’d grown up on the same streets they later chose to patrol as police officers.

I spoke with their families and told them that we are forever in debt for their sacrifice, and we will carry on their mission to restore the trust and safety every community deserves.

I’ve worked on these issues a long time.

I know what works: Investing in crime preventionand community police officers who’ll walk the beat, who’ll know the neighborhood, and who can restore trust and safety.
```

以上便是使用LangChain进行向量存储的基本步骤和方法，相信通过以上的介绍，你已经对如何使用LangChain的向量存储有了基本的理解和掌握。
# 检索器的定义和使用

检索器是什么？它与向量存储有什么区别？本文尝试解释检索器的定义以及如何使用。

####   检索器的定义

检索器是一种可以通过非结构化查询返回文档的接口。它比向量存储的应用范围更广。一个检索器并不需要能够存储文档，只需要能够返回（或检索）文档即可。向量存储可以作为检索器的支撑，但是也存在其他类型的检索器。

####   检索器的使用

LangChain中的BaseRetriever类的公开API非常简单。

```
from abc import ABC, abstractmethod
from typing import Any, List
from langchain.schema import Document
from langchain.callbacks.manager import Callbacks

class BaseRetriever(ABC):
    ...
    def get_relevant_documents(
        self, query: str, *, callbacks: Callbacks = None, **kwargs: Any
    ) -> List[Document]:
        """Retrieve documents relevant to a query.
        Args:
            query: string to find relevant documents for
            callbacks: Callback manager or list of callbacks
        Returns:
            List of relevant documents
        """
        ...

    async def aget_relevant_documents(
        self, query: str, *, callbacks: Callbacks = None, **kwargs: Any
    ) -> List[Document]:
        """Asynchronously get documents relevant to a query.
        Args:
            query: string to find relevant documents for
            callbacks: Callback manager or list of callbacks
        Returns:
            List of relevant documents
        """
        ...
```        

你可以调用get_relevant_documents或者异步的get_relevant_documents方法来检索与查询相关的文档，其中"relevance"是由你正在调用的特定检索器对象定义的。

我们主要关注的检索器类型是向量存储检索器，我们将重点介绍这个类型。

为了理解什么是向量存储检索器，理解向量存储的概念是非常重要的。所以我们首先来看看向量存储是什么。

默认情况下，LangChain使用Chroma作为向量存储来索引和搜索嵌入。在本文中，我们首先需要安装chromadb。

```
pip install chromadb
```
我们选择了在文档上进行问题回答的示例来进行演示，因为它很好地整合了许多不同的元素（文本分割器，嵌入，向量存储），并展示了如何将它们串联使用。

文档上的问题回答包括四个步骤：

1. 创建索引
2. 从索引创建检索器
3. 创建问题回答链
4. 提出问题！

每一个步骤都有多个子步骤和潜在的配置。在本指南中，我们将主要关注步骤（1）。我们将首先展示一个一行代码的示例，然后详细解析其中的内容。

首先，让我们导入公共类。

```
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
```

在通用设置中，接下来我们来指定我们想要使用的文档加载器。

```
from langchain.document_loaders import TextLoader
loader = TextLoader('../state_of_the_union.txt', encoding='utf8')
```

为了尽快开始，我们可以使用VectorstoreIndexCreator。

```
from langchain.indexes import VectorstoreIndexCreator
```

```
index = VectorstoreIndexCreator().from_loaders([loader])
```

```
    Running Chroma using direct local API.
    Using DuckDB in-memory for database. Data will be transient.
```    

创建索引后，我们就可以使用它来提出数据问题了！注意，这里面实际上也做了几个步骤，我们将在后面的指南中介绍。

```
query = "What did the president say about Ketanji Brown Jackson"
index.query(query)
```

```
    " The president said that Ketanji Brown Jackson is one of the nation's top legal minds, a former top litigator in private practice, a former federal public defender, and from a family of public school educators and police officers. He also said that she is a consensus builder and has received a broad range of support from the Fraternal Order of Police to former judges appointed by Democrats and Republicans."
```

```
query = "What did the president say about Ketanji Brown Jackson"
index.query_with_sources(query)
```

```
    {'question': 'What did the president say about Ketanji Brown Jackson',
     'answer': " The president said that he nominated Circuit Court of Appeals Judge Ketanji Brown Jackson, one of the nation's top legal minds, to continue Justice Breyer's legacy of excellence, and that she has received a broad range of support from the Fraternal Order of Police to former judges appointed by Democrats and Republicans.\n",
     'sources': '../state_of_the_union.txt'}
```


VectorstoreIndexCreator返回的是VectorStoreIndexWrapper，它提供了查询和带来源的查询功能。
如果我们只想直接访问向量存储，我们也可以这样做。

```
index.vectorstore
```

```
    <langchain.vectorstores.chroma.Chroma at 0x119aa5940>
```

如果我们只想直接访问VectorstoreRetriever，我们也可以这样做。

```
index.vectorstore.as_retriever()
```

```
    VectorStoreRetriever(vectorstore=<langchain.vectorstores.chroma.Chroma object at 0x119aa5940>, search_kwargs={})
```   

接下来，我们要深入了解一下索引是如何创建的。VectorstoreIndexCreator中隐藏了很多魔法。它做了什么呢？

在文档加载后，主要进行了三个步骤：

1. 将文档分割成块
2. 为每个文档创建嵌入
3. 在向量存储中存储文档和嵌入

让我们在代码中一步步解析这个过程。

加载文档。

```
documents = loader.load()
```
分割文档。

```
from langchain.text_splitter import CharacterTextSplitter
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)
```
创建嵌入。

```
from langchain.embeddings import OpenAIEmbeddings
embeddings = OpenAIEmbeddings()
```
创建向量存储
```
from langchain.vectorstores import Chroma
db = Chroma.from_documents(texts, embeddings)
```

```
    Running Chroma using direct local API.
    Using DuckDB in-memory for database. Data will be transient.
```
创建索引。

```
retriever = db.as_retriever()
```

我们像以前一样，创建问答链。

```
qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=retriever)
```

```
query = "What did the president say about Ketanji Brown Jackson"
qa.run(query)
```

```
    " The President said that Judge Ketanji Brown Jackson is one of the nation's top legal minds, a former top litigator in private practice, a former federal public defender, and from a family of public school educators and police officers. He said she is a consensus builder and has received a broad range of support from organizations such as the Fraternal Order of Police and former judges appointed by Democrats and Republicans."
```   

VectorstoreIndexCreator不过是一种封装了各种逻辑的工具。它的配置是灵活的，可以设定使用的文本分割器、嵌入方式以及使用的向量存储。例如，你可以按照以下方式对其进行配置：

```
index_creator = VectorstoreIndexCreator(
    vectorstore_cls=Chroma,
    embedding=OpenAIEmbeddings(),
    text_splitter=CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
)
```

我们希望能揭示出VectorstoreIndexCreator内部的运行机制。我们认为提供一种简单的索引创建方式是重要的，但同时，理解这个过程中底层的操作也同样关键。

# 检索器的应用

在高维空间中，距离型向量数据库检索工具通过将查询嵌入来找出相似的文档。然而，微妙的查询词汇改变或者嵌入不能很好地捕捉数据语义时，检索结果可能会有所不同。人们有时会进行提示工程或者调优来手动解决这些问题，但这样做可能很繁琐。

多查询检索器（MultiQueryRetriever）自动化了提示调优的过程，它使用大语言模型（LLM）从不同角度为给定的用户输入查询生成多个查询。对于每个查询，它都会检索一组相关文档，并取所有查询的唯一并集，以得到一组可能相关的更大的文档集。通过从多个角度生成对同一个问题的视角，多查询检索器可能能够克服基于距离的检索的一些局限性，得到更丰富的结果集。

当然，检索过程也面临一些挑战。通常在将数据录入系统时，你无法知道文档存储系统将面临的具体查询。这意味着与查询最相关的信息可能被埋在大量无关文本的文档中。将完整的文档传递给应用程序可能会导致更昂贵的LLM调用和较差的响应。

上下文压缩（Contextual compression）就是为了解决这个问题。其思想很简单：不是立即原样返回检索到的文档，而是可以使用给定查询的上下文来压缩它们，这样只有相关的信息会被返回。这里的“压缩”既指压缩单个文档的内容，也指整体过滤掉文档。

此外，我们还有自我查询检索器（Self-querying retriever），它能够自我查询。特别地，给定任何自然语言查询，检索器使用查询构造的LLM链来编写结构化查询，然后将该结构化查询应用到其底层的VectorStore。这使得检索器不仅可以使用用户输入的查询来与存储文档的内容进行语义相似性比较，还可以从用户查询中提取有关存储文档元数据的过滤器并执行这些过滤器。

然后是时间加权向量存储检索器（Time-weighted vector store retriever），它使用语义相似性和时间衰减的组合。

最后，我们还有向量存储支持的检索器（Vector store-backed retriever），它是一种使用向量存储来检索文档的检索器。它是围绕Vector Store类的轻量级封装，使其符合检索器接口。它使用由向量存储实现的搜索方法，如相似性搜索和MMR，来查询向量存储中的文本。

以上这些检索器都有各自独特的使用场景和优势，能够有效应对各种复杂的信息检索任务。
# 自查询检索器原理与分类

####    自查询检索器的概念

自查询检索器，顾名思义，是具有自我查询能力的检索器。具体来说，给定任何自然语言查询，检索器使用一个查询构造的大型语言模型（LLM）链来编写结构化查询，然后将该结构化查询应用于其底层的向量存储。这使得检索器不仅可以使用用户输入的查询与存储文档内容进行语义相似性比较，而且还可以从用户查询中提取关于存储文档元数据的过滤器，并执行这些过滤器。

####   自查询检索器的分类

####  1. Chroma自查询检索器

Chroma是一个用于构建嵌入式AI应用程序的数据库。在我们的示例中，我们将展示一个围绕Chroma向量存储的自查询检索器。

####  2. MyScale自查询检索器

MyScale是一个集成向量数据库。你可以通过SQL和LangChain访问你的数据库。MyScale可以利用各种数据类型和函数进行过滤。无论你是扩大数据规模还是将系统扩展到更广泛的应用，MyScale都可以提升你的LLM应用的性能。

####  3. Pinecone自查询检索器

在我们的演示中，我们将展示一个与Pinecone向量存储一起使用的自查询检索器。

####  4. Qdrant自查询检索器

Qdrant是一个向量相似性搜索引擎。它提供了一个生产就绪的服务，具有便捷的API来存储、搜索和管理点 - 带有额外负载的向量。Qdrant针对扩展过滤支持进行了优化，使其更加实用。在我们的示例中，我们将展示一个围绕Qdrant向量存储的自查询检索器。

####  5. Weaviate自查询检索器

创建Weaviate向量存储是首要步骤，我们希望为其添加一些数据。我们创建了一个包含电影摘要的小型演示文档集。

注意：自查询检索器需要你安装lark（使用pip install lark命令进行安装）。我们还需要weaviate-client包。

以上就是自查询检索器的基本概念和分类，希望能对你的学习和理解有所帮助。
# 应用检索器

LangChain提供了广泛的检索器支持，以满足各种不同的需求。以下是LangChain目前支持的检索器的分类：

1. ArxivRetriever：专门用于检索Arxiv这类科学预印本数据库中的文档。

2. AwsKendraIndexRetriever：基于亚马逊的 Kendra 索引服务的检索器。

3. AzureCognitiveSearchRetriever：基于微软Azure Cognitive Search的检索器。

4. ChatGPTPluginRetriever：与ChatGPT插件相关的检索器。

5. ContextualCompressionRetriever：实现上下文压缩功能的检索器。

6. DataberryRetriever：可能是基于Databerry数据管理平台的检索器。

7. ElasticSearchBM25Retriever：基于Elasticsearch和BM25算法的检索器。

8. KNNRetriever：基于最近邻算法的检索器。

9. LlamaIndexGraphRetriever和LlamaIndexRetriever：基于Llama Index图数据库的检索器。

10. MergerRetriever：整合多个检索器结果的检索器。

11. MetalRetriever：可能是针对特定数据集（如金属相关数据）的检索器。

12. MilvusRetriever：基于Milvus向量搜索引擎的检索器。

13. PineconeHybridSearchRetriever：结合Pinecone混合搜索的检索器。

14. PubMedRetriever：专门用于检索PubMed（生物医学文献数据库）的检索器。

15. RemoteLangChainRetriever：远程调用LangChain的检索器。

16. SVMRetriever：基于支持向量机（SVM）算法的检索器。

17. SelfQueryRetriever：能够自我查询的检索器。

18. TFIDFRetriever：基于TF-IDF算法的检索器。

19. TimeWeightedVectorStoreRetriever：引入时间权重的向量存储检索器。

20. VespaRetriever：基于Vespa搜索引擎的检索器。

21. WeaviateHybridSearchRetriever：结合Weaviate混合搜索的检索器。

22. WikipediaRetriever：专门用于检索Wikipedia的检索器。

23. ZepRetriever：可能是与Zep团队合作开发的检索器。

24. ZillizRetriever：基于Zilliz数据分析平台的检索器。

25. DocArrayRetriever：处理文档数组的检索器。

以上检索器的名称和功能可能根据具体实现有所不同，但它们共同的目标都是提供高效准确的信息检索服务。


# 4.1.1 链

在许多人第一次接触LangChain的时候，可能会因为其名字误以为它是区块链相关的内容。然而实际上，LangChain的名字源自其框架的核心设计思路：用最简单的链条，将大预言模型开发的各个组件链接起来。这里的“链”，即我们在复杂系统设计中常说的“链式结构”。

或许你已经注意到了LangChain的logo——一只鹦鹉和一节链条。鹦鹉学舌寓意着大预言模型像鹦鹉一样预测人类的下一段文本，而由无数节链组合起来的链条，象征着通过各种组件的有序连接，形成强大的应用力量。

如果没有链式结构，那么单独的语言大模型，虽然对于简单的应用可能已经足够，但是对于更复杂的应用，我们需要将多个模型或组件进行“链式”结构的连接和组合，这样才能创造出更强大、更具协同性的应用。

例如，我们可以创建一个链，该链接收用户输入，使用PromptTemplate格式化它，然后将格式化的响应传递给大模型。我们可以通过将多个链结合在一起，或者将链与其他组件结合在一起，来构建更复杂的链。

这种链式结构在创新应用中的价值已经得到了验证。最近，Johei Nakajima在Twitter上分享了一篇名为《使用 GPT-4、Pinecone、LangChain 进行多样化应用的任务驱动自主代理》的论文，其中他介绍了最新的Baby AGI。虽然Baby AGI现在还只是概念代码阶段，但是通过这个概念我们可以看出，链式结构是实现创新应用的非常有价值的工具。

下面是最简单的一个链的示例代码：

先安装库：
```
!pip -q install openai langchain
```
设置密钥：
```
import os
os.environ['OPENAI_API_KEY'] = ''
```

LLMChain 是最基本的构建块链。它接收一个提示模板，使用用户输入进行格式化，然后返回LLM的响应。

```
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

llm = OpenAI(temperature=0.9)
prompt = PromptTemplate(
    input_variables=["product"],
    template="What is a good name for a company that makes {product}?",
)
```
要使用LLMChain，首先创建一个提示模板。我们现在可以创建一个非常简单的链，它会接收用户输入，使用输入格式化提示，然后将其发送到LLM。

```
from langchain.chains import LLMChain
chain = LLMChain(llm=llm, prompt=prompt)

# Run the chain only specifying the input variable.
print(chain.run("colorful socks"))
```

就如我们在模型输入输出所说, 如果有通用语言模型的方法，那么Langchain一定有聊天模型的方法。你也可以在LLMChain中使用聊天模型：

```
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
human_message_prompt = HumanMessagePromptTemplate(
        prompt=PromptTemplate(
            template="What is a good name for a company that makes {product}?",
            input_variables=["product"],
        )
    )
chat_prompt_template = ChatPromptTemplate.from_messages([human_message_prompt])
chat = ChatOpenAI(temperature=0.9)
chain = LLMChain(llm=chat, prompt=chat_prompt_template)
print(chain.run("colorful socks"))
```


# 基础链类型

链的类型分为四种，包括LLMChain、RouterChain、顺序链（Sequential Chains）和转换链（Transformation Chain）。

LLMChain是一种简单的链，它在语言模型周围增加了一些功能。它在LangChain中被广泛应用，包括在其他链和代理中。LLMChain由提示模板和语言模型（可以是LLM或聊天模型）组成。它使用提供的输入键值（如果有，还有记忆键值）格式化提示模板，将格式化的字符串传递给LLM，并返回LLM的输出。

RouterChain是一种使用路由器链创建的链，它可以动态地选择给定输入的下一条链。路由器链由两部分组成：路由器链本身（负责选择要调用的下一条链）和目标链（路由器链可以路由到的链）。

顺序链（Sequential Chains）是在调用语言模型后的下一步，特别是当你希望将一次调用的输出作为另一次调用的输入时。顺序链允许你连接多个链并将它们组成执行特定场景的流水线。顺序链有两种类型：SimpleSequentialChain（最简单形式的顺序链，其中每一步都有一个单一的输入/输出，一个步骤的输出是下一个步骤的输入）和SequentialChain（一种更通用的顺序链，允许多个输入/输出）。

转换链（Transformation Chain）是一种使用通用转换链的方法。作为一个示例，我们将创建一个虚构的转换，它接收一个超长的文本，过滤文本以仅显示前三段，然后将其传递给LLMChain进行总结。
# 工具链的理解与应用

在Langchain 中，"链"的概念是最经常使用的。这些"链"其实就是由一系列工具链构成的，每一个工具都可以视为整个链中的一个环节。这些环节可能非常简单，例如将一个提示模板和一个大型语言模型链接起来，形成一个大型语言模型链（LLMChains）。然而，也可能更加复杂，例如在整个流程中，通过多个环节进行多个步骤的链接。这可能还包括多个大型语言模型以及各种不同的实用工具等。在工具链中，一个链的输出将成为下一个链的输入，这就形成了一个输入输出的链式流程。例如，你可能会从大型语言模型的输出中提取某些内容，将其作为Wolfram Alpha查询的输入，然后将查询结果带回，并再次通过大型模型生成将返回给用户的响应。这就是一个典型的工具链的示例。

####   常见工具链的功能与应用

在实际的应用中，一些常见的工具链如APIChain、ConversationalRetrievalQA等已经被封装好了。

APIChain使得大型语言模型可以与API进行交互，以获取相关的信息。构建该链时，需要提供一个与所提供的API文档相关的问题。

ConversationalRetrievalQA链在检索问答链的基础上提供了一个聊天历史组件。它首先将聊天历史（要么明确传入，要么从提供的内存中检索）和问题合并成一个独立的问题，然后从检索器中查找相关的文档，最后将这些文档和问题传递给一个问答链，以返回响应。

对于需要对多个文档进行文档合并的任务，我们可以使用文档合并链，如MapReduceDocumentsChain或StuffDocumentsChain等。

对于需要从同一段落中提取多个实体及其属性的任务，我们可以使用提取链。

还有一些专门设计用来满足特定需求的链，如ConstitutionalChain，这是一个保证大型语言模型输出遵循一定宪法原则的链，通过设定特定的规则和指导方针，使得生成的内容符合这些原则，从而提供更受控、符合伦理和上下文适当的回应。

####   工具链的使用方法

这些工具链的使用方法通常是先使用类方法实例化，然后通过run方法调用，输出结果是一个字符串，然后将这个字符串传递给下一个链。类方法通常以"from"和下划线开始，比较常见的有from_llm()和from_chain_type()，他们都接受外部的数据来源作为参数。

下面以SQLDatabaseChain为例子，看看如何使用它。SQLDatabaseChain就是一个通过from_llm()方法实例化的链，它用于回答SQL数据库上的问题。

```
from langchain import OpenAI, SQLDatabase, SQLDatabaseChain

db = SQLDatabase.from_uri("sqlite:///../../../../notebooks/Chinook.db")
llm = OpenAI(temperature=0, verbose=True)

db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True)

db_chain.run("How many employees are there?")
```
运行的结果是：

```
    
    
    > Entering new SQLDatabaseChain chain...
    How many employees are there?
    SQLQuery:

    /workspace/langchain/langchain/sql_database.py:191: SAWarning: Dialect sqlite+pysqlite does *not* support Decimal objects natively, and SQLAlchemy must convert from floating point - rounding errors and other issues may occur. Please consider storing Decimal numbers as strings or integers on this platform for lossless storage.
      sample_rows = connection.execute(command)


    SELECT COUNT(*) FROM "Employee";
    SQLResult: [(8,)]
    Answer:There are 8 employees.
    > Finished chain.





    'There are 8 employees.'

```
# 链的使用

####  异步支持

LangChain通过利用asyncio库为链（Chain）提供了异步支持。

目前在LLMChain（通过arun, apredict, acall）、LLMMathChain（通过arun和acall）、ChatVectorDBChain以及QA链中支持异步方法。其他链的异步支持正在规划中。

####  使用方法

所有的链都可以像函数一样被调用。当链对象只有一个输出键（也就是说，它的`output_keys`中只有一个元素）的时候，我们预期的结果只需要一个字符串，可以使用`run`方法。

在LangChain中，所有继承自`Chain`类的对象，提供了一些用于执行链逻辑的方式。其中一种比较直接的方式就是使用`__call__`方法。`__call__` 方法是 `Chain` 类的一个方法，它让 `Chain` 类的实例可以像函数一样被调用，比如 `result = chain(inputs, return_only_outputs=True)`就完成了调用链。

先看看`__call__`方法的定义：
```python  
    def __call__(
        self,
        inputs: Union[Dict[str, Any], Any],
        return_only_outputs: bool = False,
        callbacks: Callbacks = None,
        *,
        tags: Optional[List[str]] = None,
        include_run_info: bool = False,
    ) -> Dict[str, Any]:
```  

这个 `__call__` 方法接收的参数，最有用的是以下三个：

- `inputs`：这个参数是要传递给链的输入。它的类型是 `Any`，这意味着可以接收任何类型的输入。
- `return_only_outputs`：这个参数是一个布尔值，如果设为 `True`，则只返回输出结果。如果设为 `False`，则可能返回其他额外的信息。
- `callbacks`：这个参数是回调函数的列表，它们将在链执行过程中的某些时刻被调用。

`__call__` 方法返回一个字典，这个字典包含了链执行的结果和可能的其他信息。

在 Python 中，如果一个类定义了 `__call__` 方法，那么这个类的实例就可以像函数一样被调用。例如，如果 `chain` 是 `Chain` 类的一个实例，那么你可以像调用函数一样调用 `chain`：

```python
result = chain(inputs, return_only_outputs=True)
```

在这个调用中，`inputs` 是要传递给链的输入，`return_only_outputs=True` 表示只返回输出结果。返回的 `result` 是一个字典，包含了链执行的结果。


使用的时候，最重要的参数是 `inputs`:
```python
chat = ChatOpenAI(temperature=0)
prompt_template = "Tell me a {adjective} joke"
llm_chain = LLMChain(llm=chat, prompt=PromptTemplate.from_template(prompt_template))

llm_chain(inputs={"adjective": "corny"})
```
返回的结果是：

```
    {'adjective': 'corny',
     'text': 'Why did the tomato turn red? Because it saw the salad dressing!'}
```
你可以通过设置return_only_outputs为True来配置它只返回输出键值。

```
llm_chain("corny", return_only_outputs=True)
```

返回的结果就不包含 `"adjective": "corny"`：

```
    {'text': 'Why did the tomato turn red? Because it saw the salad dressing!'}
```
然而，当链对象只有一个输出键（也就是说，它的`output_keys`中只有一个元素）的时候，我们可以使用`run`方法。

```
# llm_chain only has one output key, so we can use run
llm_chain.output_keys
```

```
    ['text']
```
`output_keys`中只有一个元素 `['text']`，我们可以`run`方法：

```
llm_chain.run({"adjective": "corny"})
```

如果输入的键值只有一个，预期的输出也是一个字符串，那么输入可以是字符串也可以是对象，可以使用`run`方法也可以使用`__call__`方法。

`run`方法将整个链的输入键值（input key values）进行处理，并返回处理后的结果。需要注意的是，与`__call__`方法可能返回字典形式的结果不同，`run`方法总是返回一个字符串。这也是为什么当链对象只有一个输出键的时候，我们倾向于使用`run`方法，因为这时候处理结果自然只有一个，返回字符串形式更直观也更便于处理。

例如，假设我们有一个链对象，它的任务是根据输入的文本生成摘要，那么在调用`run`方法的时候，我们可以直接将待摘要的文本作为参数输入，然后得到摘要后的文本。在这种情况下，你可以直接输入字符串，而无需指定输入映射。

另外，你可以很容易地将一个`Chain`对象作为一个工具，通过它的`run`方法集成到你的Agent中，这样可以将链的处理能力直接用于你的Agent逻辑中。


####  支持自定义链

你可以子类化Chain并实现你自己的自定义链。从其输出中仅仅调试链对象可能会比较困难，因为大多数链对象涉及到相当多的输入提示预处理和LLM输出后处理。

####  链的调试

将verbose设置为True将会在运行链对象时打印出一些链对象的内部状态。

```python
conversation = ConversationChain(
    llm=chat,
    memory=ConversationBufferMemory(),
    verbose=True
)
conversation.run("What is ChatGPT?")
```

####  加记忆的链
链可以使用Memory对象进行初始化，这将使得在调用链时数据持久化，使得链具有状态。

```python
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

conversation = ConversationChain(
    llm=chat,
    memory=ConversationBufferMemory()
)

conversation.run("Answer briefly. What are the first 3 colors of a rainbow?")
# -> The first three colors of a rainbow are red, orange, and yellow.
conversation.run("And the next 4?")
# -> The next four colors of a rainbow are green, blue, indigo, and violet.
```


####  链序列化

我们使用的序列化格式是json或yaml。目前，只有一些链支持这种类型的序列化。我们将随着时间的推移增加支持的链的数量。首先，让我们看看如何将链保存到磁盘。这可以通过.save方法完成，并指定一个带有json或yaml扩展名的文件路径。我们可以使用load_chain方法从磁盘加载链。
# 4.2.1 LLM 链

LLMChain 将是一个非常简单的 Chains 。这绝对是您最常见到的 Chains 。基本上只是将一个大型语言模型与提示（Prompt）链在一起。然后使用提示模板来提供输入,并将一些内容输入到其中。

以下是文章的事实提取场景下，使用通用LLM链的示例代码：

安装库:
```
!pip -q install openai langchain huggingface_hub
```

设置密钥:
```
import os

os.environ['OPENAI_API_KEY'] = ''
os.environ['HUGGINGFACEHUB_API_TOKEN'] = ''
```

```
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain

```

在这里，您可以看到我正在设置OpenAI `text-davinci-003` 模型，我们将将温度设置为零，只需设置最大令牌。如果您知道这是一个默认的标准模型的话，你肯定知道其中许多将根据默认值进行设置。

```
llm = OpenAI(model_name='text-davinci-003', 
             temperature=0, 
             max_tokens = 256)
```

我在这里有一篇小文章，所以我将要做的实际上是事实提取。因此，在这里，我基本上提取了一篇关于Coinbase的文章。所以这是一篇相当长的文章，如果我们看一下，它有3500个字符。

我们要做的就是从中提取出关键事实。然后我们将对此进行调整。并尝试将这些事实改写成一种新的内容。首先，我们需要我们的Prompts（提示）模板，所以我们的Prompts（提示）模板基本上接受输入，这是我们在这里得到的。

```
article = '''Coinbase, the second-largest crypto exchange by trading volume, released its Q4 2022 earnings on Tuesday, giving shareholders and market players alike an updated look into its financials. In response to the report, the company's shares are down modestly in early after-hours trading.In the fourth quarter of 2022, Coinbase generated $605 million in total revenue, down sharply from $2.49 billion in the year-ago quarter. Coinbase's top line was not enough to cover its expenses: The company lost $557 million in the three-month period on a GAAP basis (net income) worth -$2.46 per share, and an adjusted EBITDA deficit of $124 million.Wall Street expected Coinbase to report $581.2 million in revenue and earnings per share of -$2.44 with adjusted EBITDA of -$201.8 million driven by 8.4 million monthly transaction users (MTUs), according to data provided by Yahoo Finance.Before its Q4 earnings were released, Coinbase's stock had risen 86% year-to-date. Even with that rally, the value of Coinbase when measured on a per-share basis is still down significantly from its 52-week high of $206.79.That Coinbase beat revenue expectations is notable in that it came with declines in trading volume; Coinbase historically generated the bulk of its revenues from trading fees, making Q4 2022 notable. Consumer trading volumes fell from $26 billion in the third quarter of last year to $20 billion in Q4, while institutional volumes across the same timeframe fell from $133 billion to $125 billion.The overall crypto market capitalization fell about 64%, or $1.5 trillion during 2022, which resulted in Coinbase's total trading volumes and transaction revenues to fall 50% and 66% year-over-year, respectively, the company reported.As you would expect with declines in trading volume, trading revenue at Coinbase fell in Q4 compared to the third quarter of last year, dipping from $365.9 million to $322.1 million. (TechCrunch is comparing Coinbase's Q4 2022 results to Q3 2022 instead of Q4 2021, as the latter comparison would be less useful given how much the crypto market has changed in the last year; we're all aware that overall crypto activity has fallen from the final months of 2021.)There were bits of good news in the Coinbase report. While Coinbase's trading revenues were less than exuberant, the company's other revenues posted gains. What Coinbase calls its "subscription and services revenue" rose from $210.5 million in Q3 2022 to $282.8 million in Q4 of the same year, a gain of just over 34% in a single quarter.And even as the crypto industry faced a number of catastrophic events, including the Terra/LUNA and FTX collapses to name a few, there was still growth in other areas. The monthly active developers in crypto have more than doubled since 2020 to over 20,000, while major brands like Starbucks, Nike and Adidas have dived into the space alongside social media platforms like Instagram and Reddit.With big players getting into crypto, industry players are hoping this move results in greater adoption both for product use cases and trading volumes. Although there was a lot of movement from traditional retail markets and Web 2.0 businesses, trading volume for both consumer and institutional users fell quarter-over-quarter for Coinbase.Looking forward, it'll be interesting to see if these pieces pick back up and trading interest reemerges in 2023, or if platforms like Coinbase will have to keep looking elsewhere for revenue (like its subscription service) if users continue to shy away from the market.
'''
```

然后我们需要实际的Prompts（提示），这里的Prompts（提示）是从这段文本中提取关键事实，不包括观点，给每个事实编号，并保持它们的句子简短。

```
fact_extraction_prompt = PromptTemplate(
    input_variables=["text_input"],
    template="Extract the key facts out of this text. Don't include opinions. Give each fact a number and keep them short sentences. :\n\n {text_input}"
)
```
然后我将基本上将输入设置为这个文本输入。好吧，制作 Chains 实际上非常简单，我们只需说我们将使用 LLMChain ，我们传入 LLMChain。然后我们传入我们将要使用的提示模板，所以这里我有事实提取提示。然后我们将其传入。然后我们可以运行它。

```
fact_extraction_chain = LLMChain(llm=llm, prompt=fact_extraction_prompt)

facts = fact_extraction_chain.run(article)

print(facts)
```
您可以看到，在运行它之后，确实发生了变化。

```
1. Coinbase released its Q4 2022 earnings on Tuesday.
2. Coinbase generated $605 million in total revenue in Q4 2022.
3. Coinbase lost $557 million in the three-month period on a GAAP basis.
4. Coinbase's stock had risen 86% year-to-date before its Q4 earnings were released.
5. Consumer trading volumes fell from $26 billion in Q3 2022 to $20 billion in Q4 2022.
6. Institutional volumes across the same timeframe fell from $133 billion to $125 billion.
7. The overall crypto market capitalization fell about 64%, or $1.5 trillion during 2022.
8. Trading revenue at Coinbase fell from $365.9 million in Q3 2022 to $322.1 million in Q4 2022.
9. Coinbase's "subscription and services revenue" rose from $210.5 million in Q3 2022 to $282.8 million in Q4 2022.
10. Monthly active developers in crypto have more than doubled since 2020 to over 20,000.
```

它很好地从我们的文章中获取了事实。它做得相当不错，我们从这篇文章中得到了10个事实。

# 4.2.2 顺序链

另一种通用链是顺序链 (Sequential)，它基本上是多个链(Chain)的组合。您会发现，虽然链由单个工具组成，但它们也可以由其他链组成，这些链会连接在一起。工具链是许多神奇事情发生的地方。有各种不同的工具链，这里只是其中一部分，而且随着时间的推移，预计还会添加更多。

我们在上一节制作了一个提取新闻10个事实的LLMChain, 为了让你理解顺序链是如何工作的，我们先加一个新的链。

现在我们要做的是制作一个新的。然后我们将把其中一些内容链在一起，所以下一个我要做的也是一个LLMChain。

这将采用上一节的10个事实。但我们将把它们改写成投资者报告的形式，所以您可以在这里看到，我们将说好了，你是高盛的分析师，接受以下事实列表，并用它们为投资者撰写一个简短的段落，不要遗漏关键信息。我们也可以放一些东西在这里，也不要杜撰信息，但这是我们要传入的事实。

```
investor_update_prompt = PromptTemplate(
    input_variables=["facts"],
    template="You are a Goldman Sachs analyst. Take the following list of facts and use them to write a short paragrah for investors. Don't leave out key info:\n\n {facts}"
)
```

再次强调，这是一个LLMChain ，我们传入LLM，我们仍然使用上面定义的原始模型，我们传入提示模板。然后我们可以运行它。

```
investor_update_chain = LLMChain(llm=llm, prompt=investor_update_prompt)

investor_update = investor_update_chain.run(facts)

print(investor_update)
len(investor_update)
```

您可以看到，确实回来了。

```
Coinbase released its Q4 2022 earnings on Tuesday, revealing total revenue of $605 million and a GAAP loss of $557 million. Despite the losses, Coinbase's stock had risen 86% year-to-date before its Q4 earnings were released. Consumer trading volumes fell from $26 billion in Q3 2022 to $20 billion in Q4 2022, while institutional volumes fell from $133 billion to $125 billion. The overall crypto market capitalization fell about 64%, or $1.5 trillion during 2022. Trading revenue at Coinbase fell from $365.9 million in Q3 2022 to $322.1 million in Q4 2022, while its "subscription and services revenue" rose from $210.5 million in Q3 2022 to $282.8 million in Q4 2022. Despite the market downturn, monthly active developers in crypto have more than doubled since 2020 to over 20,000.
788
```

它写了一篇相当连贯的好文章。它比之前的要短得多。

####  加入顺序链

我们将使用简单的顺序链（SimpleSequentialChain）来完成这个过程。简单的顺序链就像PyTorch中的标准顺序模型一样，你只是从A到B到C，没有做任何复杂的操作。

我们还有这样的想法，一个链的输出将成为下一个链的输入。你可以看到，我们已经设置了完整的链，我们将拥有我们的提取事实链。而且我们将在这里拥有我们的投资摘要链。

现在当我取出原始文章并运行它时，它将完成这两个操作。你可以看到，现在它已经完成了事实提取。现在它已经完成了重写。然后它在这里完成了链的操作。这是我们将这些事物链在一起的一种方式，而不是必须重写代码来完成一件事，然后再去做另一件事等等。

```
from langchain.chains import SimpleSequentialChain, SequentialChain

full_chain = SimpleSequentialChain(chains=[fact_extraction_chain, investor_update_chain], verbose=True)
```

我们确实可以看到，如果我们取出我们得到的响应，我们得到了投资摘要链的结果，即使我们传入的是原始文章。

```
response = full_chain.run(article)
```

迄今为止，你将看到的最常见的变化是由一个大型语言模型组成的。还有一个提示模板，还有一堆其他很酷的链和工具在 Langchain 中。
# 合并文档链概述

在许多应用场景中，我们需要与文档进行交互，如阅读说明书、浏览产品手册等等。近来，基于这些场景开发的应用，如chatDOC和chatPDF，都受到了广大用户的欢迎。为了满足对特定文档进行问题回答、提取摘要等需求，Langchain设定了几种合并文档链类型。

这些核心链都是为处理文档而设计的。它们在对文档进行概括、回答文档问题、从文档中提取信息等方面非常有用。

但是文档链的类型给初学者造成了很大的困扰。主要是因为我们通常不清楚在指定了这些类型后，中间的处理流程发生了什么。如果我们能从各个类型的具体步骤进行理解，就会发现，这些类型的主要区别在于它们处理输入文档的方式，以及在中间过程中与模型的交互次数和答案来源于哪些阶段。理解了这些，我们就可以更清楚地认识到各种类型的优缺点，从而在生产环境中做出更好的决策。

换句话说，一旦我们理解了每个类型的具体步骤提交了什么提示(Prompt)，提示从何而来就可以明确知道使用哪种类型更符合我们的需求。我们会在后面对每个类型经历的具体步骤进行拆解。在这里我们先做个概述，没看懂可移步相应的文档类型的小节。

“Stuff链” 是处理文档链中最直接的一个。它接收一组文档，将它们全部插入到一个提示中，然后将该提示传递给LLM。这种链适合于文档较小且大部分调用只传入少量文档的应用。

“精化（Refine）”通过遍历输入文档并迭代更新其答案来构建响应。对于每个文档，它将所有非文档输入、当前文档和最新的中间答案传递给LLM链，以获得新的答案。

由于精化链一次只向LLM传递一个文档，因此它非常适合需要分析比模型上下文能容纳更多的文档的任务。但显然，这种链会比如Stuff链这样的链调用更多的LLM。此外，还有一些任务很难通过迭代来完成。例如，当文档经常相互交叉引用或任务需要许多文档的详细信息时，精化链的表现可能较差。

“Map Reduce”首先将LLM链单独应用于每个文档（Map步骤），并将链输出视为新的文档。然后，它将所有新文档传递给一个单独的“Combine Documents Chain”，以获得单一的输出（Reduce步骤）。它可以选择首先压缩或合并映射的文档，以确保它们适合“Combine Documents Chain”（这将经常将它们传递给LLM）。如果需要，这个压缩步骤将递归地执行。

“重排链（Map Re-rank）”对每个文档运行初始提示，不仅试图完成任务，还对其答案的确定程度给出评分。得分最高的响应将被返回。


# 4.3.1 Stuff链

在大数据和自然语言处理领域，各种不同的处理链方式可以用来优化信息检索和生成答案。本文将解析 'Stuff' 类型的处理链，并说明其如何通过改变输入的组织和输出的生成方式来提高文档搜索的质量。

#### ##4    1. 整体流程

'Stuff' 文档处理链是一种直接的处理方式。它接收一组文档，将所有文档插入到一个提示中，然后将该提示传递给语言模型 (LLM)。

#### ##4    2. 插入文档阶段

在这个阶段，系统接收一组文档，将它们全部插入到一个提示中。提示(Prompt) 是全部文档内容。这种方式适用于文档较小且大部分调用只传入少量文档的应用。它可以简单地将所有文档拼接在一起，形成一个大的提示，然后将这个提示传递给 LLM。

#### ##4    3. 生成答案阶段

在这个阶段，系统将包含所有文档的提示传递给 LLM。LLM 根据这个提示生成答案。由于所有的文档都被包含在同一个提示中，所以 LLM 生成的答案会考虑到所有的文档。

#### ##4    4. 最终实现效果

通过 'Stuff' 文档处理链，系统可以对包含多个文档的问题生成一个全面的答案。这种处理方式可以提高文档搜索的质量，特别是在处理小文档和少量文档的情况下。

#### ##4    5. 适用场景

由于 'Stuff' 类型的处理链方式主要适用于处理小文档和少量文档的情况，所以它特别适用于那些大部分调用只传入少量文档的应用。然而，对于需要处理大量文档或者文档较大的情况，可能需要使用其他类型的处理链，如 'Refine' 或 'MapReduce'。

总的来说，通过使用 'Stuff' 文档处理链，系统可以有效地处理小文档和少量文档的情况，从而提高文档搜索的质量。

# 4.3.2 精化链

本文将解析 'Refine' 类型的处理链，并说明其如何通过改变输入的组织和输出的生成方式来提高文档搜索的质量。

#### ##4    1. 整体流程

'Refine' 文档处理链通过遍历输入文档并迭代更新其答案来构建响应。对于每个文档，它将所有非文档输入（例如用户的问题或其他与当前文档相关的信息）、当前文档和最新的中间答案传递给语言模型 (LLM)，以获得新的答案。

#### ##4    2. 遍历文档阶段

在这个阶段，系统会遍历输入的所有文档。对于每个文档，一起作为提示(Prompt)传递给 LLM 的内容有：

- 一些上下文信息，例如用户的问题或其他与当前文档相关的信息。
- 最新的中间答案。中间答案是系统在处理之前的文档时产生的。一开始，中间答案可能是空的，但随着系统处理更多的文档，中间答案会不断更新。
- 当前文档。

与Map reduce 链和重排链不同的是它不产生新文档，只不断更新的是提示，迭代出更全面的答案。而且文档之间的影响是传递性的，上一个文档形成的答案会影响下一个文档的答案。

#### ##4    3. 更新答案阶段

在这个阶段，系统将提示传递给 LLM，然后将 LLM 生成的答案作为新的中间答案。这个过程会迭代进行，直到所有的文档都被处理。

#### ##4    4. 最终实现效果

通过 'Refine' 文档处理链，系统可以对包含多个文档的问题生成一个全面的答案，而且每个文档的处理结果都会影响后续文档的处理。这种处理方式可以提高文档搜索的质量，特别是在处理大量文档的情况下。

#### ##4    5. 适用场景

'Refine' 类型的处理链方式主要适用于处理大量文档的情况，特别是当这些文档不能全部放入模型的上下文中时。然而，这种处理方式可能会使用更多的计算资源，并且在处理某些复杂任务（如文档之间频繁地交叉引用，或者需要从许多文档中获取详细信息）时可能表现不佳。

总的来说，通过使用 'Refine' 文档处理链，系统可以有效地处理大量文档的情况，从而提高文档搜索的质量。然而，这种处理方式可能需要更多的计算资源，并且可能在处理复杂任务时表现不佳。
# 4.3.3 Map reduce 链

在大数据和自然语言处理领域，各种不同的处理链方式可以用来优化信息检索和生成答案。本文将解析 'MapReduce' 类型的处理链，并说明其如何通过改变输入的组织和输出的生成方式来提高文档搜索的质量。

#### ##4    1. 整体流程

'MapReduce' 文档处理链主要由两个部分组成：映射（Map）阶段和归约（Reduce）阶段。在映射阶段，系统对每个文档单独应用一个语言模型（LLM）链，并将链输出视为新的文档。在归约阶段，系统将所有新文档传递给一个单独的合并文档链，以获得单一的输出。如果需要，系统会首先压缩或合并映射的文档，以确保它们适合合并文档链。

#### ##4    2. 映射阶段（Map Stage）

在映射阶段，系统使用 LLM 链对每个输入的文档进行处理。处理的方式是，将当前文档作为输入传递给 LLM 链，然后将 LLM 链的输出视为新的文档。这样，每个文档都会被转化为一个新的文档，这个新文档包含了原始文档的处理结果。

对于每个文档，作为提示(Prompt)传递给 LLM 的内容是原始文档。比起“ Stuff ”类型多了预处理。

每个原始文档都经过LLM 链处理的结果写入一个新文档，这就是映射的过程。比如原文档有2000字，经过LLM 链处理的结果是200字。200字的结果存储为一个新文档，但是跟2000字原文档存着映射关系。

#### ##4    3. 归约阶段（Reduce Stage）

在归约阶段，系统使用合并文档链将映射阶段得到的所有新文档合并成一个。如果新文档的总长度超过了合并文档链的容量，那么系统会使用一个压缩过程将新文档的数量减少到合适的数量。这个压缩过程会递归进行，直到新文档的总长度满足要求。

#### ##4    4. 最终实现效果

通过 'MapReduce' 文档处理链，系统可以对每个文档单独进行处理，然后将所有文档的处理结果合并在一起。这种处理方式可以提高文档搜索的质量，特别是在处理大量文档的情况下。

#### ##4    5. 适用场景

'MapReduce' 类型的处理链方式主要适用于处理大量文档的情况，特别是当这些文档不能全部放入模型的上下文中时。通过并行处理每个文档并合并处理结果，这种处理方式可以在有限的资源下处理大量的文档。然而，这种处理方式可能会使用更多的计算资源，并且可能在处理某些复杂任务（如文档之间频繁地交叉引用，或者需要从许多文档中获取详细信息）时可能表现不佳。

总的来说，通过使用 'MapReduce' 文档处理链，系统可以有效地处理大量文档的情况，从而提高文档搜索的质量。然而，这种处理方式可能需要更多的计算资源，并且可能在处理复杂任务时表现不佳。
# 4.3.4 重排链

标题：源码解析：'Map Re-rank' 类型在文档处理链中的应用

在大数据和自然语言处理领域，各种不同的处理链方式可以用来优化信息检索和生成答案。本文将解析 'Map Re-rank' 类型的处理链，并说明其如何通过改变输入的组织和输出的生成方式来提高文档搜索的质量。

#### ##4    1. 整体流程

'Map Re-rank' 文档处理链对每个文档运行初始提示，这个提示不仅试图完成任务，还对其答案的确定程度给出评分。最后，得分最高的响应将被返回。

#### ##4    2. 映射和评分阶段

在这个阶段，系统对每个文档运行初始提示。每个文档都会被独立地处理，处理的方式是，系统不仅试图完成任务，还对其答案的确定程度给出评分。这样，每个文档都会被转化为一个新的文档，这个新文档包含了原始文档的处理结果和评分。

对于每个文档，作为提示(Prompt)传递给 LLM 的内容是原始文档, 但是提示模板增加了评分规则。拿到LLM链的答案后，存储为一个新文档，与原文档形成映射关系。

#### ##4    3. 重排阶段（Re-rank Stage）

在这个阶段，系统根据每个新文档的评分进行重排。具体来说，系统会选择得分最高的新文档，并将其作为最终的输出。

只有这个类型有自动重排的机制，因为只有这个类型，对原始文档进行处理的时候，添加了评分规则的提示。

#### ##4    4. 最终实现效果

通过 'Map Re-rank' 文档处理链，系统可以对每个文档独立地进行处理和评分，然后选择得分最高的结果作为最终输出。这种处理方式可以提高文档搜索的质量，特别是在处理大量文档的情况下。

#### ##4    5. 适用场景

'Map Re-rank' 类型的处理链方式主要适用于处理大量文档的情况，特别是当需要从多个可能的答案中选择最优答案时。通过对每个文档的处理结果进行评分和重排，这种处理方式可以在有限的资源下找到最优的答案。然而，这种处理方式可能会使用更多的计算资源，并且可能在处理某些复杂任务（如文档之间频繁地交叉引用，或者需要从许多文档中获取详细信息）时可能表现不佳。

总的来说，通过使用 'Map Re-rank' 文档处理链，系统可以有效地处理大量文档的情况，并从多个可能的答案中选择最优答案，从而提高文档搜索的质量。然而，这种处理方式可能需要更多的计算资源，并且可能在处理复杂任务时表现不佳。
# API 工具链

另一个非常有用的工具链的例子是API工具链，所以在这里我只是向你展示了用于天气信息的一个例子。我们设置了要使用的API，这将根据这些文档编写API调用。这就是这个调用将输出的内容。然后这将使用该调用查询API并返回结果。
```
from langchain import OpenAI
from langchain.chains.api.prompt import API_RESPONSE_PROMPT

from langchain.chains import APIChain
from langchain.prompts.prompt import PromptTemplate
```

```
llm = OpenAI(temperature=0,
             max_tokens=100)
```             
显然，它只能回答API能够给你的内容，这里它基本上是在给我们返回这个并且它告诉我们，是的，这个JSON响应中的一些事情表明正在下雨，但是需要注意的是，通常文档加上URL加上JSON会超过大型语言模型可以处理的标记数，因此如果您在达芬奇模型上使用超过4000个标记，可能会出现错误。

```
from langchain.chains.api import open_meteo_docs
chain_new = APIChain.from_llm_and_api_docs(llm, 
                                           open_meteo_docs.OPEN_METEO_DOCS, 
                                           verbose=True)
```

```
chain_new.run('What is the temperature like right now in Bedok, Singapore in degrees Celcius?')
```

这里你可以看到，我问它在新加坡 Bedok 的温度是多少摄氏度。它写下了这个用于查询的URL。

```
    > Entering new APIChain chain...
https://api.open-meteo.com/v1/forecast?latitude=1.3&longitude=103.9&hourly=temperature_2m&current_weather=true&temperature_unit=celsius
{"latitude":1.375,"longitude":103.875,"generationtime_ms":0.38802623748779297,"utc_offset_seconds":0,"timezone":"GMT","timezone_abbreviation":"GMT","elevation":6.0,"current_weather":{"temperature":26.1,"windspeed":10.5,"winddirection":16.0,"weathercode":3,"time":"2023-02-22T14:00"},"hourly_units":{"time":"iso8601","temperature_2m":"°C"},"hourly":{"time":["2023-02-22T00:00","2023-02-22T01:00","2023-02-22T02:00","2023-02-22T03:00","2023-02-22T04:00",27.6,27.5,27.2,26.8,26.4,26.1,25.7,25.5,25.4,25.3,25.2,25.1,25.0,24.9,24.9,24.9,24.9,24.9]}}

    > Finished chain.
 The temperature right now in Bedok, Singapore is 26.1 degrees Celcius.
```  
它给我们返回了当前的温度和位置。

另外要考虑的一件事是，这是相当昂贵的，如果我们每千个标记支付两美分我们刚刚输入了4000个标记，只是为了获取天气或其他东西，这并不总是最高效的方法，但它确实显示了 LangChain 可以做这些事情。您可以编写一些代码来调用您想要的API调用。
# PALChain

下一个我要向你展示的是PAL Math Chain，这其实是使用了一个不同的大型语言模型。

我们基本上要做的是，当我们遇到某种数字问题时，我们将使用它。这是 Langchain 文档中的一个例子，Jan有宠物的数量是三倍。这是一个数学方程问题了。

####  为什么用 PALChain ？

我们基本上在这里Prompts（提示）模型这样做的方式是将这个文字陈述转化为一个小型的Python函数，然后计算数学问题，而不是仅仅依靠语言模型猜测。

来看一个有趣的例子。这是非常简单的数学问题，食堂有23个苹果，如果他们用了20个来吃午饭，又买了6个，那么他们现在还剩下多少个苹果。

问题在于，如果你使用的是大型语言模型，它们可能会得到正确的答案，但如果你使用的是一些较小的模型，甚至只是像T5模型一样，大多数T5模型都会得到这样的错误答案，而不是依赖其中一个模型来做这个，我们可以使用这个方法，它基本上是获取这些数据并进行重写。

```
from langchain.chains import PALChain
pal_chain = PALChain.from_math_prompt(llm, verbose=True)

question = "Jan has three times the number of pets as Marcia. Marcia has two more pets than Cindy. If Cindy has four pets, how many total pets do the three have?"
```

```
pal_chain.run(question)
```
你可以看到它写了一个Python函数，它使用了文档字符串将我们之前的内容放在这里，我们从苹果的初始开始，所以它只是将这些变量赋值。然后苹果剩下的数量等于初始苹果减去使用的苹果加上购买的苹果，它确实给了我们准确的结果。

```
    > Entering new PALChain chain...
def solution():
    """The cafeteria had 23 apples. If they used 20 for lunch and bought 6 more, how many apples do they have?"""
    apples_initial = 23
    apples_used = 20
    apples_bought = 6
    apples_left = apples_initial - apples_used + apples_bought
    result = apples_left
    return result

    > Finished chain.
9
```
然后我们可以将输出带入另一个大型语言模型中，然后以对话的方式重新表达它，这样它可以告诉你苹果剩下的数量是多少，或者我们可以直接从这个模块中获取输出。



# 5.1 记忆组件概述

想一想，我们为什么需要记忆 ？

大型语言模型本质上是无记忆的。当我们与其交互时，它仅根据提供的提示生成相应的输出，而无法存储或记住过去的交互内容。这一特性，使得大型语言模型在实现聊天机器人或聊天代理时，难以满足人们的期望。

人们期待聊天机器人具有人的品质和回应能力。当他们意识到机器人只是进行一次性的调用和响应，而无法记住以往的交流内容时，可能会感到沮丧。在现实的聊天环境中，人们的对话充满了缩写和含蓄表达，他们会引用过去的对话内容，并期待对方能够理解和回应。例如，如果聊天历史中只在一开始提到了某人的名字，随后仅用代词指代，那么他们就期望聊天机器人能够理解和记住这个指代关系。

我们期待的是聊天机器人能够跨越整个对话，理解和记住我们的交流内容。但是，要实现这个目标，我们需要赋予大型语言模型一些“记忆”。

####   LangChain 记忆组件

为此，LangChain 提供了两种记忆组件。首先，LangChain 提供了用于管理和操作之前聊天消息的辅助工具。这些工具设计得模块化，可以灵活地适应各种使用场景。其次，LangChain 提供了简单的方式将这些工具整合到模型中。

这种记忆机制的引入，使得大型语言模型能够“回顾”以往的对话，理解并确定他们正在谈论的是谁，以及当前的对话主题是什么。这无疑是对聊天机器人的一大提升，让它更接近人类的交流方式，更好地满足人们的期望。

记忆是在用户与语言模型的互动过程中保留状态概念的过程。用户与语言模型的交互被封装在聊天信息（ChatMessages）这个概念中，因此记忆的关键在于如何摄取、捕捉、转化以及从一系列的聊天信息中提取知识。为此，我们有多种方式可以实现，每种方式都有其对应的记忆类型。

对于每种记忆，我们通常有两种理解方式。一种是独立的函数，从一系列信息中提取信息；另一种则是在链中使用这种记忆类型的方式。

记忆可以返回多种信息（例如，最近的N条信息和所有先前信息的摘要）。返回的信息可以是字符串或是一列信息。

我们会在后面分别介绍这些记忆类型。我们将展示如何在此使用模块化的工具函数，然后展示如何在链中使用它。

####   记忆组件的核心类

记忆组件的核心实用类之一，是支持大多数（如果不是所有）记忆组件的ChatMessageHistory类。这是一个超轻量级的封装器，它提供了便利的方法，可以保存人类与AI的消息，并取出所有的消息。

如果你在链之外管理记忆，你可能会直接使用这个类。

ChatMessageHistory类以非常方便的方式封装了人类和AI的对话消息。它提供了一种简单的添加方法，使得我们可以将新的聊天信息加入到历史聊天记录中。这个类有一个重要的特性，就是它可以以字符串或消息列表的形式提取信息。如果你的程序在链调用中有特定的记忆对象类型，那么这个基类将在链调用不存在时发挥作用。

以下是ChatMessageHistory类的示例代码：

```
from langchain.memory import ChatMessageHistory

history = ChatMessageHistory()

history.add_user_message("hi!")

history.add_ai_message("whats up?")
```
这个记忆还可以继续添加，我们打印看看 `history.messages` 的结果：

```
    [HumanMessage(content='hi!', additional_kwargs={}),
     AIMessage(content='whats up?', additional_kwargs={})]
```

####   与数据库程序集成

ChatMessageHistory类可以与所有跨平台的数据库（包括MongoDB，Redis 等主流数据库）进行集成。

langchain提供了一个集成包装器，你可以利用它调用ChatMessageHistory类的方法。值得一提的是，你不必清楚所有数据库都被包装成了什么样的类，因为它们都基于ChatMessageHistory类。你可以直接使用这个类的方法，就像调用ChatMessageHistory类的方法一样简单。

下面的代码是与Redis 数据库程序集成的记忆组件的调用方式。

```
from langchain.memory import RedisChatMessageHistory

history = RedisChatMessageHistory("foo")

history.add_user_message("hi!")

history.add_ai_message("whats up?")
```

从代码中我们可以看到跟ChatMessageHistory类添加记忆的方式完全一样。只是“ ChatMessageHistory ”类名加上了“ Redis ”。

####   模型记忆遇到的问题

记忆组件的工作原理相当直观：它接收聊天消息，将这些消息作为提示，然后传递给语言模型。随后，模型将根据这些提示的约束，生成相应的输出。

你可能会注意到，所有这些信息都会被存储在“提示”中。然而，这种方法在某些情况下会出现问题，那就是当我们进行长期的对话时，语言模型可接受的令牌范围内可能无法容纳所有的信息。

当前，OpenAI的最新模型在一次处理中可以使用大约16k个令牌。尽管这是一个巨大的信息量，但它仍然无法保存完整的对话内容。这就引发了一系列问题，我们该如何解决这个问题呢？

比如对话摘要记忆组件就提供了一个解决方案。这种记忆模式会在对话进行的过程中实时总结对话内容，并将当前的摘要存储在记忆中。然后，这些记忆可以被用于将迄今为止的对话摘要注入到提示或链中。这种记忆方式对于长期的对话来说尤其有用，因为如果直接在提示中保留过去的消息历史，可能会占用太多的令牌。

这就是记忆组件在LangChain中的运作方式。

在接下来的部分中，我们主要会对比和探讨几种不同的记忆类型，但这会涉及到链和代理的应用。如果你对链和代理的概念不熟悉，你可能会觉得难以理解。因此，建议你在深入了解这两个概念之后再继续阅读下文。
# 5.2 记忆组件的区别

在进行长期对话时，由于语言模型可接受的令牌范围有限，我们可能无法将所有的对话信息都包含进去。为了解决这个问题，LangChain提供了多种记忆组件，下面，我将通过一些代码示例来阐释我们在使用这些不同记忆组件时的区别。

首先，我们需要了解的是，LangChain提供了包括聊天窗口缓冲记忆类、摘要记忆类、知识图谱和实体记忆类等多种记忆组件。这些组件的不同主要体现在参数配置和实现效果上。选择哪一种记忆组件，需要根据我们在实际生产环境的需求来决定。

例如，如果你与聊天机器人的互动次数较少，那么可以选择使用ConversationBufferMemory组件。而ConversationSummaryMemory组件不会保存对话消息的格式，而是通过调用模型的摘要能力来得到摘要内容，因此它返回的都是摘要，而不是分角色的消息。

ConversationBufferWindowMemory组件则是通过设置参数k来决定保留的交互次数。例如，如果我们设置k=2，那么只会保留最后两次的交互记录。

此外，ConversationSummaryBufferMemory组件可以设置缓冲区的标记数`max_token_limit`，在做摘要的同时记录最近的对话。

ConversationSummaryBufferWindowMemory组件既可以做摘要，也可以记录聊天信息。

实体记忆类是专门用于提取对话中出现的特定实体和其关系信息的。知识图谱记忆类则试图通过对话内容来提取信息，并以知识图谱的形式呈现这些信息。

值得注意的是，摘要类、实体记忆、知识图谱这些类别的记忆组件在实现上相对复杂一些。例如，实现摘要记忆的时候，需要先调用大模型得到结果，然后再将这个结果作为摘要记忆的内容。而在实现这一过程时，我们需要传递语言模型进去。

在实体记忆和知识图谱这些类别的记忆组件中，返回的不是消息列表类型，而是可以格式化为三元组的信息。因此，我们在使用这些记忆组件的时候，最为困难的部分就是编写提示。所以，如果你想要更深入地理解和学习这两种记忆组件，那么就需要特别关注他们的输出类型和提示模板。
# 会话记忆和窗口记忆

在 LangChain 中，我们需要导入所需的记忆类型，然后实例化，最后传递给链（Chain），完成了模型的记忆功能。

为了比较ConversationBufferMemory 和 ConversationBufferWindowMemory 之间的区别。我们先导入公共的代码。

####  公共代码

先安装库：
```
!pip -q install openai langchain
```
设置密钥：
```
import os
os.environ['OPENAI_API_KEY'] = ''
```
引入各组件，实例化一个会话链（ConversationChain）。 这里我们使用的 Chain 只是一个简单的对话链 `ConversationChain`，允许我们跟 OpenAI 模型交互，并传递我们想要说的内容。：

```python
from langchain import OpenAI
llm = OpenAI(model_name='text-davinci-003', 
             temperature=0, 
             max_tokens = 256)
from langchain.chains import ConversationChain           
```
####   缓冲区记忆代码

我们先看对话缓冲区记忆, 这里先导入且实例化ConversationBufferMemory组件。
```
from langchain.chains.conversation.memory import ConversationBufferMemory
memory = ConversationBufferMemory()  
```

让我们开始对话，每次输入后等待AI返回的信息后，再输下一条：

```python
# 请依次运行以下代码，不要一次性运行。
conversation.predict(input="你好，我叫美丽")
conversation.predict(input="今天心情怎么样？")
conversation.predict(input="我想找客户服务中心")
conversation.predict(input="我的洗衣机坏了")
```

执行完最后一条对话后，我们看到的会话链显示：

``` 
'> Entering new  chain...
Prompt after formatting:
The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.

Current conversation:
Human: 你好，我叫美丽
AI:  你好，美丽！很高兴认识你！我是一个AI，我可以回答你的问题，也可以与你聊天。你想问我什么？
Human: 今天心情怎么样？
AI:  今天我的心情很好！我很开心能够和你聊天！
Human: 我想找客户服务中心
AI:  好的，我可以帮助你找到客户服务中心。你知道客户服务中心在哪里吗？
Human: 我的洗衣机坏了
AI:  哦，很抱歉听到你的洗衣机坏了。你可以联系客户服务中心来获得帮助。你知道客户服务中心的联系方式吗？
Human: 我的洗衣机坏了
AI:

'> Finished chain.
```

你会看到，Current conversation 内部把所有人类和AI的对话记录都保存了。这样做可以让你看到我们之前对话的确切内容，这是 LangChain 中最简单的记忆形式，但仍然非常强大，特别是当你知道人们与你的聊天的互动次数有限，或者你实际上要在五次互动后关闭它，诸如此类的情况下，这种记忆将非常有效。

####   窗口缓冲区记忆

运行完公共代码后，再导入且实例化窗口缓冲区记忆（ConversationBufferWindowMemory）组件。

```
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
memory = ConversationBufferWindowMemory(k=2)

conversation = ConversationChain(
    llm=llm, 
    verbose=True, 
    memory=memory
)
```

让我们开始对话，每次输入后等待AI返回的信息后，再输下一条：

```python
# 请依次运行以下代码，不要一次性运行。
conversation.predict(input="你好，我叫美丽")
conversation.predict(input="今天心情怎么样？")
conversation.predict(input="我想找客户服务中心")
conversation.predict(input="我的洗衣机坏了")
```

执行完最后一条对话后，我们看到的会话链显示：

``` 
'> Entering new  chain...
Prompt after formatting:
The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.

Current conversation:
Human: 今天心情怎么样？
AI:  今天我的心情很好！我很开心能够和你聊天！
Human: 我想找客户服务中心
AI:  好的，我可以帮助你找到客户服务中心。你知道客户服务中心在哪里吗？
Human: 我的洗衣机坏了
AI:

'>  Finished chain.
```

####  跟缓冲区记忆的区别在哪里？

为了对比两种类型的区别，我们采用的对话的内容是一样，但是由于我们设置了k = 2，但你实际上可以将其设置得比这个高得多，可以获取最后 5 次或 10 次互动，具体取决于你想使用多少个标记。

最后打印的Current conversation部分，我们可以看到最初打招呼，介绍我自己的信息和AI回应并没有记忆下来。它 **丢失** 了“你好，我叫美丽”这句话，它只将最后两次互动传递给大型语言模型。

所以如果你有一些情况，可以将其设置为 k = 5 或者 k = 10 ，大多数对话可能不会有很大变化。实际上缓冲区窗口记忆比缓冲区记忆多了限制，不会记录所有人和AI的会话记录,而是根据配置K的数值来对话记录条目，实现控制提示长度的目的。 
# 5.1.5 会话摘要记忆

在进行长期的对话时，由于语言模型可接受的令牌范围有限，我们可能无法将所有的对话信息都包含进去。为了解决这个问题，LangChain提供了不同的摘要记忆组件来控制提示的tokens。接下来，我将通过一些代码示例来阐释我们在使用这些不同记忆类型时的区别。

首先，我们需要了解的是，LangChain提供了两种主要的摘要记忆组件：会话缓冲区摘要记忆（ConversationSummaryBufferMemory）和会话摘要记忆（ConversationSummaryMemory）。这两者的定义是什么呢？

会话摘要记忆是一种摘要记忆组件，它不会逐字逐句地存储我们的对话，而是将对话内容进行摘要，并将这些摘要存储起来。这种摘要通常是整个对话的摘要，因此每次我们需要生成新的摘要时，都需要对大型语言模型进行多次调用，以获取响应。

而会话缓冲区摘要记忆则结合了会话摘要记忆的特性和缓冲区的概念。它会保存最近的交互记录，并将旧的交互记录编译成摘要，同时保留两者。但与会话摘要记忆不同的是，会话缓冲区摘要记忆是使用令牌长度而非交互次数来决定何时清除交互记录。这个记忆组件设定了缓冲区的标记数`max_token_limit`，超过此限制的对话记录将被清除。


####  公共代码

先安装库：
```
!pip -q install openai langchain
```
设置密钥：
```
import os
os.environ['OPENAI_API_KEY'] = ''
```
引入各组件，实例化一个会话链（ConversationChain）。 
这里我们使用的 Chain 只是一个简单的对话链 `ConversationChain`，允许我们跟 OpenAI 模型交互，并传递我们想要说的内容。：

```python
from langchain import OpenAI
llm = OpenAI(model_name='text-davinci-003', 
             temperature=0, 
             max_tokens = 256)
from langchain.chains import ConversationChain           
```
####   会话摘要记忆

我们先看会话摘要记忆, 这里先导入且实例化组件。
```
from langchain.chains.conversation.memory import ConversationSummaryMemory
memory = ConversationSummaryMemory()  
```

让我们开始对话，每次输入后等待AI返回的信息后，再输下一条：

```python
# 请依次运行以下代码，不要一次性运行。
conversation.predict(input="你好，我叫美丽")
conversation.predict(input="今天心情怎么样？")
conversation.predict(input="我想找客户服务中心")
conversation.predict(input="我的洗衣机坏了")
```

执行完最后一条对话后，我们看到的会话链显示：

``` 
'> Entering new  chain...
Prompt after formatting:
The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.

Current conversation:

The human introduces themselves as "美丽", to which the AI responds with a friendly greeting and informs them that they are an AI who can both answer questions and converse. The AI then asks the human what they would like to ask, to which the human responds by asking the AI what their mood is. The AI responds that they are feeling good and are excited to be learning new things, and then asks the human what their mood is. The human then requests to find the customer service center, to which the AI responds that they can help them find it and asks where they want to find the customer service center.
Human: 我的洗衣机坏了
AI:
'> Finished chain.
```

你会看到，Current conversation 它不会逐字逐句地存储我们的对话，而是将对话内容进行摘要，并将这些摘要存储起来。这种摘要通常是整个对话的摘要。

####   会话缓冲区摘要记忆

运行完公共代码后，再导入且实例化组件。我们设置为缓冲区的标记数 `max_token_limit` 限制为 40 个标记。

```
from langchain.chains.conversation.memory import ConversationSummaryBufferMemory
memory = ConversationSummaryBufferMemory(llm=OpenAI(),max_token_limit=40)
conversation = ConversationChain(
    llm=llm, 
    verbose=True, 
    memory=memory
)
```

让我们开始对话，每次输入后等待AI返回的信息后，再输下一条：

```python
# 请依次运行以下代码，不要一次性运行。
conversation.predict(input="你好，我叫美丽")
conversation.predict(input="今天心情怎么样？")
conversation.predict(input="我想找客户服务中心")
conversation.predict(input="我的洗衣机坏了")
```

执行完最后一条对话后，我们看到的会话链显示：

``` 
'> Entering new  chain...
Prompt after formatting:
The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.

Current conversation:
System: 

The AI is introduced and greets the human, telling the human that it is an AI and can answer questions or have a conversation. The AI then asks the human what they would like to ask, to which the human replied asking how the AI is feeling. The AI replied that it was feeling good and was excited for the conversation. The human then asked to find the customer service center, to which the AI replied that it could help the human find the customer service center, and asked if the human wanted to know the location.
Human: 我的洗衣机坏了
AI:

'>  Finished chain.
'很抱歉听到你的洗衣机坏了。我可以帮助你找到客户服务中心，你可以联系他们来解决你的问题。你想知道客户服务中心的位置吗？'
```

你会看到，Current conversation 内容保,丢失了前面对话内容，但做了摘要，将对话以摘要的形式其包含在内，比如丢失了打招呼的对话。在最后给出了范围在40个数量以内的标记的对话记录,即保留了 “Human: 我的洗衣机坏了” 。

但是在这里好像并没有对我们控制提示的长度有明显的改善，只不过是把所有的对话编程了摘要，从结果上来看，似乎比对话更啰嗦，使用的单词数量更多。

#### ##4    更新摘要


我们接着与他对话，不同的是我们抛出更多的问题，之前说洗衣机坏了，现在我抛出一些干扰的对话，比如说手机也坏了。我们看看他的摘要是否会更新？

```
# 请依次执行而不是一次性执行
conversation.predict(input="你知道洗衣机的操作屏幕显示ERROR是怎么回事吗?")
conversation.predict(input="我不知道他们的位置，你可以帮我找到他们的位置吗？")
conversation.predict(input="我打过他们客服中心的电话了，但是没人接听？")
conversation.predict(input="我的手机也坏了")
```
最后我们看看记忆内存中保存了什么：

```
print(memory.moving_summary_buffer)
```

```
The AI is introduced and greets the human, telling the human that it is an AI and can answer questions or have a conversation. The AI then asks the human what they would like to ask, to which the human replied asking how the AI is feeling. The AI replied that it was feeling good and was excited for the conversation. The human then asked to find the customer service center, to which the AI replied that it could help the human find the customer service center and asked if the human wanted to know the location. The human then stated that their washing machine and phone were broken, to which the AI apologized and offered to help the human find the customer service center so they can contact them to solve their problem, asking if the human wanted to know the address of the customer service center. The AI then offered to provide the address of the customer service center for the human, asking if they wanted to know it and offering to help the human find other contact methods such as email or social media accounts.
```
我们可以看到记忆存的是“The human then stated that their washing machine and phone were broken”, 多次对话的摘要，更新了。很早之前我们说洗衣机坏了，但是对话结束说手机坏了。这个组件做摘要记忆的时候，将两个相同的事情合并了。

####  摘要记忆组件的优势

我们可以看到，会话摘要记忆和会话缓冲区摘要记忆在实现方式上有着显著的差异。由于会话摘要记忆是基于整个对话生成的，所以每次进行新的摘要调用时，我们需要对大型语言模型进行多次调用，以获取对话的摘要。

这两种摘要记忆组件在对话管理中起着重要的作用，特别是在对话的token数超过模型能够处理的范围时，这种摘要能力就显得尤为重要。

无论是对话长度较长，还是需要进行精细管理的情况，会话摘要记忆和会话缓冲区摘要记忆都能够提供有效的帮助。
# 5.1.4 实体记忆和知识图谱记忆

在处理复杂对话时，我们常常需要提取对话中的关键信息。这种需求促使我们发展出了知识图谱和实体记忆这两种主要工具。

知识图谱是一种特殊的记忆类型，它能够根据对话内容构建出一个信息网络。每当它识别到相关的信息，都会接收这些信息并逐步构建出一个小型的知识图谱。与此同时，这种类型的记忆也会产生一种特殊的数据类型——知识图谱数据类型。

另一方面，实体记忆则专注于在对话中提取特定实体的信息。它使用大型语言模型(LLM)提取实体信息，并随着时间的推移，通过同样的方式积累对这个实体的知识。因此，实体记忆的结果通常表现为关于特定事物的关键信息字典。

这两种工具都试图根据对话内容来表示对话，并提取其中的信息。实体记忆和知识图谱记忆就是这种情况的两个例子。

为了让你能更清晰地理解它们的功能，我们在代码示例中放入了一些简单的提示。这些提示能够让你看到当AI仅使用相关信息部分中包含的信息，并且不会产生幻觉时，它们实际上在做什么。我们的目标是让AI更专注于我们所说的内容，而不是无端地添加新的信息。

在实践中，我们通常会使用简单的提示来引导AI的操作。当AI仅使用相关信息部分中包含的信息，并且不会产生幻觉时，这些提示才会被使用。通过这种方式，我们可以确保AI在处理对话时始终保持清晰和准确。

####  公共代码

先安装库：
```
!pip -q install openai langchain
```
设置密钥：
```
import os
os.environ['OPENAI_API_KEY'] = ''
```
引入各组件，实例化一个会话链（ConversationChain）。为了让你能更清晰地理解它们的功能，我们在引入了提示模板组件，构造一个提示，方便大家理解。
这里我们使用的 Chain 只是一个简单的对话链 `ConversationChain`，允许我们跟 OpenAI 模型交互，并传递我们想要说的内容。：

```python
from langchain import OpenAI
llm = OpenAI(model_name='text-davinci-003', 
             temperature=0, 
             max_tokens = 256)
from langchain.chains import ConversationChain
from langchain.prompts.prompt import PromptTemplate           
```
####   知识图谱记忆代码


先导入且实例化组件。
```
from langchain.chains.conversation.memory import ConversationKGMemory
```
构建一个简单的提示,让AI仅使用相关信息部分中包含的信息，并且不会产生幻觉。通过这种方式，我们可以确保 AI 在处理对话时始终保持清晰和准确。

```
template = """The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. 
If the AI does not know the answer to a question, it truthfully says it does not know. The AI ONLY uses information contained in the "Relevant Information" section and does not hallucinate.

Relevant Information:

{history}

Conversation:
Human: {input}
AI:"""
prompt = PromptTemplate(
    input_variables=["history", "input"], template=template)

conversation = ConversationChain(
    llm=llm, 
    verbose=True, 
    prompt=prompt,
    memory=ConversationKGMemory(llm=llm)
)
```

让我们开始对话，每次输入后等待AI返回的信息后，再输下一条：

```python
# 请依次运行以下代码，不要一次性运行。
conversation.predict(input="你好，我叫美丽")
conversation.predict(input="今天心情怎么样？")
conversation.predict(input="我想找客户服务中心")
conversation.predict(input="我的洗衣机坏了,操作面板出现ERROR字样")
conversation.predict(input="我的保修卡编号是A512423")
```

执行完最后一条对话后，我们打印记忆组件保存的知识图谱数据的结果：

``` 
print(conversation.memory.kg)
print(conversation.memory.kg.get_triples())
```
我们可以看到记忆内存里面保存了对话中的关键信息。
```
<langchain.graphs.networkx_graph.NetworkxEntityGraph object at 0x000001C953D48CD0>
[('AI', 'good mood', 'has a'), ('AI', 'new skills', 'is learning'), ('AI', 'talking to Human', 'enjoys'), ('Customer Service Center', 'city center', 'is located in'), ('Customer Service Center', '24 hour service', 'provides'), ('Customer Service Center', 'website', 'can be found on'), ('Customer Service Center', 'phone', 'can be contacted by'), ('Washing machine', 'ERROR on the control panel', 'has'), ('Human', 'A512423', 'has a warranty card number')]
```


####   实体知识记忆

先导入且实例化组件。
```
from langchain.chains.conversation.memory import ConversationEntityMemory
from langchain.chains.conversation.prompt import ENTITY_MEMORY_CONVERSATION_TEMPLATE
```
我们引入Langchain封装好的实体记忆的提示模板 ENTITY_MEMORY_CONVERSATION_TEMPLATE。

```
conversation = ConversationChain(
    llm=llm, 
    verbose=True, 
    prompt=ENTITY_MEMORY_CONVERSATION_TEMPLATE,
    memory=ConversationEntityMemory(llm=llm)
)
```

让我们开始对话，每次输入后等待AI返回的信息后，再输下一条：

```python
# 请依次运行以下代码，不要一次性运行。
conversation.predict(input="你好，我叫美丽")
conversation.predict(input="今天心情怎么样？")
conversation.predict(input="我想找客户服务中心")
conversation.predict(input="我的洗衣机坏了,操作面板出现ERROR字样")
conversation.predict(input="我的保修卡编号是A512423")
```

执行完最后一条对话后，我们打印记忆组件保存的知识图谱数据的结果：

``` 
print(conversation.memory.entity_cache )
```
我们可以看到记忆内存里面保存了对话中的关键信息。
```
['A512423', 'ERROR']
```

这两种工具都试图根据对话内容来表示对话，并提取其中的信息。
# 自定义记忆组件

虽然LangChain中预设了几种类型的记忆组件，但在实际应用中，你可能需要根据自己的应用场景来添加自定义的记忆类型。

在本节中，我们将向`ConversationChain`添加一个自定义的记忆类型。为了添加自定义的记忆类，我们需要引入基础记忆类并对其进行子类化。

```
from langchain import OpenAI, ConversationChain
from langchain.schema import BaseMemory
from pydantic import BaseModel
from typing import List, Dict, Any
```

在这个例子中，我们将编写一个自定义记忆类，它使用spacy来提取实体，并将有关它们的信息保存在一个简单的哈希表中。然后，在对话中，我们将观察输入文本，提取任何实体，并将关于它们的任何信息放入上下文中。

请注意，这种实现方式相当简单且脆弱，可能在生产环境中并不实用。它的目的是展示你可以添加自定义记忆实现。为此，我们需要spacy。

```
!pip install spacy
!python -m spacy download en_core_web_lg
```

```
import spacy

nlp = spacy.load("en_core_web_lg")
```

```
class SpacyEntityMemory(BaseMemory, BaseModel):
    """Memory class for storing information about entities."""

    # Define dictionary to store information about entities.
    entities: dict = {}
    # Define key to pass information about entities into prompt.
    memory_key: str = "entities"

    def clear(self):
        self.entities = {}

    @property
    def memory_variables(self) -> List[str]:
        """Define the variables we are providing to the prompt."""
        return [self.memory_key]

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, str]:
        """Load the memory variables, in this case the entity key."""
        # Get the input text and run through spacy
        doc = nlp(inputs[list(inputs.keys())[0]])
        # Extract known information about entities, if they exist.
        entities = [
            self.entities[str(ent)] for ent in doc.ents if str(ent) in self.entities
        ]
        # Return combined information about entities to put into context.
        return {self.memory_key: "\n".join(entities)}

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Save context from this conversation to buffer."""
        # Get the input text and run through spacy
        text = inputs[list(inputs.keys())[0]]
        doc = nlp(text)
        # For each entity that was mentioned, save this information to the dictionary.
        for ent in doc.ents:
            ent_str = str(ent)
            if ent_str in self.entities:
                self.entities[ent_str] += f"\n{text}"
            else:
                self.entities[ent_str] = text
```

我们现在定义一个提示，它接受关于实体的信息以及用户输入的信息。

```
from langchain.prompts.prompt import PromptTemplate

template = """The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know. You are provided with information about entities the Human mentions, if relevant.

Relevant entity information:
{entities}

Conversation:
Human: {input}
AI:"""
prompt = PromptTemplate(input_variables=["entities", "input"], template=template)
```

然后我们组合在一起。

```
llm = OpenAI(temperature=0)
conversation = ConversationChain(
    llm=llm, prompt=prompt, verbose=True, memory=SpacyEntityMemory()
)
```

在第一个例子中，由于没有关于Harrison的先验知识，"Relevant entity information"（相关实体信息）部分是空的。

```
conversation.predict(input="Harrison likes machine learning")
```

```
    
    
    > Entering new ConversationChain chain...
    Prompt after formatting:
    The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know. You are provided with information about entities the Human mentions, if relevant.
    
    Relevant entity information:
    
    
    Conversation:
    Human: Harrison likes machine learning
    AI:
    
    > Finished ConversationChain chain.





    " That's great to hear! Machine learning is a fascinating field of study. It involves using algorithms to analyze data and make predictions. Have you ever studied machine learning, Harrison?"
```

现在在第二个例子中，我们可以看到它抽取了关于Harrison的信息。

```
conversation.predict(
    input="What do you think Harrison's favorite subject in college was?"
)
```

```
    
    
    > Entering new ConversationChain chain...
    Prompt after formatting:
    The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know. You are provided with information about entities the Human mentions, if relevant.
    
    Relevant entity information:
    Harrison likes machine learning
    
    Conversation:
    Human: What do you think Harrison's favorite subject in college was?
    AI:
    
    > Finished ConversationChain chain.





    ' From what I know about Harrison, I believe his favorite subject in college was machine learning. He has expressed a strong interest in the subject and has mentioned it often.'
```

请再次注意，这种实现方式相当简单且脆弱，可能在生产环境中并不实用。它的目的是为了展示你可以添加自定义的记忆实现。
# 代理概述

想象一下，如果人工智能能像我们一样，具有推理能力，能够自主提出计划，批判性地评估这些想法，甚至将其付诸实践，那会是怎样的景象？就像电影《HER》中的人工智能萨曼莎，她不仅能与人进行深入的对话，还能够帮助西奥多规划日常生活，安排行程等。这样的设想一度只存在于科幻作品中，但现在通过代理技术，它似乎已经触手可及。

最近Yohei Nakajima的研究论文《Task-driven Autonomous Agent Utilizing GPT-4, Pinecone, and LangChain for Diverse Applications》展示了代理的突破创造性。在该研究中，作者提出了一个利用OpenAI的GPT-4语言模型、Pinecone向量搜索和LangChain框架的任务驱动的自主代理，该代理可以在多样化的领域中完成各种任务，生成基于完成结果的新任务，并在实时中优先处理任务。

那么Langchain中的代理是什么？为什么要使用代理？

####   代理的定义

代理的核心思想是将语言模型作为推理引擎，依据其确定如何与外部世界交互以及应采取何种行动。这意味着，代理的行动序列是根据用户输入而变化的，无需遵循硬编码的顺序，例如，“先做A，再做B，然后做C”。相反，代理依据用户的输入和之前的行动结果来决定下一步采取何种行动。

####   为什么使用代理?

代理和工具使用的概念紧密相关。工具能够将代理和外部数据源或计算（如搜索API、数据库）链接起来，这克服了语言模型的某些限制，例如语言模型无法直接理解你的数据，也不擅长处理数学运算。然而，工具的使用并不仅限于代理。你依然可以使用工具，将语言模型连接到搜索引擎，而无需代理。然而，使用代理的优势在于它们具有更高的灵活性、更强大的处理能力，能够更好地从错误中恢复，并处理复杂的任务。

例如，当与SQL数据库交互时，可能需要执行多个查询才能回答某些问题，而简单的工具序列可能很快就会遇到一些边缘情况。在这种情况下，代理这种更为灵活的框架能够更好地解决问题。

####   代理的应用实现

在代理的典型实现中，用户首先提出一个请求，然后利用语言模型选择应使用的工具。接下来，执行对该工具的操作，获取观察结果，再将其反馈给语言模型，进行下一步操作，如此循环，直到满足停止条件。

停止条件有多种类型，最常见的是语言模型本身意识到任务已完成，应该给出回复。也可能有其他更具体的规则，比如，如果代理已经连续执行了五个步骤，但还没有得到最终答案，那么可能需要返回某些结果。在讨论代理的可靠性时，这些规则可以提供帮助。

代理的基本思想是选择一个工具，观察其输出，然后继续进行下一步。使用代理的方式，可以有效提高处理复杂任务的效率和准确性。

####   代理的应用场景

Agent的一项重要功能是根据用户的问题来决定最佳的工具。它首先会检查输入，然后根据你初始化的工具来做出选择。接着，它会根据用户的问题，实际运行这些工具，从而生成预期的输出。通过这种方式，代理可以根据不同的用户输入，灵活地选择和使用不同的工具，从而提供丰富和多样的服务。

例如，假设您正在为一家电子商务企业构建一个聊天机器人。虽然您可以使用GBT4等语言模型进行聊天，但这些模型对于了解您的产品非常有限。为了解决这个问题，我们可以使用LangChain的向量存储功能，将产品数据存储在数据库中，并让语言模型可以访问这些数据。这样一来，聊天模型就可以更好地了解您的产品。

然而，单单了解产品还不够。如果聊天机器人在网页上运行，它还需要了解访问的上下文。这可能包括一些信息，比如访问者是新潜在客户还是现有客户，或者基于浏览历史来推荐产品。为了让语言模型在与客户交互时具备这些上下文信息，我们可以通过微服务向聊天模型提供这些信息。

通过让语言模型访问资源和上下文信息，语言模型可以与客户进行更好的交互，帮助企业转化客户并增加销售额。这些资源和信息可以通过代理的方式提供给语言模型。


####   代理实现的方式：ReAct

一种主要且广泛应用的方法和策略被称为ReAct，这是“Reasoning and Acting（推理与行动）”的缩写。这一策略首次由普林斯顿大学在他们的一篇优秀论文中提出，现已被广泛应用于代理实现。

####  ReAct的优势

在许多应用场景中，ReAct策略已证明自己是非常有效的。例如，考虑这样一个问题："除了Apple remote之外，还有哪些设备可以控制与Apple remote最初设计的互动的程序？"。最基本的提示策略是直接将这个问题交给语言模型处理，但ReAct策略赋予了代理更大的灵活性和实力。代理不仅可以使用语言模型，还可以连接到其他工具、数据源或计算环境，例如搜索API和数据库，以此来克服语言模型的某些局限性，例如对数据的不了解或数学运算能力有限。这样，即使遇到需要多次查询才能回答的问题，或者其他一些边界情况，代理也能够灵活应对，从而使其成为一种更强大的问题解决工具。

####  ReAct 如何工作?

ReAct策略的工作原理是什么呢？重申一下，代理的核心思想是将语言模型作为推理引擎。ReAct策略是将推理和行动结合在一起的方式。代理接收到用户的请求，然后使用语言模型选择要使用的工具。然后代理执行该工具的操作，观察结果，然后将这些结果反馈给语言模型。这个过程会持续进行，直到满足某些停止条件。停止条件可以有很多种，最常见的是语言模型认为任务已经完成，需要将结果返回给用户。这种方式使得代理具有更高的灵活性和强大的问题解决能力，这是ReAct策略的核心优势。

####   实现代理应用的挑战

在实现代理应用的过程中，我们面临许多挑战，以下列举了几个主要的：

首先，使代理在适当的场景下使用工具是我们面临的一个基本挑战。如何在合适的情况下让代理采用恰当的工具，并优化其使用效果呢？在ReAct论文中，通过引入推理的角度，以及使用"CoT 思考链"的提示方式，我们寻求解决这个问题。在实际操作中，我们常常需要明确告知代理可使用的工具，以及通过这些工具能克服的限制。所以，工具的描述信息也非常重要，如果我们希望代理能用特定的工具，就需要提供足够的上下文信息，使代理能理解工具的优点和应用场景。

其次，对于工具的选择，我们需要进行检索。这一步骤可以解决上述的问题。我们可以运行一些检索步骤，例如嵌入式搜索查找，以获取可能的工具，然后将这些工具传递给提示，由语言模型进行后续步骤。

此外，提供相关的示例也是一种有效的方法。选择与当前任务类似的示例，通常比随机示例更有帮助。相同地，检索最相关的示例也有巨大的潜力。

最后，我们还需要注意避免在不需要的情况下使用工具。可以在提示中加入相关信息或提醒，告诉代理在对话时不必使用工具。

####   实现代理应用的实用技巧

在解决这些挑战的过程中，我们总结出了一些实用的技巧：

首先，结构化的响应更易于解析。通常情况下，你提问的响应越结构化，解析起来就越容易。语言模型在编写JSON方面表现得很好，因此我们将一些代理转换为使用JSON格式。

其次，我们引入了输出解析器的概念。输出解析器封装了解析响应所需的全部逻辑，并以尽可能模块化的方式实现。另一个相关的概念是，输出解析器可以重试和修复错误。如果有格式错误的模式，你可以通过将输出和错误传递给它来显式地修复响应。

此外，记住之前的步骤也是很重要的。最基本的方法是在内存中保留这些步骤的列表。然而，在处理长时间运行的任务时，会遇到一些上下文窗口的问题。我们已经找到了一种解决方法，即使用一些检索方法来获取之前的步骤，并将其放入上下文中。

最后，在处理API时，我们经常遇到观察结果太长的问题。因为API通常会返回非常大且难以放入上下文的JSON数据。常见的解决方法是对其进行解析，可以简单地将该大数据块转换为字符串，并将前1000个字符作为响应。

####   最新的代理应用项目及其对改进的探索

近期的代理应用项目研发涉猎广泛，主要集中在如何改善代理的各种工作方式上。下面介绍四个具有代表性的项目。

1. AutoGPT：AutoGPT 的目标设置有别于 ReAct 代理的重大不同。AutoGPT 的追求在于如何增加 Twitter 的关注者数量或实现其他类似的开放性、广泛性和长期性目标。相较之下，ReAct 代理则专注于实现短期内可量化的目标。为了实现这样的目标，AutoGPT引入了长期记忆的概念，促进代理与工具之间的互动，这有助于提升代理的规划和执行效率。

2. Baby AGI：Baby AGI 的研发采用了逐步解决子问题的方法，以提升代理的规划和执行能力。这一项目明确了策划和执行步骤的定义，这一创新为提升长期目标代理的可行性和关注度提供了有益的思考途径。最初，Baby AGI 的策略实现主要依靠自主设定，然而现在已经开始融入了各种工具，从而优化代理执行计划的能力。

3. Camel：Camel 项目的一项主要创新是在模拟环境中进行代理之间的交互。通过这种方法，可以对代理进行评估和测试，并且可以作为一种娱乐手段。这种方法为检测代理交互提供了一种无需人工干预的方式，能够有效地测试代理模型。

4. Generative Agents：该项目的目标是通过构建一个复杂的模拟环境，让 25 个不同的代理在这个环境中进行互动，从而实现生成型代理。项目同时也注重处理代理的记忆和反思能力，代理能够通过记忆中的事件来指导下一步的行动，并在反思环节对最近的事件进行评估和更新。这种基于反思的状态更新机制适用于各种类型的记忆，例如实体记忆和知识图谱，从而提高代理对环境的建模能力。


####   最简单的代理示例

首先，让我们加载语言模型。

```
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.llms import OpenAI
```

```python
llm = OpenAI(temperature=0)
```
接下来，让我们加载一些要使用的工具。请注意，llm-math工具使用了一个LLM，所以我们需要传递进去。

```python
tools = load_tools(["serpapi", "llm-math"], llm=llm)
```


最后，让我们用这些工具、语言模型和我们想要使用的代理类型来初始化一个代理。

```python
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
```

现在，让我们来测试一下吧！

```
agent.run("Who is Leo DiCaprio's girlfriend? What is her current age raised to the 0.43 power?")
```

```
    > Entering new AgentExecutor chain...
     I need to find out who Leo DiCaprio's girlfriend is and then calculate her age raised to the 0.43 power.
    Action: Search
    Action Input: "Leo DiCaprio girlfriend"
    Observation: Camila Morrone
    Thought: I need to find out Camila Morrone's age
    Action: Search
    Action Input: "Camila Morrone age"
    Observation: 25 years
    Thought: I need to calculate 25 raised to the 0.43 power
    Action: Calculator
    Action Input: 25^0.43
    Observation: Answer: 3.991298452658078
    
    Thought: I now know the final answer
    Final Answer: Camila Morrone is Leo DiCaprio's girlfriend and her current age raised to the 0.43 power is 3.991298452658078.
    
    > Finished chain.





    "Camila Morrone is Leo DiCaprio's girlfriend and her current age raised to the 0.43 power is 3.991298452658078."
```

# 代理概述

想象一下，如果人工智能能像我们一样，具有推理能力，能够自主提出计划，批判性地评估这些想法，甚至将其付诸实践，那会是怎样的景象？就像电影《HER》中的人工智能萨曼莎，她不仅能与人进行深入的对话，还能够帮助西奥多规划日常生活，安排行程等。这样的设想一度只存在于科幻作品中，但现在通过代理技术，它似乎已经触手可及。

最近Yohei Nakajima的研究论文《Task-driven Autonomous Agent Utilizing GPT-4, Pinecone, and LangChain for Diverse Applications》展示了代理的突破创造性。在该研究中，作者提出了一个利用OpenAI的GPT-4语言模型、Pinecone向量搜索和LangChain框架的任务驱动的自主代理，该代理可以在多样化的领域中完成各种任务，生成基于完成结果的新任务，并在实时中优先处理任务。

那么Langchain中的代理是什么？为什么要使用代理？

####   代理的定义

代理的核心思想是将语言模型作为推理引擎，依据其确定如何与外部世界交互以及应采取何种行动。这意味着，代理的行动序列是根据用户输入而变化的，无需遵循硬编码的顺序，例如，“先做A，再做B，然后做C”。相反，代理依据用户的输入和之前的行动结果来决定下一步采取何种行动。

####   为什么使用代理?

代理和工具使用的概念紧密相关。工具能够将代理和外部数据源或计算（如搜索API、数据库）链接起来，这克服了语言模型的某些限制，例如语言模型无法直接理解你的数据，也不擅长处理数学运算。然而，工具的使用并不仅限于代理。你依然可以使用工具，将语言模型连接到搜索引擎，而无需代理。然而，使用代理的优势在于它们具有更高的灵活性、更强大的处理能力，能够更好地从错误中恢复，并处理复杂的任务。

例如，当与SQL数据库交互时，可能需要执行多个查询才能回答某些问题，而简单的工具序列可能很快就会遇到一些边缘情况。在这种情况下，代理这种更为灵活的框架能够更好地解决问题。

####   代理的应用实现

在代理的典型实现中，用户首先提出一个请求，然后利用语言模型选择应使用的工具。接下来，执行对该工具的操作，获取观察结果，再将其反馈给语言模型，进行下一步操作，如此循环，直到满足停止条件。

停止条件有多种类型，最常见的是语言模型本身意识到任务已完成，应该给出回复。也可能有其他更具体的规则，比如，如果代理已经连续执行了五个步骤，但还没有得到最终答案，那么可能需要返回某些结果。在讨论代理的可靠性时，这些规则可以提供帮助。

代理的基本思想是选择一个工具，观察其输出，然后继续进行下一步。使用代理的方式，可以有效提高处理复杂任务的效率和准确性。

####   代理的应用场景

Agent的一项重要功能是根据用户的问题来决定最佳的工具。它首先会检查输入，然后根据你初始化的工具来做出选择。接着，它会根据用户的问题，实际运行这些工具，从而生成预期的输出。通过这种方式，代理可以根据不同的用户输入，灵活地选择和使用不同的工具，从而提供丰富和多样的服务。

例如，假设您正在为一家电子商务企业构建一个聊天机器人。虽然您可以使用GBT4等语言模型进行聊天，但这些模型对于了解您的产品非常有限。为了解决这个问题，我们可以使用LangChain的向量存储功能，将产品数据存储在数据库中，并让语言模型可以访问这些数据。这样一来，聊天模型就可以更好地了解您的产品。

然而，单单了解产品还不够。如果聊天机器人在网页上运行，它还需要了解访问的上下文。这可能包括一些信息，比如访问者是新潜在客户还是现有客户，或者基于浏览历史来推荐产品。为了让语言模型在与客户交互时具备这些上下文信息，我们可以通过微服务向聊天模型提供这些信息。

通过让语言模型访问资源和上下文信息，语言模型可以与客户进行更好的交互，帮助企业转化客户并增加销售额。这些资源和信息可以通过代理的方式提供给语言模型。


####   代理实现的方式：ReAct

一种主要且广泛应用的方法和策略被称为ReAct，这是“Reasoning and Acting（推理与行动）”的缩写。这一策略首次由普林斯顿大学在他们的一篇优秀论文中提出，现已被广泛应用于代理实现。

####  ReAct的优势

在许多应用场景中，ReAct策略已证明自己是非常有效的。例如，考虑这样一个问题："除了Apple remote之外，还有哪些设备可以控制与Apple remote最初设计的互动的程序？"。最基本的提示策略是直接将这个问题交给语言模型处理，但ReAct策略赋予了代理更大的灵活性和实力。代理不仅可以使用语言模型，还可以连接到其他工具、数据源或计算环境，例如搜索API和数据库，以此来克服语言模型的某些局限性，例如对数据的不了解或数学运算能力有限。这样，即使遇到需要多次查询才能回答的问题，或者其他一些边界情况，代理也能够灵活应对，从而使其成为一种更强大的问题解决工具。

####  ReAct 如何工作?

ReAct策略的工作原理是什么呢？重申一下，代理的核心思想是将语言模型作为推理引擎。ReAct策略是将推理和行动结合在一起的方式。代理接收到用户的请求，然后使用语言模型选择要使用的工具。然后代理执行该工具的操作，观察结果，然后将这些结果反馈给语言模型。这个过程会持续进行，直到满足某些停止条件。停止条件可以有很多种，最常见的是语言模型认为任务已经完成，需要将结果返回给用户。这种方式使得代理具有更高的灵活性和强大的问题解决能力，这是ReAct策略的核心优势。

####   实现代理应用的挑战

在实现代理应用的过程中，我们面临许多挑战，以下列举了几个主要的：

首先，使代理在适当的场景下使用工具是我们面临的一个基本挑战。如何在合适的情况下让代理采用恰当的工具，并优化其使用效果呢？在ReAct论文中，通过引入推理的角度，以及使用"CoT 思考链"的提示方式，我们寻求解决这个问题。在实际操作中，我们常常需要明确告知代理可使用的工具，以及通过这些工具能克服的限制。所以，工具的描述信息也非常重要，如果我们希望代理能用特定的工具，就需要提供足够的上下文信息，使代理能理解工具的优点和应用场景。

其次，对于工具的选择，我们需要进行检索。这一步骤可以解决上述的问题。我们可以运行一些检索步骤，例如嵌入式搜索查找，以获取可能的工具，然后将这些工具传递给提示，由语言模型进行后续步骤。

此外，提供相关的示例也是一种有效的方法。选择与当前任务类似的示例，通常比随机示例更有帮助。相同地，检索最相关的示例也有巨大的潜力。

最后，我们还需要注意避免在不需要的情况下使用工具。可以在提示中加入相关信息或提醒，告诉代理在对话时不必使用工具。

####   实现代理应用的实用技巧

在解决这些挑战的过程中，我们总结出了一些实用的技巧：

首先，结构化的响应更易于解析。通常情况下，你提问的响应越结构化，解析起来就越容易。语言模型在编写JSON方面表现得很好，因此我们将一些代理转换为使用JSON格式。

其次，我们引入了输出解析器的概念。输出解析器封装了解析响应所需的全部逻辑，并以尽可能模块化的方式实现。另一个相关的概念是，输出解析器可以重试和修复错误。如果有格式错误的模式，你可以通过将输出和错误传递给它来显式地修复响应。

此外，记住之前的步骤也是很重要的。最基本的方法是在内存中保留这些步骤的列表。然而，在处理长时间运行的任务时，会遇到一些上下文窗口的问题。我们已经找到了一种解决方法，即使用一些检索方法来获取之前的步骤，并将其放入上下文中。

最后，在处理API时，我们经常遇到观察结果太长的问题。因为API通常会返回非常大且难以放入上下文的JSON数据。常见的解决方法是对其进行解析，可以简单地将该大数据块转换为字符串，并将前1000个字符作为响应。

####   最新的代理应用项目及其对改进的探索

近期的代理应用项目研发涉猎广泛，主要集中在如何改善代理的各种工作方式上。下面介绍四个具有代表性的项目。

1. AutoGPT：AutoGPT 的目标设置有别于 ReAct 代理的重大不同。AutoGPT 的追求在于如何增加 Twitter 的关注者数量或实现其他类似的开放性、广泛性和长期性目标。相较之下，ReAct 代理则专注于实现短期内可量化的目标。为了实现这样的目标，AutoGPT引入了长期记忆的概念，促进代理与工具之间的互动，这有助于提升代理的规划和执行效率。

2. Baby AGI：Baby AGI 的研发采用了逐步解决子问题的方法，以提升代理的规划和执行能力。这一项目明确了策划和执行步骤的定义，这一创新为提升长期目标代理的可行性和关注度提供了有益的思考途径。最初，Baby AGI 的策略实现主要依靠自主设定，然而现在已经开始融入了各种工具，从而优化代理执行计划的能力。

3. Camel：Camel 项目的一项主要创新是在模拟环境中进行代理之间的交互。通过这种方法，可以对代理进行评估和测试，并且可以作为一种娱乐手段。这种方法为检测代理交互提供了一种无需人工干预的方式，能够有效地测试代理模型。

4. Generative Agents：该项目的目标是通过构建一个复杂的模拟环境，让 25 个不同的代理在这个环境中进行互动，从而实现生成型代理。项目同时也注重处理代理的记忆和反思能力，代理能够通过记忆中的事件来指导下一步的行动，并在反思环节对最近的事件进行评估和更新。这种基于反思的状态更新机制适用于各种类型的记忆，例如实体记忆和知识图谱，从而提高代理对环境的建模能力。


####   最简单的代理示例

首先，让我们加载语言模型。

```
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.llms import OpenAI
```

```python
llm = OpenAI(temperature=0)
```
接下来，让我们加载一些要使用的工具。请注意，llm-math工具使用了一个LLM，所以我们需要传递进去。

```python
tools = load_tools(["serpapi", "llm-math"], llm=llm)
```


最后，让我们用这些工具、语言模型和我们想要使用的代理类型来初始化一个代理。

```python
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
```

现在，让我们来测试一下吧！

```
agent.run("Who is Leo DiCaprio's girlfriend? What is her current age raised to the 0.43 power?")
```

```
    > Entering new AgentExecutor chain...
     I need to find out who Leo DiCaprio's girlfriend is and then calculate her age raised to the 0.43 power.
    Action: Search
    Action Input: "Leo DiCaprio girlfriend"
    Observation: Camila Morrone
    Thought: I need to find out Camila Morrone's age
    Action: Search
    Action Input: "Camila Morrone age"
    Observation: 25 years
    Thought: I need to calculate 25 raised to the 0.43 power
    Action: Calculator
    Action Input: 25^0.43
    Observation: Answer: 3.991298452658078
    
    Thought: I now know the final answer
    Final Answer: Camila Morrone is Leo DiCaprio's girlfriend and her current age raised to the 0.43 power is 3.991298452658078.
    
    > Finished chain.





    "Camila Morrone is Leo DiCaprio's girlfriend and her current age raised to the 0.43 power is 3.991298452658078."
```

# 代理概述

想象一下，如果人工智能能像我们一样，具有推理能力，能够自主提出计划，批判性地评估这些想法，甚至将其付诸实践，那会是怎样的景象？就像电影《HER》中的人工智能萨曼莎，她不仅能与人进行深入的对话，还能够帮助西奥多规划日常生活，安排行程等。这样的设想一度只存在于科幻作品中，但现在通过代理技术，它似乎已经触手可及。

最近Yohei Nakajima的研究论文《Task-driven Autonomous Agent Utilizing GPT-4, Pinecone, and LangChain for Diverse Applications》展示了代理的突破创造性。在该研究中，作者提出了一个利用OpenAI的GPT-4语言模型、Pinecone向量搜索和LangChain框架的任务驱动的自主代理，该代理可以在多样化的领域中完成各种任务，生成基于完成结果的新任务，并在实时中优先处理任务。

那么Langchain中的代理是什么？为什么要使用代理？

####   代理的定义

代理的核心思想是将语言模型作为推理引擎，依据其确定如何与外部世界交互以及应采取何种行动。这意味着，代理的行动序列是根据用户输入而变化的，无需遵循硬编码的顺序，例如，“先做A，再做B，然后做C”。相反，代理依据用户的输入和之前的行动结果来决定下一步采取何种行动。

####   为什么使用代理?

代理和工具使用的概念紧密相关。工具能够将代理和外部数据源或计算（如搜索API、数据库）链接起来，这克服了语言模型的某些限制，例如语言模型无法直接理解你的数据，也不擅长处理数学运算。然而，工具的使用并不仅限于代理。你依然可以使用工具，将语言模型连接到搜索引擎，而无需代理。然而，使用代理的优势在于它们具有更高的灵活性、更强大的处理能力，能够更好地从错误中恢复，并处理复杂的任务。

例如，当与SQL数据库交互时，可能需要执行多个查询才能回答某些问题，而简单的工具序列可能很快就会遇到一些边缘情况。在这种情况下，代理这种更为灵活的框架能够更好地解决问题。

####   代理的应用实现

在代理的典型实现中，用户首先提出一个请求，然后利用语言模型选择应使用的工具。接下来，执行对该工具的操作，获取观察结果，再将其反馈给语言模型，进行下一步操作，如此循环，直到满足停止条件。

停止条件有多种类型，最常见的是语言模型本身意识到任务已完成，应该给出回复。也可能有其他更具体的规则，比如，如果代理已经连续执行了五个步骤，但还没有得到最终答案，那么可能需要返回某些结果。在讨论代理的可靠性时，这些规则可以提供帮助。

代理的基本思想是选择一个工具，观察其输出，然后继续进行下一步。使用代理的方式，可以有效提高处理复杂任务的效率和准确性。

####   代理的应用场景

Agent的一项重要功能是根据用户的问题来决定最佳的工具。它首先会检查输入，然后根据你初始化的工具来做出选择。接着，它会根据用户的问题，实际运行这些工具，从而生成预期的输出。通过这种方式，代理可以根据不同的用户输入，灵活地选择和使用不同的工具，从而提供丰富和多样的服务。

例如，假设您正在为一家电子商务企业构建一个聊天机器人。虽然您可以使用GBT4等语言模型进行聊天，但这些模型对于了解您的产品非常有限。为了解决这个问题，我们可以使用LangChain的向量存储功能，将产品数据存储在数据库中，并让语言模型可以访问这些数据。这样一来，聊天模型就可以更好地了解您的产品。

然而，单单了解产品还不够。如果聊天机器人在网页上运行，它还需要了解访问的上下文。这可能包括一些信息，比如访问者是新潜在客户还是现有客户，或者基于浏览历史来推荐产品。为了让语言模型在与客户交互时具备这些上下文信息，我们可以通过微服务向聊天模型提供这些信息。

通过让语言模型访问资源和上下文信息，语言模型可以与客户进行更好的交互，帮助企业转化客户并增加销售额。这些资源和信息可以通过代理的方式提供给语言模型。


####   代理实现的方式：ReAct

一种主要且广泛应用的方法和策略被称为ReAct，这是“Reasoning and Acting（推理与行动）”的缩写。这一策略首次由普林斯顿大学在他们的一篇优秀论文中提出，现已被广泛应用于代理实现。

####  ReAct的优势

在许多应用场景中，ReAct策略已证明自己是非常有效的。例如，考虑这样一个问题："除了Apple remote之外，还有哪些设备可以控制与Apple remote最初设计的互动的程序？"。最基本的提示策略是直接将这个问题交给语言模型处理，但ReAct策略赋予了代理更大的灵活性和实力。代理不仅可以使用语言模型，还可以连接到其他工具、数据源或计算环境，例如搜索API和数据库，以此来克服语言模型的某些局限性，例如对数据的不了解或数学运算能力有限。这样，即使遇到需要多次查询才能回答的问题，或者其他一些边界情况，代理也能够灵活应对，从而使其成为一种更强大的问题解决工具。

####  ReAct 如何工作?

ReAct策略的工作原理是什么呢？重申一下，代理的核心思想是将语言模型作为推理引擎。ReAct策略是将推理和行动结合在一起的方式。代理接收到用户的请求，然后使用语言模型选择要使用的工具。然后代理执行该工具的操作，观察结果，然后将这些结果反馈给语言模型。这个过程会持续进行，直到满足某些停止条件。停止条件可以有很多种，最常见的是语言模型认为任务已经完成，需要将结果返回给用户。这种方式使得代理具有更高的灵活性和强大的问题解决能力，这是ReAct策略的核心优势。

####   实现代理应用的挑战

在实现代理应用的过程中，我们面临许多挑战，以下列举了几个主要的：

首先，使代理在适当的场景下使用工具是我们面临的一个基本挑战。如何在合适的情况下让代理采用恰当的工具，并优化其使用效果呢？在ReAct论文中，通过引入推理的角度，以及使用"CoT 思考链"的提示方式，我们寻求解决这个问题。在实际操作中，我们常常需要明确告知代理可使用的工具，以及通过这些工具能克服的限制。所以，工具的描述信息也非常重要，如果我们希望代理能用特定的工具，就需要提供足够的上下文信息，使代理能理解工具的优点和应用场景。

其次，对于工具的选择，我们需要进行检索。这一步骤可以解决上述的问题。我们可以运行一些检索步骤，例如嵌入式搜索查找，以获取可能的工具，然后将这些工具传递给提示，由语言模型进行后续步骤。

此外，提供相关的示例也是一种有效的方法。选择与当前任务类似的示例，通常比随机示例更有帮助。相同地，检索最相关的示例也有巨大的潜力。

最后，我们还需要注意避免在不需要的情况下使用工具。可以在提示中加入相关信息或提醒，告诉代理在对话时不必使用工具。

####   实现代理应用的实用技巧

在解决这些挑战的过程中，我们总结出了一些实用的技巧：

首先，结构化的响应更易于解析。通常情况下，你提问的响应越结构化，解析起来就越容易。语言模型在编写JSON方面表现得很好，因此我们将一些代理转换为使用JSON格式。

其次，我们引入了输出解析器的概念。输出解析器封装了解析响应所需的全部逻辑，并以尽可能模块化的方式实现。另一个相关的概念是，输出解析器可以重试和修复错误。如果有格式错误的模式，你可以通过将输出和错误传递给它来显式地修复响应。

此外，记住之前的步骤也是很重要的。最基本的方法是在内存中保留这些步骤的列表。然而，在处理长时间运行的任务时，会遇到一些上下文窗口的问题。我们已经找到了一种解决方法，即使用一些检索方法来获取之前的步骤，并将其放入上下文中。

最后，在处理API时，我们经常遇到观察结果太长的问题。因为API通常会返回非常大且难以放入上下文的JSON数据。常见的解决方法是对其进行解析，可以简单地将该大数据块转换为字符串，并将前1000个字符作为响应。

####   最新的代理应用项目及其对改进的探索

近期的代理应用项目研发涉猎广泛，主要集中在如何改善代理的各种工作方式上。下面介绍四个具有代表性的项目。

1. AutoGPT：AutoGPT 的目标设置有别于 ReAct 代理的重大不同。AutoGPT 的追求在于如何增加 Twitter 的关注者数量或实现其他类似的开放性、广泛性和长期性目标。相较之下，ReAct 代理则专注于实现短期内可量化的目标。为了实现这样的目标，AutoGPT引入了长期记忆的概念，促进代理与工具之间的互动，这有助于提升代理的规划和执行效率。

2. Baby AGI：Baby AGI 的研发采用了逐步解决子问题的方法，以提升代理的规划和执行能力。这一项目明确了策划和执行步骤的定义，这一创新为提升长期目标代理的可行性和关注度提供了有益的思考途径。最初，Baby AGI 的策略实现主要依靠自主设定，然而现在已经开始融入了各种工具，从而优化代理执行计划的能力。

3. Camel：Camel 项目的一项主要创新是在模拟环境中进行代理之间的交互。通过这种方法，可以对代理进行评估和测试，并且可以作为一种娱乐手段。这种方法为检测代理交互提供了一种无需人工干预的方式，能够有效地测试代理模型。

4. Generative Agents：该项目的目标是通过构建一个复杂的模拟环境，让 25 个不同的代理在这个环境中进行互动，从而实现生成型代理。项目同时也注重处理代理的记忆和反思能力，代理能够通过记忆中的事件来指导下一步的行动，并在反思环节对最近的事件进行评估和更新。这种基于反思的状态更新机制适用于各种类型的记忆，例如实体记忆和知识图谱，从而提高代理对环境的建模能力。


####   最简单的代理示例

首先，让我们加载语言模型。

```
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.llms import OpenAI
```

```python
llm = OpenAI(temperature=0)
```
接下来，让我们加载一些要使用的工具。请注意，llm-math工具使用了一个LLM，所以我们需要传递进去。

```python
tools = load_tools(["serpapi", "llm-math"], llm=llm)
```


最后，让我们用这些工具、语言模型和我们想要使用的代理类型来初始化一个代理。

```python
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
```

现在，让我们来测试一下吧！

```
agent.run("Who is Leo DiCaprio's girlfriend? What is her current age raised to the 0.43 power?")
```

```
    > Entering new AgentExecutor chain...
     I need to find out who Leo DiCaprio's girlfriend is and then calculate her age raised to the 0.43 power.
    Action: Search
    Action Input: "Leo DiCaprio girlfriend"
    Observation: Camila Morrone
    Thought: I need to find out Camila Morrone's age
    Action: Search
    Action Input: "Camila Morrone age"
    Observation: 25 years
    Thought: I need to calculate 25 raised to the 0.43 power
    Action: Calculator
    Action Input: 25^0.43
    Observation: Answer: 3.991298452658078
    
    Thought: I now know the final answer
    Final Answer: Camila Morrone is Leo DiCaprio's girlfriend and her current age raised to the 0.43 power is 3.991298452658078.
    
    > Finished chain.





    "Camila Morrone is Leo DiCaprio's girlfriend and her current age raised to the 0.43 power is 3.991298452658078."
```


#  代理类型

尽管现代的语言模型，如GPT-4，已经极其强大，能够理解和生成高度复杂和连贯的文本，然而，这些模型往往还无法独立完成特定的任务。这就是代理的作用。代理是一种灵活的接口，允许语言模型和其他工具之间形成一个灵活的调用链，来解决特定的问题或任务。这种灵活性，使得语言模型可以适应更多不同的应用场景，实现其单独无法完成的功能。

具体来说，代理拥有一整套工具的访问权限，它可以根据用户的输入来决定使用哪些工具。一个代理可以使用多种工具，甚至可以把一个工具的输出作为下一个工具的输入。通过这种方式，代理将工具和语言模型有机地结合在一起，实现了高度复杂和特定的任务。

根据任务的不同，代理主要有两种类型：行动代理和计划执行代理。

行动代理：每个时间步都会根据前面所有动作的输出来决定下一步的行动。行动代理适合于小型任务，它的优点在于能够实时地处理信息和作出决策。

计划执行代理：首先决定全部行动序列，然后一次性执行所有的动作，而不更新计划。计划执行代理更适合于需要保持长期目标和重点的复杂或长期运行的任务。

值得注意的是，常常将这两种代理结合起来使用是最佳的做法，也就是说，让计划执行代理使用行动代理来执行计划。这样的结合，既保持了行动代理的动态性，又利用了计划执行代理的规划能力，从而最大化地发挥了代理的功能和效用。


####   行动代理

代理利用语言大模型（LLM）决定要执行哪些行动以及行动的顺序。行动可以是使用工具并观察其输出，也可以是向用户返回响应。

首先，让我们查看源码里代理的类型。

```
class AgentType(str, Enum):
    ZERO_SHOT_REACT_DESCRIPTION = "zero-shot-react-description"
    REACT_DOCSTORE = "react-docstore"
    SELF_ASK_WITH_SEARCH = "self-ask-with-search"
    CONVERSATIONAL_REACT_DESCRIPTION = "conversational-react-description"
    CHAT_ZERO_SHOT_REACT_DESCRIPTION = "chat-zero-shot-react-description"
    CHAT_CONVERSATIONAL_REACT_DESCRIPTION = "chat-conversational-react-description"
    STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION = (
        "structured-chat-zero-shot-react-description"
    )
    OPENAI_FUNCTIONS = "openai-functions"
```    

1. Zero-shot ReAct
这种代理使用ReAct框架，仅根据工具的描述决定使用哪个工具。可以提供任意数量的工具。这种代理要求为每个工具提供一个描述。

2. 结构化输入反应（Structured input ReAct）
结构化工具聊天代理能够使用多输入工具。旧代理被配置为将行动输入指定为单个字符串，但是这种代理可以使用工具的参数模式创建结构化的行动输入。这对于更复杂的工具使用，例如精确导航浏览器，非常有用。

3. OpenAI函数（OpenAI Functions）
某些OpenAI模型（如gpt-3.5-turbo-0613和gpt-4-0613）被明确地微调，以便检测何时应该调用函数，并响应应传递给函数的输入。OpenAI函数代理旨在与这些模型一起工作。

4. 对话（Conversational）
这种代理被设计用于对话设置。提示被设计为使代理有帮助和交谈。它使用ReAct框架决定使用哪个工具，并使用记忆来记住以前的对话交互。

5. 自问与搜索（Self ask with search）
这种代理使用一个名为中间答案的工具。这个工具应该能够查找问题的事实答案。这个代理相当于原始的self ask with search paper论文，其中提供了Google搜索API作为工具。

6. ReAct文档存储（ReAct document store）
这种代理使用ReAct框架与文档存储进行交互。必须提供两个工具：搜索工具和查找工具（必须精确地命名为这样）。搜索工具应该搜索一个文档，而查找工具应该在最近找到的文档中查找一个词条。这个代理等同于原始的ReAct论文，特别是维基百科的例子。

####   计划并执行代理

紧接着行动代理的类型，让我们继续了解计划并执行代理的类型。

计划并执行代理（Plan-and-Solve Agents）通过首先规划要做什么，然后执行子任务来实现目标。这个思想很大程度上受到BabyAGI以及"Plan-and-Solve"论文的启发。

每种代理类型都有其特定的用途和应用场景，代理的灵活性和丰富性为LangChain提供了强大的功能性。

# 代理的使用

我们通过一个实际的应用案例代码，解释如何使用代理和工具。

####   设置Agent

我们先安装要用到的库。

```
pip -q install langchain huggingface_hub openai google-search-results tiktoken wikipedia
```

设置密钥。

```
import os
os.environ["OPENAI_API_KEY"] = ""
os.environ["SERPAPI_API_KEY"] = ""
```

设置Agent的过程包含了两个主要步骤：加载Agent将使用的工具，然后用这些工具初始化Agent。在代码示例中，我们首先初始化了一些基础设置，然后加载了两个工具：一个使用搜索API进行搜索的工具，以及一个可以进行数学运算的计算器工具。

加载工具和初始化Agent。
```
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.llms import OpenAI

llm = OpenAI(temperature=0)
```

每个工具都有一个名称和描述，告诉我们它是用来做什么的。在我们的示例中，“serpapi”工具用于搜索，而“llm-math”工具则用于解决数学问题。这些工具内部有很多内容，包括模板和许多不同的chains。

```
tools = load_tools(["serpapi", "llm-math"], llm=llm)
```

####   初始化Agent

一旦我们设置好了工具，我们就可以开始初始化Agent。初始化Agent需要我们传入工具和语言模型，以及Agent的类型或风格。在我们的示例中，我们使用了零镜像反应性Agent，这是基于一篇关于让语言模型采取行动并生成操作步骤的论文。

```
agent = initialize_agent(tools, 
                         llm, 
                         agent="zero-shot-react-description", 
                         verbose=True)
```     


####   Agent的提示

初始化Agent的重要步骤之一是设置执行器的提示。这些提示会在Agent开始运行时提示语言模型，告诉它应该做什么。

```
agent.agent.llm_chain.prompt.template
```

在我们的示例中，我们为Agent设置了两个工具：搜索引擎和计算器。然后，我们设置了Agent应该返回的格式，这包括它需要回答的问题，以及它应该采取的行动和行动的输入。

```
'Answer the following questions as best you can. You have access to the following tools:\n\nSearch: A search engine. Useful for when you need to answer questions about current events. Input should be a search query.\nCalculator: Useful for when you need to answer questions about math.\n\nUse the following format:\n\nQuestion: the input question you must answer\nThought: you should always think about what to do\nAction: the action to take, should be one of [Search, Calculator]\nAction Input: the input to the action\nObservation: the result of the action\n... (this Thought/Action/Action Input/Observation can repeat N times)\nThought: I now know the final answer\nFinal Answer: the final answer to the original input question\n\nBegin!\n\nQuestion: {input}\nThought:{agent_scratchpad}'
```

####   Agent的运行

最后，我们运行Agent。需要注意的是，Agent并不总是需要使用工具。在我们的示例中，我们问Agent "你今天好吗？"。对于这样的问题，Agent并不需要进行搜索或计算，而是可以直接生成回答。

```
agent.run("Hi How are you today?")
```

这就是Langchain Agents的基本概念和使用方法。


####   使用Math模块


我们在前半部分介绍了Langchain agents（代理）的基础知识和功能。现在，我们要继续探讨如何在实际中应用Agent，以及在某些情况下，Agent可能遇到的问题。


```
agent.run("Where is DeepMind's office?")
```

在我们的示例中，我们尚未使用到math模块，让我们来看一下它的作用。我们让Agent查找Deep Mind的街道地址的数字，然后进行平方。

```
agent.run("If I square the number for the street address of DeepMind what answer do I get?")
```

Agent首先进行搜索获取地址，然后找到了数字5（假设为地址的一部分），最后进行平方运算，得出结果25。然而，如果问题中包含多个数字，Agent可能会对哪个数字进行平方产生混淆，这就是一些可能需要考虑和解决的问题。

```
> Entering new AgentExecutor chain...
 I need to find the street address of DeepMind first.
Action: Search
Action Input: "DeepMind street address"
Observation: DeepMind Technologies Limited, is a company organised under the laws of England and Wales, with registered office at 5 New Street Square, London, EC4A 3TW (“DeepMind”, “us”, “we”, or “our”). DeepMind is a wholly owned subsidiary of Alphabet Inc. and operates https://deepmind.com (the “Site”).
Thought: I now need to calculate the square of the street address.
Action: Calculator
Action Input: 5^2
Observation: Answer: 25
Thought: I now know the final answer.
Final Answer: 25

> Finished chain.
'25'
```

####   使用终端工具

在我们的工具库中，还有一个我们还未使用过的工具，那就是终端工具。例如，我们可以问Agent当前目录中有哪些文件。

```
agent.run("What files are in my current directory?")
```

Agent将运行一个LS命令来查看文件夹，并返回一个文件列表。

```

> Entering new AgentExecutor chain...
 I need to find out what files are in my current directory.
Action: Terminal
Action Input: ls
Observation: sample_data

Thought: I need to find out more information about this file.
Action: Terminal
Action Input: ls -l sample_data
Observation: total 55504
-rwxr-xr-x 1 root root     1697 Jan  1  2000 anscombe.json
-rw-r--r-- 1 root root   301141 Mar 10 20:51 california_housing_test.csv
-rw-r--r-- 1 root root  1706430 Mar 10 20:51 california_housing_train.csv
-rw-r--r-- 1 root root 18289443 Mar 10 20:51 mnist_test.csv
-rw-r--r-- 1 root root 36523880 Mar 10 20:51 mnist_train_small.csv
-rwxr-xr-x 1 root root      930 Jan  1  2000 README.md

Thought: I now know the final answer.
Final Answer: The files in my current directory are anscombe.json, california_housing_test.csv, california_housing_train.csv, mnist_test.csv, mnist_train_small.csv, and README.md.

> Finished chain.
'The files in my current directory are anscombe.json, california_housing_test.csv, california_housing_train.csv, mnist_test.csv, mnist_train_small.csv, and README.md.'
```



我们甚至可以问它特定的文件是否存在，比如询问当前目录中是否有一个关于加利福尼亚的文件。

```
agent.run("Does my current directory have a file about California?")
```

Agent会使用grep命令进行检索，并向我们报告结果。然而，这个过程可能需要多次调用语言模型，从而产生一定的运行成本。

```
> Entering new AgentExecutor chain...
 I need to find out what files are in my current directory.
Action: Terminal
Action Input: ls
Observation: sample_data

Thought: I need to find out if there is a file about California in this directory.
Action: Terminal
Action Input: grep California sample_data
Observation: Command 'grep California sample_data' returned non-zero exit status 2.
Thought: I need to look for a different way to search for the file.
Action: Search
Action Input: "California file in current directory"
Observation: The pwd command displays the full, absolute path of the current, or working, directory. It's not something you'll use all the time, but it can ...
Thought: I need to use the pwd command to find the file.
Action: Terminal
Action Input: pwd
Observation: /content

Thought: I need to use the ls command to find the file.
Action: Terminal
Action Input: ls
Observation: sample_data

Thought: I need to use the grep command to find the file.
Action: Terminal
Action Input: grep California sample_data
Observation: Command 'grep California sample_data' returned non-zero exit status 2.
Thought: I need to look for a different way to search for the file.
Action: Search
Action Input: "California file in current directory"
Observation: The pwd command displays the full, absolute path of the current, or working, directory. It's not something you'll use all the time, but it can ...
Thought: I need to use the find command to find the file.
Action: Terminal
Action Input: find . -name "*California*"
Observation: Command 'find . -name "*California*' returned non-zero exit status 2.
Thought: I need to use the grep command to find the file.
Action: Terminal
Action Input: grep -R California *
Observation: sample_data/README.md:*   `california_housing_data*.csv` is California housing data from the 1990 US

Thought: I now know the final answer.
Final Answer: Yes, there is a file about California in the current directory.

> Finished chain.
'Yes, there is a file about California in the current directory.'
```

####   注意事项

使用终端工具时，需要非常谨慎。你不希望最终用户能够通过运行终端命令来操作你的文件系统，因此在添加这个工具时，需要确保适当的安全防护措施已经到位。不过，尽管有其潜在风险，但在某些情况下，使用终端工具还是很有帮助的，比如当你需要设置某些功能时。


以上就是Langchain agents的一些主要特点和应用示例。
# 6.2.1 自定义代理

这一节我们介绍了如何创建自定义代理（agent）。

一个代理由两部分组成：

- 工具（tools）：代理可以使用的工具。
- 代理类本身：决定采取哪种行动。

我们将逐步介绍如何创建一个自定义代理。

Tool，AgentExecutor，BaseSingleActionAgent是从langchain.agents模块导入的类，用于创建自定义的Agent和工具。OpenAI和SerpAPIWrapper是从langchain模块导入的类，用于访问OpenAI的功能和SerpAPI的包。

```python
from langchain.agents import Tool, AgentExecutor, BaseSingleActionAgent
from langchain import OpenAI, SerpAPIWrapper
```
创建一个SerpAPIWrapper实例，然后将其run方法封装到一个Tool对象中。

```python
search = SerpAPIWrapper()
tools = [
    Tool(
        name="Search",
        func=search.run,
        description="useful for when you need to answer questions about current events",
        return_direct=True,
    )
]
```

定义了一个自定义的Agent类FakeAgent，这个类从BaseSingleActionAgent继承。该类定义了两个方法plan和aplan，这两个方法是Agent根据给定的输入和中间步骤来决定下一步要做什么的核心逻辑。

```python
from typing import List, Tuple, Any, Union
from langchain.schema import AgentAction, AgentFinish


class FakeAgent(BaseSingleActionAgent):
    """Fake Custom Agent."""

    @property
    def input_keys(self):
        return ["input"]

    def plan(
        self, intermediate_steps: List[Tuple[AgentAction, str]], **kwargs: Any
    ) -> Union[AgentAction, AgentFinish]:
        """Given input, decided what to do.

        Args:
            intermediate_steps: Steps the LLM has taken to date,
                along with observations
            **kwargs: User inputs.

        Returns:
            Action specifying what tool to use.
        """
        return AgentAction(tool="Search", tool_input=kwargs["input"], log="")

    async def aplan(
        self, intermediate_steps: List[Tuple[AgentAction, str]], **kwargs: Any
    ) -> Union[AgentAction, AgentFinish]:
        """Given input, decided what to do.

        Args:
            intermediate_steps: Steps the LLM has taken to date,
                along with observations
            **kwargs: User inputs.

        Returns:
            Action specifying what tool to use.
        """
        return AgentAction(tool="Search", tool_input=kwargs["input"], log="")
```

创建了一个FakeAgent的实例。

```python
agent = FakeAgent()
```

创建了一个AgentExecutor的实例，该实例将使用前面定义的FakeAgent和tools来执行任务。
```python
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent, tools=tools, verbose=True
)
```
调用AgentExecutor的run方法来执行一个任务，任务是查询"2023年加拿大有多少人口"。

```python
agent_executor.run("How many people live in canada as of 2023?")
```
打印最终的结果。
```
    
    
    > Entering new AgentExecutor chain...
    The current population of Canada is 38,669,152 as of Monday, April 24, 2023, based on Worldometer elaboration of the latest United Nations data.
    
    > Finished chain.





    'The current population of Canada is 38,669,152 as of Monday, April 24, 2023, based on Worldometer elaboration of the latest United Nations data.'
```    

# 代理实现ReAct

由于ReAct策略的特性，目前它已经成为首选的代理实现方式。代理的基本理念是将语言模型当作推理的引擎。ReAct策略实际上是把推理和动作有机地结合在一起。当代理接收到用户的请求后，语言模型就会帮助选择使用哪个工具。接着，代理会执行该工具的操作，观察其结果，并把这些结果反馈给语言模型。

下面用代码演示了如何使用代理实现ReAct策略。

首先，让我们加载将用于控制代理的语言模型。

```python
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.llms import OpenAI
llm = OpenAI(temperature=0)
```

接着，我们需要加载一些工具。

```python
tools = load_tools(["serpapi", "llm-math"], llm=llm)
```

请注意，llm-math工具使用了LLM，因此我们需要输入这个模型。

最后，我们需要使用工具、语言模型和我们想要使用的代理类型来初始化一个代理。

```python
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
```


现在让我们来测试一下！

```python
agent.run("Who is Leo DiCaprio's girlfriend? What is her current age raised to the 0.43 power?")
```

```
    > Entering new AgentExecutor chain...
     I need to find out who Leo DiCaprio's girlfriend is and then calculate her age raised to the 0.43 power.
    Action: Search
    Action Input: "Leo DiCaprio girlfriend"
    Observation: Camila Morrone
    Thought: I need to find out Camila Morrone's age
    Action: Search
    Action Input: "Camila Morrone age"
    Observation: 25 years
    Thought: I need to calculate 25 raised to the 0.43 power
    Action: Calculator
    Action Input: 25^0.43
    Observation: Answer: 3.991298452658078
    
    Thought: I now know the final answer
    Final Answer: Camila Morrone is Leo DiCaprio's girlfriend and her current age raised to the 0.43 power is 3.991298452658078.
    
    > Finished chain.


    "Camila Morrone is Leo DiCaprio's girlfriend and her current age raised to the 0.43 power is 3.991298452658078."
```

除此之外，你还可以创建使用聊天模型作为代理驱动器的ReAct代理，而不是使用LLM。

```python
from langchain.chat_models import ChatOpenAI

chat_model = ChatOpenAI(temperature=0)
agent = initialize_agent(tools, chat_model, agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
agent.run("Who is Leo DiCaprio's girlfriend? What is her current age raised to the 0.43 power?")
```
# 工具和工具包概述

####   工具

在人工智能代理系统中，工具是代理用来与世界互动的接口。这些工具实际上就是代理可以使用的函数，用于与外部世界进行交互。这些工具可以是通用的实用程序（例如搜索功能），也可以是其他的工具链，甚至是其他的代理。

####   工具包

工具包（Toolkits）是设计用于完成特定任务的工具集合，它们具有方便的加载方法。工具包将一组具有共同目标或特性的工具集中在一起，提供统一而便捷的使用方式，使得用户能够更加方便地完成特定任务。

####   工具的定义方法

在构建自己的代理时，你需要提供一个工具列表，这些工具是代理可以使用的。除了实际被调用的函数外，工具还包括一些组成部分：名称（必需的，并且在提供给代理的工具集中必须是唯一的）；描述（可选的，但建议提供，因为代理会用它来判断工具的使用情况）；直接返回（默认为False）；参数模式（可选的，但建议提供，可以用来提供更多的信息（例如，少数样本的例子）或者对预期参数进行验证）。

####   工具包的应用

工具包是为特定任务而设计的工具集合，具有便捷的加载方法。它们可以被设计为一起使用，以完成特定的任务。这种设计方式提供了一种更为便捷和高效的方式来处理复杂的任务，提升了工作效率。

####   LangChain封装的工具

LangChain封装了许多工具，用户可以随时调用这些工具，完成各种复杂的任务。除了使用LangChain提供的工具，用户也可以自定义工具，形成自己的工具包，以完成特殊的任务。这种灵活性使得LangChain成为了一个强大而灵活的工具，能够满足各种复杂的任务需求。
# 工具的类型

Langchain 提供了一系列的工具，它们封装了各种功能，可以直接在你的项目中使用。这些工具涵盖了从数据处理到网络请求，从文件操作到数据库查询，从搜索引擎查询到人工智能模型的应用，等等。这个工具清单还在不断地扩展和更新。以下是目前可用的工具列表：

1. "AIPluginTool": 一个插件工具，允许用户将其他的人工智能模型或服务集成到系统中。
2. "APIOperation": 用于调用外部API的工具。
3. "ArxivQueryRun": 用于查询Arxiv的工具。
4. "AzureCogsFormRecognizerTool": 利用Azure认知服务中的表单识别器的工具。
5. "AzureCogsImageAnalysisTool": 利用Azure认知服务中的图像分析的工具。
6. "AzureCogsSpeech2TextTool": 利用Azure认知服务中的语音转文本的工具。
7. "AzureCogsText2SpeechTool": 利用Azure认知服务中的文本转语音的工具。
8. "BaseGraphQLTool": 用于发送GraphQL查询的基础工具。
9. "BaseRequestsTool": 用于发送HTTP请求的基础工具。
10. "BaseSQLDatabaseTool": 用于与SQL数据库交互的基础工具。
11. "BaseSparkSQLTool": 用于执行Spark SQL查询的基础工具。
12. "BingSearchResults": 用于获取Bing搜索结果的工具。
13. "BingSearchRun": 用于执行Bing搜索的工具。
14. "BraveSearch": 用于执行Brave搜索的工具。
15. "ClickTool": 模拟点击操作的工具。
16. "CopyFileTool": 用于复制文件的工具。
17. "CurrentWebPageTool": 用于获取当前网页信息的工具。
18. "DeleteFileTool": 用于删除文件的工具。
19. "DuckDuckGoSearchResults": 用于获取DuckDuckGo搜索结果的工具。
20. "DuckDuckGoSearchRun": 用于执行DuckDuckGo搜索的工具。
21. "ExtractHyperlinksTool": 用于从文本或网页中提取超链接的工具。
22. "ExtractTextTool": 用于从文本或其他源中提取文本的工具。
23. "FileSearchTool": 用于搜索文件的工具。
24. "GetElementsTool": 用于从网页或其他源中获取元素的工具。
25. "GmailCreateDraft": 用于创建Gmail草稿的工具。
26. "GmailGetMessage": 用于获取Gmail消息的工具。
27. "GmailGetThread": 用于获取Gmail线程的工具。
28. "GmailSearch": 用于搜索Gmail的工具。
29. "GmailSendMessage": 用于发送Gmail消息的工具。
30. "GooglePlacesTool": 用于搜索Google Places的工具。
31. "GoogleSearchResults": 用于获取Google搜索结果的工具。
32. "GoogleSearchRun": 用于执行Google搜索的工具。
33. "GoogleSerperResults": 用于获取Google SERP（搜索引擎结果页面）的工具。
34. "GoogleSerperRun": 用于执行Google SERP查询的工具。
35. "HumanInputRun": 用于模拟人类输入的工具。
36. "IFTTTWebhook": 用于触发IFTTT（如果这个，那么那个）Webhook的工具。
37. "InfoPowerBITool": 用于获取PowerBI信息的工具。
38. "InfoSQLDatabaseTool": 用于获取SQL数据库信息的工具。
39. "InfoSparkSQLTool": 用于获取Spark SQL信息的工具。
40. "JiraAction": 用于在Jira上执行操作的工具。
41. "JsonGetValueTool": 用于从JSON数据中获取值的工具。
42. "JsonListKeysTool": 用于列出JSON数据中的键的工具。
43. "ListDirectoryTool": 用于列出目录内容的工具。
44. "ListPowerBITool": 用于列出PowerBI信息的工具。
45. "ListSQLDatabaseTool": 用于列出SQL数据库信息的工具。

请注意，这些工具的具体实现和功能可能会根据实际的需求和环境进行调整。
# 工具包
Langchain 提供了一系列与各种代理进行交互的工具包，以帮助我们快速建立解决各种问题的代理。以下是目前可用的工具包列表：

1. "create_json_agent": 一个设计用于与 JSON 数据交互的代理。
2. "create_sql_agent": 一个设计用于与 SQL 数据库交互的代理。
3. "create_openapi_agent": 一个设计用于与 OpenAPI 交互的代理。
4. "create_pbi_agent": 一个设计用于与 Power BI 交互的代理。
5. "create_pbi_chat_agent": 一个设计用于与 Power BI 聊天交互的代理。
6. "create_python_agent": 一个设计用于与 Python 交互的代理。
7. "create_vectorstore_agent": 一个设计用于与 Vector Store 交互的代理。
8. "JsonToolkit": 一个用于处理 JSON 数据的工具包。
9. "SQLDatabaseToolkit": 一个用于处理 SQL 数据库的工具包。
10. "SparkSQLToolkit": 一个用于处理 Spark SQL 的工具包。
11. "NLAToolkit": 一个用于处理自然语言应用的工具包。
12. "PowerBIToolkit": 一个用于处理 Power BI 应用的工具包。
13. "OpenAPIToolkit": 一个用于处理 OpenAPI 的工具包。
14. "VectorStoreToolkit": 一个用于处理 Vector Store 的工具包。
15. "create_vectorstore_router_agent": 一个设计用于与 Vector Store 路由交互的代理。
16. "VectorStoreInfo": 一个用于获取 Vector Store 信息的工具。
17. "VectorStoreRouterToolkit": 一个用于处理 Vector Store 路由的工具包。
18. "create_pandas_dataframe_agent": 一个设计用于与 Pandas 数据帧交互的代理。
19. "create_spark_dataframe_agent": 一个设计用于与 Spark 数据帧交互的代理。
20. "create_spark_sql_agent": 一个设计用于与 Spark SQL 交互的代理。
21. "create_csv_agent": 一个设计用于与 CSV 文件交互的代理。
22. "ZapierToolkit": 一个用于处理 Zapier 应用的工具包。
23. "GmailToolkit": 一个用于处理 Gmail 应用的工具包。
24. "JiraToolkit": 一个用于处理 Jira 应用的工具包。
25. "FileManagementToolkit": 一个用于文件管理的工具包。
26. "PlayWrightBrowserToolkit": 一个用于处理 PlayWright 浏览器的工具包。
27. "AzureCognitiveServicesToolkit": 一个用于处理 Azure 认知服务的工具包。

这些工具包的具体功能和实现可能会根据实际的需求和环境进行调整。




