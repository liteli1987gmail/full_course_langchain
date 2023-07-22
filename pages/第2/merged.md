1. Langchain 概论

### 1.1.1 为什么需要 Langchain

我们为什么需要 LangChain 呢？

这个问题可以反向思考。我们难道不能只是简单地发送一个 API 请求或模型，然后就算完成了吗？对于简单的应用，确实如此。但一旦你开始增加复杂性，比如将语言模型与你自己的数据（如 Google Analytics、Stripe、SQL、PDF、CSV 等）连接起来，或者使语言模型执行一些操作，比如发送电子邮件、搜索网络或在终端中运行代码，情况就会变得混乱且重复。LangChain 通过提供组件来解决这个问题。

自从 chatgpt 问世以来，LangChain 这个开源框架，已经成为了模型开发中的热门技术。但是硅谷的评论家说 LangChain 只是个“玩具”。

如果没有 Langchain，我们能否开发大语言模型？答案无疑是可以的。但问题在于，为什么在 GPT-4 发布之后，越来越多的开发者选择使用 Langchain 呢？Langchain 的吸引力究竟来自哪里？

### 1.1.2 Langchain 的核心

Langchain 的核心在于它的“组件”和预设的“结构化链”。就像乐高积木一样，Langchain 的组件和链为开发者提供了无限的可能性，使得模型开发变得更加简单和高效。在 LangChain 中，我们可以使用文档加载器从 PDF、HTML 网页等来源加载数据，然后在存储在向量数据库中之前，可以选择使用文本分割器将其分块。在运行时，可以将数据注入到提示模板中，然后作为输入发送给模型。我们还可以使用工具执行一些操作，例如使用输出内容发送电子邮件。这种方式的链是实现这一魔法的方式，我们将组件链接在一起，以完成特定任务。选择 LangChain 意味着你可以轻松地切换到另一个语言模型，以节约成本或享受其他功能，测试另一个向量数据库的功能，或者摄取另一个数据源，只需几行代码即可实现。

你可以把组件看作 Langchain 中的基础单元，它们就像流水线上的工人，每个工人都有明确的工作内容，比如处理数据的输入、输出，或者将数据从一种格式转化为另一种格式。正如同流水线上每个工人的工作都是接力的一部分，这些组件通过协作保证了数据在模型中的正确流动。

那么，结构化链又是什么呢？可以把它想象成一条产品生产线。这条生产线由一系列的组件链接形成，就好像是流水线上的各个工作站。我们只需要按照需求，像选购快餐一样选择不同的链条，就可以完成特定的任务，比如进行文本摘要、窗口记忆，或者会话记忆等。

通过这些组件和链的无限组合，Langchain 可以应对各种复杂场景，就像乐高积木可以拼接出无数种造型。无论是简单的数据处理，还是与现实世界中的软件数据交互，Langchain 都能像万能工具一样胜任。

我们可以举个例子来看看 Langchain 是如何工作的。假设你想让一个语言模型帮你总结一篇 PDF 文档。这似乎是个简单的任务，就像让一个儿童用积木拼出一个小房子一样容易，只需导入文本，然后提示模型进行摘要即可。

但是，如果你要处理 10 个 PDF 文件呢？这就好比你要用积木拼出一个城堡，每个步骤，无论是文档上传、文档切割，还是数据格式转换、查找关联文档，都会变得复杂而繁琐。此时，你或许会意识到 Langchain 的威力，它不再是你眼中的“玩具”，而是一把能助你搭建城堡的神奇工具。

### 1.1.3 模型开发的最后一公里

以最简单的模型交互为例，比如发起一个 get 或 post 的 http 请求，对于初学者来说，这本身就可能是一道难以逾越的门槛。Langchain 的存在，就是为了将这种门槛降低到最低。

这正是 Langchain 的价值所在，只需几行代码，就可以运行一个大语言模型的程序，为开发者们打通了大语言模型开发的最后一公里。

除了代码量的减少，Langchain 还提供了许多预封装的 “轮子”。比如，对于摘要任务，它提供了摘要链。如果你不知道如何编写模型的提示，它提供了各种模板。AI 代理则使我们可以专注于完成任务，而不必分心去处理各种工具。

总的来说，Langchain 解决了两类问题：一是模型与模型间的交流；二是模型与现实世界数据的交互。

如果你采用了 Langchain 的这些组件、链和代理，你只需要学习 Langchain，编写模型的提示，指定要使用的模型厂家，就可以得到结构化的输出，实现与软件世界的无缝连接。你可以在私有环境中部署模型，也可以选择更小、更专业的模型来完成特定的分类任务。

如果你没有使用 Langchain，那么你可能需要独自处理每个模型的接口方式，封装结构化的输入提示，处理输出的格式等问题。每一步都亲历亲为的结果很可能是“从入门到放弃”。

总结起来，无论你是要构建复杂的应用，处理大量数据，还是仅仅是一个初学者，Langchain 都能提供组件和链结合的解决方案，帮助你挖掘大语言模型的无限潜力。
# 1.1.2 环境和密钥配置
# 1.3.1 四行代码开始


使用 LangChain 通常需要集成一个或多个模型提供商，数据存储，API 等。在这个例子中，我们将使用 OpenAI 的模型 API。

首先，我们需要安装他们的 Python 包：

```shell
pip install openai langchain
```

访问 API 需要一个 API 密钥，你可以通过创建一个账户并访问此处获得。一旦我们得到密钥，我们会想要将其设置为环境变量，通过运行：

```shell
export OPENAI_API_KEY=“...”
```

如果你不会设置环境变量，你可以在初始化 OpenAI LLM 类时，直接通过 `openai_api_key` 参数将密钥传入：

```python
from langchain.llms import OpenAI

llm = OpenAI(openai_api_key=“...”)
```

### 构建应用程序

现在我们可以开始构建我们的语言模型应用程序。LangChain 提供了许多可以用来构建语言模型应用程序的模块。模块可以在简单应用程序中单独使用，也可以在更复杂的使用场景中组合使用。

#### LLMs

从语言模型获取预测结果

LangChain 的基本构建块是 LLM，它接收文本并生成更多的文本。

例如，假设我们正在构建一个根据公司描述生成公司名称的应用程序。为了做到这一点，我们需要初始化一个 OpenAI 模型包装器。在这种情况下，因为我们希望输出更随机和创意性，所以我们会用 “temperature” 来初始化我们的模型。

值得注意的是，模型的 `temperature` 参数。`temperature` 的值范围在 0-1 之间。此参数用于控制生成的文本的随机性。当 `temperature` 设置为 0 时，模型产生的结果会具有更低的随机性，通常更确定性和连贯性。相反，当 `temperature` 设置为 1 时，模型产生的结果会更加随机，这可能会产生更有创新性和多样性的结果。

```python
from langchain.llms import OpenAI

llm = OpenAI(temperature=0.9)
```

现在我们可以输入问题，并且获得答案了！

```python
llm.predict(“What would be a good company name for a company that makes colorful socks?”)
# >> Feetful of Fun
```

#### 聊天模型

聊天模型是语言模型的一种变体。虽然聊天模型在底层使用语言模型，但是它们暴露的接口有所不同：而不是暴露一个 “文本输入，文本输出” 的 API，它们暴露的接口是 “聊天消息” 作为输入和输出。

你可以通过将一个或多个消息传递给聊天模型来获取聊天完成。响应将是一条消息。目前在 LangChain 中支持的消息类型有 AIMessage、HumanMessage、SystemMessage 和 ChatMessage--ChatMessage 接受一个任意的角色参数。大多数情况下，你只会处理 HumanMessage、AIMessage 和 SystemMessage。

```python
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

chat = ChatOpenAI(temperature=0)
chat.predict_messages([HumanMessage(content=“Translate this sentence from English to French. I love programming.”)])
# >> AIMessage(content=“J'aime programmer.”, additional_kwargs={})
```

理解聊天模型与普通 LLM 的区别是有用的，但往往能够将它们视为相同是方便的。LangChain 通过暴露一个接口，你可以通过这个接口与聊天模型交互，就像与普通的 LLM 一样。你可以通过 predict 接口访问这个功能。

```python
chat.predict(“Translate this sentence from English to French. I love programming.”)
# >> J'aime programmer
```
# 1.3.2 核心组件

LangChain 是一个用于开发由语言模型驱动的应用程序的框架。它能让应用程序具备两大特性：连接数据和代理性。

连接数据的应用

那么，什么是连接数据呢？让我们通过一个实际的商业案例来解释。假设你正在管理一个亚马逊的商品，获取用户访问数据、留存和订单取消的原因对你来说是相当简单的。但是，如果我们可以将这些数据发送给模型进行分析，编写营销挽留的邮件，并自动发送到每个用户的邮箱，那么这个过程就变得非常有价值。比如，如果模型分析出订单取消的原因是价格太高，我们可以选择发送分期付款的策略邮件；如果是没有优惠，我们可以让模型调用优惠券接口，发送优惠券给用户。这样的自动化流程，每分每秒都在为商家创造商业利润。

LangChain 的代理性

LangChain 还具有代理性，通过提供结构化的组件和链，使得语言模型能够像电影《HER》中的虚拟助理一样，能够安排行程和制定生活计划。

组件分类

LangChain 提供了标准、可扩展的接口和外部集成，用于以下组件（从简单到复杂排序）：

- 模型输入输出 （Model I/O）：与语言模型进行交互。
- 数据连接（Data connection）：与特定应用的数据进行交互。
- 链（Chains）：构造调用序列。
- 代理 （Agents）：让链选择给定高级指令要使用的工具。
- 记忆（Memory）：在链的运行之间持久化应用状态。
- 回调 （Callbacks）：记录并传送任何链的中间步骤。

模型输入输出

在模型交互中，有两种不同的子类型：LLMs 和 Chat Models。LLMs 封装了接受文本输入并返回文本输出的 API，而 Chat Models 封装了接受聊天消息输入并返回聊天消息输出的模型。尽管它们之间存在细微差别，但使用它们的接口是相同的。我们可以导入这两个类，实例化它们，然后在这两个类上使用 predict 函数并观察它们之间的区别。然而，在实际应用中，我们可能不会直接将文本传递给模型，而是使用提示（prompts）。

数据连接组件

Data connection 组件提供了文档加载，文本切割，文本嵌入，向量管理，信息提取等工具，把私有化的数据发送给模型处理，让模型拥有现实世界的数据的能力。例如，如果你想让模型阅读一份金融研报，通过使用这个组件的一系列方法，我们可以轻松解决数据连接的问题。

链组件

Chains 链组件是 LangChain 中最重要的部分，理念是 “万物皆可链”。无论你是语言模型，还是文档处理，还是代理处理计划和日程，都可以通过链的方式进行。

代理组件

组件是 AI 向通用性跳转的桥梁。Agents 就像人一样，可以组织链条有序地工作。

记忆组件

记忆组件可以让不同的链保持状态，记住上下文，知道我们的人称缩写指的是谁，甚至在几天之后，机器人还能记得你曾经说过的话。

回调组件

回调组件主要是服务链的调用，记录链发生的中间步骤。通过它，我们可以清楚地知道每个链在执行过程中发生了什么，以便进行优化和调整。

总的来说，无论你是构建复杂应用，处理大量数据，还是初学者，LangChain 都能提供组件和链结合的解决方案，助你轻松驾驭大语言模型的无限潜力。
# 1.3.3 解决大模型先天缺陷


#### GPT-4 的挑战与局限

尽管 GPT-4 被广泛认为是人工智能领域的里程碑之一，但仍存在一些明显的限制。让我们详细地分析一下这些挑战：

**知识库依赖性**：GPT-4 并未包含实时更新的真实世界知识库，因此其处理和生成响应的能力主要依赖于训练数据。这意味着 GPT-4 可能无法理解或处理一些特定的、需要依赖现有知识库的问题。

**上下文理解**： GPT-4 的上下文理解能力有限，它主要依靠在训练数据中学习到的上下文信息来生成响应。这可能导致 GPT-4 在理解隐含的上下文或进行复杂的推理时出现困难。

**对话一致性**：GPT-4 可能在保持对话一致性方面存在挑战，可能会在对话的不同部分之间出现语法、词汇和主题的不一致性。

#### LangChain 的解决方案

针对 GPT-4 的这些挑战，LangChain 提出了一系列有效的解决方案：

**数据连接组件**：为解决 GPT-4 的知识库依赖性问题，LangChain 的数据连接组件提供了一系列功能，例如文档上传、文本嵌入，向量库存储、文档提取等，这样可以实时更新知识库，提供更准确和多样的信息源。例如，如果用户在处理法律相关问题时需要引用最新的法规，LangChain 可以通过数据连接组件，将最新的法规上传并嵌入到向量库中，从而模型能够输出最新、最准确的法规信息。

**实时信息获取**：LangChain 整合了 API 工具链，允许模型搜索新闻、获取天气信息、地理位置等实时信息，大大提升了模型的实用性和准确性。比如，如果用户需要获取纽约的实时天气信息，LangChain 可以通过调用天气 API，获取并提供准确的天气信息。

**数学计算能力**：GPT-4 对数学计算不擅长。为弥补这一缺点，LangChain 封装了终端工具，使得模型可以调用 Python 算数，还有继承了 Program-aided Language Model (PAL) 数据模型。例如，如果用户问 “两个千亿相加等于多少”，LangChain 可以利用其终端工具调用 Python 的数学库进行计算，然后将计算结果返回给用户。

**对话一致性**：GPT-4 在保持对话一致性方面可能存在困难。LangChain 的记忆组件可以保存聊天记忆或者提取聊天关键信息，让模型的对话记住我们聊天的时候，人称代词缩写，人物对应，保持对话的一致性，获取像真人一样的聊天体验。例如，如果在聊天过程中引入了新的人物或主题，LangChain 可以提取和存储这些关键信息，以便在后续的对话中使用，从而保持对话的连贯性和一致性。

通过以上的改进，LangChain 在实用性、准确性以及问题处理能力等方面对 GPT-4 进行了有效的补充和优化，使其成为了一个更加强大、灵活且实用的 AI 开发框架。

# 1.4.1 Langchain 开发流程

基于你的要求和提供的原材料，这里是我为你撰写的技术说明书：

# LangChain：构建先进的聊天模型

## 1. 模型适配与扩展

在 LangChain 的开发初期，我们了解到需要与 GPT4 等先进的语言模型保持同步，并利用它们的强大功能。为此，我们主动将模型输入和输出进行了扩展，使之更加适合用于聊天环境。我们的代码使用的模型都是聊天模型，这些模型既能适配最新的 GPT4，也能与早期版本如 GPT-turbo-3。5 兼容。

## 2. 聊天模型的特点

聊天模型是语言模型的一种变体。尽管聊天模型内部使用的是语言模型，但它们提供的接口有所不同：聊天模型并不是简单地提供 “输入文本，输出文本” 的 API，而是将聊天信息作为输入和输出。你可以通过向聊天模型传递一个或多个消息来获取聊天完成。LangChain 目前支持的消息类型包括 AIMessage、HumanMessage、SystemMessage 和 ChatMessage——ChatMessage 可以接受任意的角色参数。在大多数情况下，你将主要处理 HumanMessage、AIMessage 和 SystemMessage。

## 3. 构建全功能聊天机器人

我们的目标是创建第一个可以陪伴用户聊天、回答天气、实时搜索的聊天机器人。通过构建这样的机器人，我们可以让 LangChain 的主要模块都得到应用。聊天机器人不仅可以提供丰富的信息服务，还能让用户获得自然、流畅的对话体验，从而让 AI 更好地融入到人们的日常生活中。

LangChain 正在不断发展和创新，我们希望通过提供高效的聊天模型和强大的功能，来为用户带来更多的便利和乐趣。
# 1.4.2 创建你的第一个聊天机器人

在 LangChain 的开发之前，我们需要与 GPT4 等先进的语言模型保持同步，并利用它们的强大功能。为此，LangChain 主动将模型输入和输出进行了扩展，使之更加适合用于聊天环境。我们的代码使用的模型都是聊天模型（Chat model），最新的 GPT-turbo-3。5 和 GPT4 都是运用的这种聊天模型接口。

聊天模型的特点

聊天模型是语言模型的一种变体。尽管聊天模型内部使用的是语言模型，但它们提供的接口有所不同：聊天模型并不是简单地提供 “输入文本，输出文本” 的 API，而是将聊天信息作为输入和输出。你可以通过向聊天模型传递一个或多个消息来获取聊天完成。LangChain 目前支持的消息类型包括 AIMessage、HumanMessage、SystemMessage 和 ChatMessage——ChatMessage 可以接受任意的角色参数。在大多数情况下，你将主要处理 HumanMessage、AIMessage 和 SystemMessage。

构建全功能聊天机器人

我们的目标是创建第一个可以陪伴用户聊天、回答天气、实时搜索的聊天机器人。通过构建这样的机器人，我们可以让 LangChain 的主要模块都得到应用。聊天机器人不仅可以提供丰富的信息服务，还能让用户获得自然、流畅的对话体验，从而让 AI 更好地融入到人们的日常生活中。



# 1.4.3 开始翻译

首先，我们需要安装他们的 Python 包：

```shell
pip install openai langchain
```

访问 API 需要一个 API 密钥，你可以通过创建一个账户并访问此处获得。一旦我们得到密钥，我们会想要将其设置为环境变量，通过运行：

```shell
export OPENAI_API_KEY=“...”
```
#### 聊天模型

LangChain 的 schema 定义了 AIMessage、HumanMessage 和 SystemMessage 这三种角色类型的数据模式。这些都是我们设计的数据模型，通过这些模型，我们可以像使用函数一样将参数传递给它们。

例如，如果我们想要与聊天机器人进行对话，我们只需要把想要说的话用 HumanMessage 函数封装起来，像这样：`HumanMessage(content="你好!")`。然后我们将这个消息放入一个列表中，传递给 ChatModel 模型。这样，我们就可以开始与聊天机器人进行交流了。

如果我们想要使用这个聊天机器人来翻译一段英文，我们可以这样编写代码：

```python
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

chat = ChatOpenAI(temperature=0)
chat.predict_messages([HumanMessage(content=“Translate this sentence from English to French. I love programming.”)])
# 输出：AIMessage(content=“J'aime programmer.”, additional_kwargs={})
```

在这段代码中，我们首先导入了需要的模块和函数。然后，我们创建了一个 ChatOpenAI 对象，并且设置了温度参数为 0，这意味着模型的输出将会具有更低的随机性。之后，我们调用 `chat.predict_messages` 方法，向它传递了一个包含 HumanMessage 对象的列表。这个 HumanMessage 对象包含了我们想要翻译的英文句子。最后，我们的模型将返回一个 AIMessage 对象，它包含了这句英文的法文翻译。
```

#### 提示模板

提示模板是一种特殊的文本，它可以为特定任务提供额外的上下文信息。在大语言模型（LLM）的应用中，通常并不直接将用户的输入传递给LLM，而是将用户输入添加到一个更大的文本中，即提示模板。提示模板为当前的具体任务提供了额外的上下文信息，这能够更好地引导模型生成预期的输出。

如何使用提示模板？

在LangChain中，我们可以使用MessagePromptTemplate来创建提示模板。我们可以从一个或多个MessagePromptTemplates创建一个ChatPromptTemplate。示例代码如下：

```
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

template = “You are a helpful assistant that translates {input_language} to {output_language}.”
system_message_prompt = SystemMessagePromptTemplate.from_template(template)
human_template = “{text}”
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

chat_prompt.format_messages(input_language = “English”, output_language = “French”, text = “I love programming.”)

```

在上述代码中，我们首先定义了两种模板：一个是系统消息模板，描述了任务的上下文（翻译助手的角色和翻译任务）；另一个是人类消息模板，这将会是用户的输入。

然后，我们使用ChatPromptTemplate。from_messages方法，将这两个模板结合起来，生成了一个聊天提示模板。

当我们需要生成预期的输出时，我们可以调用ChatPromptTemplate的format_messages方法，像这样：

```
[
    SystemMessage(content = “You are a helpful assistant that translates English to French.”, additional_kwargs ={}),
    HumanMessage(content = “I love programming.”)
]
```
通过这种方式，我们不仅可以生成预期的输出，还能让用户无需担心提供模型指令，他们只需要提供具体的任务信息即可。

#### 创建第一个链

现在，让我们将上述步骤整合为一条链，以此创建我们的第一个链。我们将使用LangChain的LLMChain（大语言模型链）对模型进行包装，实现与提示模板类似的功能。这种方式更为直观易懂，你会发现我们导入了一个包装链 LLMChain， 将提示模板和模型传递进去后，我们就造好了链。而链的运行只要 run 一下。以下是相关代码：

``` python
from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

# 初始化 ChatOpenAI 聊天模型，温度设置为 0
chat = ChatOpenAI(temperature = 0)

# 定义系统消息的模板
template = “You are a helpful assistant that translates {input_language} to {output_language}.”
system_message_prompt = SystemMessagePromptTemplate.from_template(template)

# 定义人类消息的模板
human_template = “{text}”
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

# 将这两种模板组合到聊天提示模板中
chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

# 使用 LLMChain 包装模型和提示模板
chain = LLMChain(llm = chat, prompt = chat_prompt)

# 运行模型链，传入参数
chain.run(input_language = “English”, output_language = “French”, text = “I love programming.”)
```

在这段代码中，我们首先初始化了一个ChatOpenAI聊天模型，然后定义了系统消息模板和人类消息模板，并将它们组合在一起创建了一个聊天提示模板。接着，我们使用LLMChain来包装我们的聊天模型和聊天提示模板。最后，我们运行了这个模型链，并传入了参数。这样，我们就可以方便地运行模型，并且不需要每次都为提示模板提供所有的参数。
# 1.4.4 开始对话

当我们的生活越来越依赖于各种信息，比如我们可能想要去郊游，需要查询当天的天气状况，路况信息等，这时候，我们的聊天机器人就可以发挥巨大的作用。不仅如此，他甚至可以帮我们制定计划。那么，如何让聊天机器人完成这样的任务呢？这就需要借助 Langchain 的结构化组件：“代理”。 

#### 代理

“代理”在 LangChain 中，是目前最先进的模块，它的主要职责是基于输入的信息，动态地选择执行哪些动作，以及确定这些动作的执行顺序。一个代理会被赋予一些“工具”，这些工具可以执行特定的任务。代理会反复选择一个工具，运行这个工具，观察输出结果，直到得出最终的答案。换句话说，代理就像一个决策者，它决定使用什么工具来获取天气信息，而我们只需要关注它给我们的最终答案。

要创建并加载一个代理，你需要选择以下几个要素：

1. 大语言模型（LLM）或聊天模型：这是驱动代理的语言模型。
2. 工具（Tool）：执行特定任务的函数。例如，谷歌搜索、数据库查询、Python REPL，甚至其它模型链。对于预定义的工具及其规格，可以查看工具文档。
3. 代理名称：引用受支持的代理类的字符串。代理类主要由语言模型用于决定执行哪个动作的提示模板参数化。

在以下的代码示例中，我们将使用 SerpAPI 查询搜索引擎来创建一个代理：

安装库。
```
pip -q install  openai
pip install git+https://github.com/hwchase17/langchain
```
设置密钥。

```
# 设置OpenAI的API密钥
os.environ[“OPENAI_API_KEY”] = “”
# 设置谷歌搜索的API密钥
os.environ[“SERPAPI_API_KEY”] = “”
```

```python
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI

# 首先，加载控制代理的语言模型
chat = ChatOpenAI(temperature=0)

# 加载一些工具，注意这里的`llm-math`工具使用了一个LLM，因此需要将其传入
llm = OpenAI(temperature=0)
tools = load_tools([“serpapi”, “llm-math”], llm=llm)

# 最后，用工具、语言模型以及我们想要使用的代理类型初始化一个代理
agent = initialize_agent(tools, chat, agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

# 现在我们测试一下代理
agent.run(“What will be the weather in Shanghai three days from now?”)
```

通过以上步骤，我们成功创建并运行了一个代理，这个代理能够帮助我们从网络上获取信息，并进行一些数学计算。这样，无论我们想要查询天气、路况，还是计划郊游，我们都可以轻松地通过这个聊天机器人得到所需的信息。

#### 记忆

在此之前，我们制造的机器人虽然已经能使用工具进行搜索，进行数学运算，但它仍然是无状态的。这意味着它无法引用过去的交互，也就无法根据过去的交互理解新的消息。这显然对于聊天机器人来说是不足的，因为我们希望机器人能够理解新消息，并在此基础上理解过去的消息。

为了解决这个问题，langchain 提供了一个记忆模块。记忆模块提供了一种维持应用状态的方式。这个基础的记忆界面非常简单：它允许我们根据最新的运行输入和输出更新状态，并允许我们利用存储的状态修改或上下文化下一个输入。

在内置的记忆系统中，最简单的就是缓冲记忆。缓冲记忆只是将最近的一些输入/输出预置到当前的输入中。我们可以用代码来看这个过程：

首先，我们需要从 langchain。prompts 导入一些类和函数。然后，我们创建一个 ChatOpenAI 对象，这是我们的语言模型。

```python
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory

prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        “The following is a friendly conversation between a human and an AI. The AI is talkative and ”
        “provides lots of specific details from its context. If the AI does not know the answer to a ”
        “question, it truthfully says it does not know.”
    ),
    MessagesPlaceholder(variable_name=“history”),
    HumanMessagePromptTemplate.from_template(“{input}”)
])

llm = ChatOpenAI(temperature=0)
```


接着，我们创建一个 ConversationBufferMemory 对象，这是我们的记忆。

```
memory = ConversationBufferMemory(return_messages=True)
```

最后，我们创建一个 ConversationChain 对象，它是我们的会话链，会话链会用到之前创建的记忆和语言模型。

```python
conversation = ConversationChain(memory=memory, prompt=prompt, llm=llm)
```

创建了会话链之后，我们就可以用它来预测输入了。

```python
conversation.predict(input=“你好，我是美丽!”)
```

例如，我们可以向会话链输入 “你好，我是美丽!”，然后会话链就会根据存储的状态和输入，生成一个响应。由于我们的记忆模型是缓冲记忆，所以会话链的响应会考虑到最近的一些输入/输出。

在后面的对话中，机器人会记住我们的名字。相反我们可以给机器人取一个特别的名字，因为有记忆的存在，机器人会记住他的名字。

总的来说，通过使用记忆模块，我们的聊天机器人不仅可以进行搜索和数学运算，还能引用过去的交互，理解新的消息。这大大提高了聊天机器人的实用性和智能水平。

祝贺大家，我们的第一个聊天机器人现已完成。
# 1.5  本章小结

本章主要讨论了为什么我们需要 Langchain 这个工具。Langchain 是一个基于语言模型的开发工具，它可以让我们更容易地开发出具有自然语言理解能力的程序。利用 Langchain，我们可以利用模型的大规模语言理解能力，以及它内置的功能，如搜索、数据库查找、Python REPL 等，构建出一系列有用的应用。

Langchain 解决了什么问题呢？许多现代的应用程序都需要理解和生成自然语言。例如，我们可能需要开发一个聊天机器人，可以理解用户的问题，并提供有用的答案。或者，我们可能需要开发一个工具，可以理解和处理自然语言的命令。然而，理解和生成自然语言是一个复杂的问题，需要大量的专业知识。Langchain 通过为开发者提供高级的 API 和工具，大大简化了这个过程。

最后我们通过代码示例展示了如何使用 Langchain 来构建一个简单的聊天机器人。在这个过程中，我们使用了 Langchain 的所有核心模块。只要遵循这个开发流程，我们就可以轻松地构建出更复杂的应用程序。为此，我们需要了解 Langchain 的各种工具，这些工具可以帮助我们实现各种特定的任务；我们需要了解 Langchain 的热门链，这些链代表了一些常见的工作流程；我们还需要了解 Langchain 的不同记忆类型，这可以帮助我们管理和引用过去的互动；最后，我们需要了解如何处理文档，这样我们就可以更有效地和聊天机器人进行交流。总的来说，只要我们理解了这些知识点，我们就可以使用 Langchain 来构建各种复杂的应用程序。
