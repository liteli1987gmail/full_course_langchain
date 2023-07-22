# 4.1.1 链

在许多人第一次接触 LangChain 的时候，可能会因为其名字误以为它是区块链相关的内容。然而实际上，LangChain 的名字源自其框架的核心设计思路：用最简单的链条，将大预言模型开发的各个组件链接起来。这里的“链”，即我们在复杂系统设计中常说的“链式结构”。

或许你已经注意到了 LangChain 的 logo——一只鹦鹉和一节链条。鹦鹉学舌寓意着大预言模型像鹦鹉一样预测人类的下一段文本，而由无数节链组合起来的链条，象征着通过各种组件的有序连接，形成强大的应用力量。

如果没有链式结构，那么单独的语言大模型，虽然对于简单的应用可能已经足够，但是对于更复杂的应用，我们需要将多个模型或组件进行“链式”结构的连接和组合，这样才能创造出更强大、更具协同性的应用。

例如，我们可以创建一个链，该链接收用户输入，使用 PromptTemplate 格式化它，然后将格式化的响应传递给大模型。我们可以通过将多个链结合在一起，或者将链与其他组件结合在一起，来构建更复杂的链。

这种链式结构在创新应用中的价值已经得到了验证。最近，Johei Nakajima 在 Twitter 上分享了一篇名为《使用 GPT-4、Pinecone、LangChain 进行多样化应用的任务驱动自主代理》的论文，其中他介绍了最新的 Baby AGI。虽然 Baby AGI 现在还只是概念代码阶段，但是通过这个概念我们可以看出，链式结构是实现创新应用的非常有价值的工具。

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

LLMChain 是最基本的构建块链。它接收一个提示模板，使用用户输入进行格式化，然后返回 LLM 的响应。

```
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

llm = OpenAI(temperature=0.9)
prompt = PromptTemplate(
    input_variables=["product"],
    template="What is a good name for a company that makes {product}?",
)
```
要使用 LLMChain，首先创建一个提示模板。我们现在可以创建一个非常简单的链，它会接收用户输入，使用输入格式化提示，然后将其发送到 LLM。

```
from langchain.chains import LLMChain
chain = LLMChain(llm=llm, prompt=prompt)

# Run the chain only specifying the input variable.
print(chain.run("colorful socks"))
```

就如我们在模型输入输出所说, 如果有通用语言模型的方法，那么 Langchain 一定有聊天模型的方法。你也可以在 LLMChain 中使用聊天模型：

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

链的类型分为四种，包括 LLMChain、RouterChain、顺序链（Sequential Chains）和转换链（Transformation Chain）。

LLMChain 是一种简单的链，它在语言模型周围增加了一些功能。它在 LangChain 中被广泛应用，包括在其他链和代理中。LLMChain 由提示模板和语言模型（可以是 LLM 或聊天模型）组成。它使用提供的输入键值（如果有，还有记忆键值）格式化提示模板，将格式化的字符串传递给 LLM，并返回 LLM 的输出。

RouterChain 是一种使用路由器链创建的链，它可以动态地选择给定输入的下一条链。路由器链由两部分组成：路由器链本身（负责选择要调用的下一条链）和目标链（路由器链可以路由到的链）。

顺序链（Sequential Chains）是在调用语言模型后的下一步，特别是当你希望将一次调用的输出作为另一次调用的输入时。顺序链允许你连接多个链并将它们组成执行特定场景的流水线。顺序链有两种类型：SimpleSequentialChain（最简单形式的顺序链，其中每一步都有一个单一的输入/输出，一个步骤的输出是下一个步骤的输入）和 SequentialChain（一种更通用的顺序链，允许多个输入/输出）。

转换链（Transformation Chain）是一种使用通用转换链的方法。作为一个示例，我们将创建一个虚构的转换，它接收一个超长的文本，过滤文本以仅显示前三段，然后将其传递给 LLMChain 进行总结。
# 工具链的理解与应用

在 Langchain 中，"链" 的概念是最经常使用的。这些 "链" 其实就是由一系列工具链构成的，每一个工具都可以视为整个链中的一个环节。这些环节可能非常简单，例如将一个提示模板和一个大语言模型链接起来，形成一个大语言模型链（LLMChains）。然而，也可能更加复杂，例如在整个流程中，通过多个环节进行多个步骤的链接。这可能还包括多个大语言模型以及各种不同的实用工具等。在工具链中，一个链的输出将成为下一个链的输入，这就形成了一个输入输出的链式流程。例如，你可能会从大语言模型的输出中提取某些内容，将其作为 Wolfram Alpha 查询的输入，然后将查询结果带回，并再次通过大型模型生成将返回给用户的响应。这就是一个典型的工具链的示例。

## 常见工具链的功能与应用

在实际的应用中，一些常见的工具链如 APIChain、ConversationalRetrievalQA 等已经被封装好了。

APIChain 使得大语言模型可以与 API 进行交互，以获取相关的信息。构建该链时，需要提供一个与所提供的 API 文档相关的问题。

ConversationalRetrievalQA 链在检索问答链的基础上提供了一个聊天历史组件。它首先将聊天历史（要么明确传入，要么从提供的内存中检索）和问题合并成一个独立的问题，然后从检索器中查找相关的文档，最后将这些文档和问题传递给一个问答链，以返回响应。

对于需要对多个文档进行文档合并的任务，我们可以使用文档合并链，如 MapReduceDocumentsChain 或 StuffDocumentsChain 等。

对于需要从同一段落中提取多个实体及其属性的任务，我们可以使用提取链。

还有一些专门设计用来满足特定需求的链，如 ConstitutionalChain，这是一个保证大语言模型输出遵循一定宪法原则的链，通过设定特定的规则和指导方针，使得生成的内容符合这些原则，从而提供更受控、符合伦理和上下文适当的回应。

## 工具链的使用方法

这些工具链的使用方法通常是先使用类方法实例化，然后通过 run 方法调用，输出结果是一个字符串，然后将这个字符串传递给下一个链。类方法通常以 "from" 和下划线开始，比较常见的有 from_llm()和 from_chain_type()，他们都接受外部的数据来源作为参数。

下面以 SQLDatabaseChain 为例子，看看如何使用它。SQLDatabaseChain 就是一个通过 from_llm()方法实例化的链，它用于回答 SQL 数据库上的问题。

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

### 异步支持

LangChain 通过利用 asyncio 库为链（Chain）提供了异步支持。

目前在 LLMChain（通过 arun, apredict, acall）、LLMMathChain（通过 arun 和 acall）、ChatVectorDBChain 以及 QA 链中支持异步方法。其他链的异步支持正在规划中。

### 使用方法

所有的链都可以像函数一样被调用。当链对象只有一个输出键（也就是说，它的 `output_keys` 中只有一个元素）的时候，我们预期的结果只需要一个字符串，可以使用 `run` 方法。

在 LangChain 中，所有继承自 `Chain` 类的对象，提供了一些用于执行链逻辑的方式。其中一种比较直接的方式就是使用 `__call__` 方法。`__call__` 方法是 `Chain` 类的一个方法，它让 `Chain` 类的实例可以像函数一样被调用，比如 `result = chain(inputs, return_only_outputs=True)` 就完成了调用链。

先看看 `__call__` 方法的定义：
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
你可以通过设置 return_only_outputs 为 True 来配置它只返回输出键值。

```
llm_chain("corny", return_only_outputs=True)
```

返回的结果就不包含 `"adjective": "corny"`：

```
    {'text': 'Why did the tomato turn red? Because it saw the salad dressing!'}
```
然而，当链对象只有一个输出键（也就是说，它的 `output_keys` 中只有一个元素）的时候，我们可以使用 `run` 方法。

```
# llm_chain only has one output key, so we can use run
llm_chain.output_keys
```

```
    ['text']
```
`output_keys` 中只有一个元素 `['text']`，我们可以 `run` 方法：

```
llm_chain.run({"adjective": "corny"})
```

如果输入的键值只有一个，预期的输出也是一个字符串，那么输入可以是字符串也可以是对象，可以使用 `run` 方法也可以使用 `__call__` 方法。

`run` 方法将整个链的输入键值（input key values）进行处理，并返回处理后的结果。需要注意的是，与 `__call__` 方法可能返回字典形式的结果不同，`run` 方法总是返回一个字符串。这也是为什么当链对象只有一个输出键的时候，我们倾向于使用 `run` 方法，因为这时候处理结果自然只有一个，返回字符串形式更直观也更便于处理。

例如，假设我们有一个链对象，它的任务是根据输入的文本生成摘要，那么在调用 `run` 方法的时候，我们可以直接将待摘要的文本作为参数输入，然后得到摘要后的文本。在这种情况下，你可以直接输入字符串，而无需指定输入映射。

另外，你可以很容易地将一个 `Chain` 对象作为一个工具，通过它的 `run` 方法集成到你的 Agent 中，这样可以将链的处理能力直接用于你的 Agent 逻辑中。


### 支持自定义链

你可以子类化 Chain 并实现你自己的自定义链。从其输出中仅仅调试链对象可能会比较困难，因为大多数链对象涉及到相当多的输入提示预处理和 LLM 输出后处理。

### 链的调试

将 verbose 设置为 True 将会在运行链对象时打印出一些链对象的内部状态。

```python
conversation = ConversationChain(
    llm=chat,
    memory=ConversationBufferMemory(),
    verbose=True
)
conversation.run("What is ChatGPT?")
```

### 加记忆的链
链可以使用 Memory 对象进行初始化，这将使得在调用链时数据持久化，使得链具有状态。

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


### 链序列化

我们使用的序列化格式是 json 或 yaml。目前，只有一些链支持这种类型的序列化。我们将随着时间的推移增加支持的链的数量。首先，让我们看看如何将链保存到磁盘。这可以通过.save 方法完成，并指定一个带有 json 或 yaml 扩展名的文件路径。我们可以使用 load_chain 方法从磁盘加载链。
# 4.2.1 LLM 链

LLMChain 将是一个非常简单的 Chains 。这绝对是你最常见到的 Chains 。基本上只是将一个大语言模型与提示（Prompt）链在一起。然后使用提示模板来提供输入, 并将一些内容输入到其中。

以下是文章的事实提取场景下，使用通用 LLM 链的示例代码：

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

在这里，你可以看到我正在设置 OpenAI `text-davinci-003` 模型，我们将将温度设置为零，只需设置最大令牌。如果你知道这是一个默认的标准模型的话，你肯定知道其中许多将根据默认值进行设置。

```
llm = OpenAI(model_name='text-davinci-003', 
             temperature=0, 
             max_tokens = 256)
```

我在这里有一篇小文章，所以我将要做的实际上是事实提取。因此，在这里，我基本上提取了一篇关于 Coinbase 的文章。所以这是一篇相当长的文章，如果我们看一下，它有 3500 个字符。

我们要做的就是从中提取出关键事实。然后我们将对此进行调整。并尝试将这些事实改写成一种新的内容。首先，我们需要我们的 Prompts（提示）模板，所以我们的 Prompts（提示）模板基本上接受输入，这是我们在这里得到的。

```
article = '''Coinbase, the second-largest crypto exchange by trading volume, released its Q4 2022 earnings on Tuesday, giving shareholders and market players alike an updated look into its financials. In response to the report, the company's shares are down modestly in early after-hours trading.In the fourth quarter of 2022, Coinbase generated $605 million in total revenue, down sharply from $2.49 billion in the year-ago quarter. Coinbase's top line was not enough to cover its expenses: The company lost $557 million in the three-month period on a GAAP basis (net income) worth -$2.46 per share, and an adjusted EBITDA deficit of $124 million.Wall Street expected Coinbase to report $581.2 million in revenue and earnings per share of -$2.44 with adjusted EBITDA of -$201.8 million driven by 8.4 million monthly transaction users (MTUs), according to data provided by Yahoo Finance.Before its Q4 earnings were released, Coinbase's stock had risen 86% year-to-date. Even with that rally, the value of Coinbase when measured on a per-share basis is still down significantly from its 52-week high of $206.79.That Coinbase beat revenue expectations is notable in that it came with declines in trading volume; Coinbase historically generated the bulk of its revenues from trading fees, making Q4 2022 notable. Consumer trading volumes fell from $26 billion in the third quarter of last year to $20 billion in Q4, while institutional volumes across the same timeframe fell from $133 billion to $125 billion.The overall crypto market capitalization fell about 64%, or $1.5 trillion during 2022, which resulted in Coinbase's total trading volumes and transaction revenues to fall 50% and 66% year-over-year, respectively, the company reported.As you would expect with declines in trading volume, trading revenue at Coinbase fell in Q4 compared to the third quarter of last year, dipping from $365.9 million to $322.1 million. (TechCrunch is comparing Coinbase's Q4 2022 results to Q3 2022 instead of Q4 2021, as the latter comparison would be less useful given how much the crypto market has changed in the last year; we're all aware that overall crypto activity has fallen from the final months of 2021.)There were bits of good news in the Coinbase report. While Coinbase's trading revenues were less than exuberant, the company's other revenues posted gains. What Coinbase calls its "subscription and services revenue" rose from $210.5 million in Q3 2022 to $282.8 million in Q4 of the same year, a gain of just over 34% in a single quarter.And even as the crypto industry faced a number of catastrophic events, including the Terra/LUNA and FTX collapses to name a few, there was still growth in other areas. The monthly active developers in crypto have more than doubled since 2020 to over 20,000, while major brands like Starbucks, Nike and Adidas have dived into the space alongside social media platforms like Instagram and Reddit.With big players getting into crypto, industry players are hoping this move results in greater adoption both for product use cases and trading volumes. Although there was a lot of movement from traditional retail markets and Web 2.0 businesses, trading volume for both consumer and institutional users fell quarter-over-quarter for Coinbase.Looking forward, it'll be interesting to see if these pieces pick back up and trading interest reemerges in 2023, or if platforms like Coinbase will have to keep looking elsewhere for revenue (like its subscription service) if users continue to shy away from the market.
'''
```

然后我们需要实际的 Prompts（提示），这里的 Prompts（提示）是从这段文本中提取关键事实，不包括观点，给每个事实编号，并保持它们的句子简短。

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
你可以看到，在运行它之后，确实发生了变化。

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

它很好地从我们的文章中获取了事实。它做得相当不错，我们从这篇文章中得到了 10 个事实。

# 4.2.2 顺序链

另一种通用链是顺序链 (Sequential)，它基本上是多个链(Chain)的组合。你会发现，虽然链由单个工具组成，但它们也可以由其他链组成，这些链会连接在一起。工具链是许多神奇事情发生的地方。有各种不同的工具链，这里只是其中一部分，而且随着时间的推移，预计还会添加更多。

我们在上一节制作了一个提取新闻 10 个事实的 LLMChain, 为了让你理解顺序链是如何工作的，我们先加一个新的链。

现在我们要做的是制作一个新的。然后我们将把其中一些内容链在一起，所以下一个我要做的也是一个 LLMChain。

这将采用上一节的 10 个事实。但我们将把它们改写成投资者报告的形式，所以你可以在这里看到，我们将说好了，你是高盛的分析师，接受以下事实列表，并用它们为投资者撰写一个简短的段落，不要遗漏关键信息。我们也可以放一些东西在这里，也不要杜撰信息，但这是我们要传入的事实。

```
investor_update_prompt = PromptTemplate(
    input_variables=["facts"],
    template="You are a Goldman Sachs analyst. Take the following list of facts and use them to write a short paragrah for investors. Don't leave out key info:\n\n {facts}"
)
```

再次强调，这是一个 LLMChain ，我们传入 LLM，我们仍然使用上面定义的原始模型，我们传入提示模板。然后我们可以运行它。

```
investor_update_chain = LLMChain(llm=llm, prompt=investor_update_prompt)

investor_update = investor_update_chain.run(facts)

print(investor_update)
len(investor_update)
```

你可以看到，确实回来了。

```
Coinbase released its Q4 2022 earnings on Tuesday, revealing total revenue of $605 million and a GAAP loss of $557 million. Despite the losses, Coinbase's stock had risen 86% year-to-date before its Q4 earnings were released. Consumer trading volumes fell from $26 billion in Q3 2022 to $20 billion in Q4 2022, while institutional volumes fell from $133 billion to $125 billion. The overall crypto market capitalization fell about 64%, or $1.5 trillion during 2022. Trading revenue at Coinbase fell from $365.9 million in Q3 2022 to $322.1 million in Q4 2022, while its "subscription and services revenue" rose from $210.5 million in Q3 2022 to $282.8 million in Q4 2022. Despite the market downturn, monthly active developers in crypto have more than doubled since 2020 to over 20,000.
788
```

它写了一篇相当连贯的好文章。它比之前的要短得多。

### 加入顺序链

我们将使用简单的顺序链（SimpleSequentialChain）来完成这个过程。简单的顺序链就像 PyTorch 中的标准顺序模型一样，你只是从 A 到 B 到 C，没有做任何复杂的操作。

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

迄今为止，你将看到的最常见的变化是由一个大语言模型组成的。还有一个提示模板，还有一堆其他很酷的链和工具在 Langchain 中。
# 4.3.1 Stuff 链

在大数据和自然语言处理领域，各种不同的处理链方式可以用来优化信息检索和生成答案。本文将解析 'Stuff' 类型的处理链，并说明其如何通过改变输入的组织和输出的生成方式来提高文档搜索的质量。

#### 1. 整体流程

'Stuff' 文档处理链是一种直接的处理方式。它接收一组文档，将所有文档插入到一个提示中，然后将该提示传递给语言模型 (LLM)。

#### 2. 插入文档阶段

在这个阶段，系统接收一组文档，将它们全部插入到一个提示中。提示(Prompt) 是全部文档内容。这种方式适用于文档较小且大部分调用只传入少量文档的应用。它可以简单地将所有文档拼接在一起，形成一个大的提示，然后将这个提示传递给 LLM。

#### 3. 生成答案阶段

在这个阶段，系统将包含所有文档的提示传递给 LLM。LLM 根据这个提示生成答案。由于所有的文档都被包含在同一个提示中，所以 LLM 生成的答案会考虑到所有的文档。

#### 4. 最终实现效果

通过 'Stuff' 文档处理链，系统可以对包含多个文档的问题生成一个全面的答案。这种处理方式可以提高文档搜索的质量，特别是在处理小文档和少量文档的情况下。

#### 5. 适用场景

由于 'Stuff' 类型的处理链方式主要适用于处理小文档和少量文档的情况，所以它特别适用于那些大部分调用只传入少量文档的应用。然而，对于需要处理大量文档或者文档较大的情况，可能需要使用其他类型的处理链，如 'Refine' 或 'MapReduce'。

总的来说，通过使用 'Stuff' 文档处理链，系统可以有效地处理小文档和少量文档的情况，从而提高文档搜索的质量。

# 4.3.2 精化链

本文将解析 'Refine' 类型的处理链，并说明其如何通过改变输入的组织和输出的生成方式来提高文档搜索的质量。

#### 1. 整体流程

'Refine' 文档处理链通过遍历输入文档并迭代更新其答案来构建响应。对于每个文档，它将所有非文档输入（例如用户的问题或其他与当前文档相关的信息）、当前文档和最新的中间答案传递给语言模型 (LLM)，以获得新的答案。

#### 2. 遍历文档阶段

在这个阶段，系统会遍历输入的所有文档。对于每个文档，一起作为提示(Prompt)传递给 LLM 的内容有：

- 一些上下文信息，例如用户的问题或其他与当前文档相关的信息。
- 最新的中间答案。中间答案是系统在处理之前的文档时产生的。一开始，中间答案可能是空的，但随着系统处理更多的文档，中间答案会不断更新。
- 当前文档。

与 Map reduce 链和重排链不同的是它不产生新文档，只不断更新的是提示，迭代出更全面的答案。而且文档之间的影响是传递性的，上一个文档形成的答案会影响下一个文档的答案。

#### 3. 更新答案阶段

在这个阶段，系统将提示传递给 LLM，然后将 LLM 生成的答案作为新的中间答案。这个过程会迭代进行，直到所有的文档都被处理。

#### 4. 最终实现效果

通过 'Refine' 文档处理链，系统可以对包含多个文档的问题生成一个全面的答案，而且每个文档的处理结果都会影响后续文档的处理。这种处理方式可以提高文档搜索的质量，特别是在处理大量文档的情况下。

#### 5. 适用场景

'Refine' 类型的处理链方式主要适用于处理大量文档的情况，特别是当这些文档不能全部放入模型的上下文中时。然而，这种处理方式可能会使用更多的计算资源，并且在处理某些复杂任务（如文档之间频繁地交叉引用，或者需要从许多文档中获取详细信息）时可能表现不佳。

总的来说，通过使用 'Refine' 文档处理链，系统可以有效地处理大量文档的情况，从而提高文档搜索的质量。然而，这种处理方式可能需要更多的计算资源，并且可能在处理复杂任务时表现不佳。
# 4.3.3 Map reduce 链

在大数据和自然语言处理领域，各种不同的处理链方式可以用来优化信息检索和生成答案。本文将解析 'MapReduce' 类型的处理链，并说明其如何通过改变输入的组织和输出的生成方式来提高文档搜索的质量。

#### 1. 整体流程

'MapReduce' 文档处理链主要由两个部分组成：映射（Map）阶段和归约（Reduce）阶段。在映射阶段，系统对每个文档单独应用一个语言模型（LLM）链，并将链输出视为新的文档。在归约阶段，系统将所有新文档传递给一个单独的合并文档链，以获得单一的输出。如果需要，系统会首先压缩或合并映射的文档，以确保它们适合合并文档链。

#### 2. 映射阶段（Map Stage）

在映射阶段，系统使用 LLM 链对每个输入的文档进行处理。处理的方式是，将当前文档作为输入传递给 LLM 链，然后将 LLM 链的输出视为新的文档。这样，每个文档都会被转化为一个新的文档，这个新文档包含了原始文档的处理结果。

对于每个文档，作为提示(Prompt)传递给 LLM 的内容是原始文档。比起“ Stuff ”类型多了预处理。

每个原始文档都经过 LLM 链处理的结果写入一个新文档，这就是映射的过程。比如原文档有 2000 字，经过 LLM 链处理的结果是 200 字。200 字的结果存储为一个新文档，但是跟 2000 字原文档存着映射关系。

#### 3. 归约阶段（Reduce Stage）

在归约阶段，系统使用合并文档链将映射阶段得到的所有新文档合并成一个。如果新文档的总长度超过了合并文档链的容量，那么系统会使用一个压缩过程将新文档的数量减少到合适的数量。这个压缩过程会递归进行，直到新文档的总长度满足要求。

#### 4. 最终实现效果

通过 'MapReduce' 文档处理链，系统可以对每个文档单独进行处理，然后将所有文档的处理结果合并在一起。这种处理方式可以提高文档搜索的质量，特别是在处理大量文档的情况下。

#### 5. 适用场景

'MapReduce' 类型的处理链方式主要适用于处理大量文档的情况，特别是当这些文档不能全部放入模型的上下文中时。通过并行处理每个文档并合并处理结果，这种处理方式可以在有限的资源下处理大量的文档。然而，这种处理方式可能会使用更多的计算资源，并且可能在处理某些复杂任务（如文档之间频繁地交叉引用，或者需要从许多文档中获取详细信息）时可能表现不佳。

总的来说，通过使用 'MapReduce' 文档处理链，系统可以有效地处理大量文档的情况，从而提高文档搜索的质量。然而，这种处理方式可能需要更多的计算资源，并且可能在处理复杂任务时表现不佳。
# 4.3.4 重排链

标题：源码解析：'Map Re-rank' 类型在文档处理链中的应用

在大数据和自然语言处理领域，各种不同的处理链方式可以用来优化信息检索和生成答案。本文将解析 'Map Re-rank' 类型的处理链，并说明其如何通过改变输入的组织和输出的生成方式来提高文档搜索的质量。

#### 1. 整体流程

'Map Re-rank' 文档处理链对每个文档运行初始提示，这个提示不仅试图完成任务，还对其答案的确定程度给出评分。最后，得分最高的响应将被返回。

#### 2. 映射和评分阶段

在这个阶段，系统对每个文档运行初始提示。每个文档都会被独立地处理，处理的方式是，系统不仅试图完成任务，还对其答案的确定程度给出评分。这样，每个文档都会被转化为一个新的文档，这个新文档包含了原始文档的处理结果和评分。

对于每个文档，作为提示(Prompt)传递给 LLM 的内容是原始文档, 但是提示模板增加了评分规则。拿到 LLM 链的答案后，存储为一个新文档，与原文档形成映射关系。

#### 3. 重排阶段（Re-rank Stage）

在这个阶段，系统根据每个新文档的评分进行重排。具体来说，系统会选择得分最高的新文档，并将其作为最终的输出。

只有这个类型有自动重排的机制，因为只有这个类型，对原始文档进行处理的时候，添加了评分规则的提示。

#### 4. 最终实现效果

通过 'Map Re-rank' 文档处理链，系统可以对每个文档独立地进行处理和评分，然后选择得分最高的结果作为最终输出。这种处理方式可以提高文档搜索的质量，特别是在处理大量文档的情况下。

#### 5. 适用场景

'Map Re-rank' 类型的处理链方式主要适用于处理大量文档的情况，特别是当需要从多个可能的答案中选择最优答案时。通过对每个文档的处理结果进行评分和重排，这种处理方式可以在有限的资源下找到最优的答案。然而，这种处理方式可能会使用更多的计算资源，并且可能在处理某些复杂任务（如文档之间频繁地交叉引用，或者需要从许多文档中获取详细信息）时可能表现不佳。

总的来说，通过使用 'Map Re-rank' 文档处理链，系统可以有效地处理大量文档的情况，并从多个可能的答案中选择最优答案，从而提高文档搜索的质量。然而，这种处理方式可能需要更多的计算资源，并且可能在处理复杂任务时表现不佳。
# 合并文档链概述

在许多应用场景中，我们需要与文档进行交互，如阅读说明书、浏览产品手册等等。近来，基于这些场景开发的应用，如 chatDOC 和 chatPDF，都受到了广大用户的欢迎。为了满足对特定文档进行问题回答、提取摘要等需求，Langchain 设定了几种合并文档链类型。

这些核心链都是为处理文档而设计的。它们在对文档进行概括、回答文档问题、从文档中提取信息等方面非常有用。

但是文档链的类型给初学者造成了很大的困扰。主要是因为我们通常不清楚在指定了这些类型后，中间的处理流程发生了什么。如果我们能从各个类型的具体步骤进行理解，就会发现，这些类型的主要区别在于它们处理输入文档的方式，以及在中间过程中与模型的交互次数和答案来源于哪些阶段。理解了这些，我们就可以更清楚地认识到各种类型的优缺点，从而在生产环境中做出更好的决策。

换句话说，一旦我们理解了每个类型的具体步骤提交了什么提示(Prompt)，提示从何而来就可以明确知道使用哪种类型更符合我们的需求。我们会在后面对每个类型经历的具体步骤进行拆解。在这里我们先做个概述，没看懂可移步相应的文档类型的小节。

“Stuff 链” 是处理文档链中最直接的一个。它接收一组文档，将它们全部插入到一个提示中，然后将该提示传递给 LLM。这种链适合于文档较小且大部分调用只传入少量文档的应用。

“精化（Refine）”通过遍历输入文档并迭代更新其答案来构建响应。对于每个文档，它将所有非文档输入、当前文档和最新的中间答案传递给 LLM 链，以获得新的答案。

由于精化链一次只向 LLM 传递一个文档，因此它非常适合需要分析比模型上下文能容纳更多的文档的任务。但显然，这种链会比如 Stuff 链这样的链调用更多的 LLM。此外，还有一些任务很难通过迭代来完成。例如，当文档经常相互交叉引用或任务需要许多文档的详细信息时，精化链的表现可能较差。

“Map Reduce”首先将 LLM 链单独应用于每个文档（Map 步骤），并将链输出视为新的文档。然后，它将所有新文档传递给一个单独的“Combine Documents Chain”，以获得单一的输出（Reduce 步骤）。它可以选择首先压缩或合并映射的文档，以确保它们适合“Combine Documents Chain”（这将经常将它们传递给 LLM）。如果需要，这个压缩步骤将递归地执行。

“重排链（Map Re-rank）”对每个文档运行初始提示，不仅试图完成任务，还对其答案的确定程度给出评分。得分最高的响应将被返回。


# API 工具链

另一个非常有用的工具链的例子是 API 工具链，所以在这里我只是向你展示了用于天气信息的一个例子。我们设置了要使用的 API，这将根据这些文档编写 API 调用。这就是这个调用将输出的内容。然后这将使用该调用查询 API 并返回结果。
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
显然，它只能回答 API 能够给你的内容，这里它基本上是在给我们返回这个并且它告诉我们，是的，这个 JSON 响应中的一些事情表明正在下雨，但是需要注意的是，通常文档加上 URL 加上 JSON 会超过大语言模型可以处理的标记数，因此如果你在达芬奇模型上使用超过 4000 个标记，可能会出现错误。

```
from langchain.chains.api import open_meteo_docs
chain_new = APIChain.from_llm_and_api_docs(llm, 
                                           open_meteo_docs.OPEN_METEO_DOCS, 
                                           verbose=True)
```

```
chain_new.run('What is the temperature like right now in Bedok, Singapore in degrees Celcius?')
```

这里你可以看到，我问它在新加坡 Bedok 的温度是多少摄氏度。它写下了这个用于查询的 URL。

```
    > Entering new APIChain chain...
https://api.open-meteo.com/v1/forecast?latitude=1.3&longitude=103.9&hourly=temperature_2m&current_weather=true&temperature_unit=celsius
{"latitude":1.375,"longitude":103.875,"generationtime_ms":0.38802623748779297,"utc_offset_seconds":0,"timezone":"GMT","timezone_abbreviation":"GMT","elevation":6.0,"current_weather":{"temperature":26.1,"windspeed":10.5,"winddirection":16.0,"weathercode":3,"time":"2023-02-22T14:00"},"hourly_units":{"time":"iso8601","temperature_2m":"°C"},"hourly":{"time":["2023-02-22T00:00","2023-02-22T01:00","2023-02-22T02:00","2023-02-22T03:00","2023-02-22T04:00",27.6,27.5,27.2,26.8,26.4,26.1,25.7,25.5,25.4,25.3,25.2,25.1,25.0,24.9,24.9,24.9,24.9,24.9]}}

    > Finished chain.
 The temperature right now in Bedok, Singapore is 26.1 degrees Celcius.
```  
它给我们返回了当前的温度和位置。

另外要考虑的一件事是，这是相当昂贵的，如果我们每千个标记支付两美分我们刚刚输入了 4000 个标记，只是为了获取天气或其他东西，这并不总是最高效的方法，但它确实显示了 LangChain 可以做这些事情。你可以编写一些代码来调用你想要的 API 调用。
# PALChain

下一个我要向你展示的是 PAL Math Chain，这其实是使用了一个不同的大语言模型。

我们基本上要做的是，当我们遇到某种数字问题时，我们将使用它。这是 Langchain 文档中的一个例子，Jan 有宠物的数量是三倍。这是一个数学方程问题了。

### 为什么用 PALChain ？

我们基本上在这里 Prompts（提示）模型这样做的方式是将这个文字陈述转化为一个小型的 Python 函数，然后计算数学问题，而不是仅仅依靠语言模型猜测。

来看一个有趣的例子。这是非常简单的数学问题，食堂有 23 个苹果，如果他们用了 20 个来吃午饭，又买了 6 个，那么他们现在还剩下多少个苹果。

问题在于，如果你使用的是大语言模型，它们可能会得到正确的答案，但如果你使用的是一些较小的模型，甚至只是像 T5 模型一样，大多数 T5 模型都会得到这样的错误答案，而不是依赖其中一个模型来做这个，我们可以使用这个方法，它基本上是获取这些数据并进行重写。

```
from langchain.chains import PALChain
pal_chain = PALChain.from_math_prompt(llm, verbose=True)

question = "Jan has three times the number of pets as Marcia. Marcia has two more pets than Cindy. If Cindy has four pets, how many total pets do the three have?"
```

```
pal_chain.run(question)
```
你可以看到它写了一个 Python 函数，它使用了文档字符串将我们之前的内容放在这里，我们从苹果的初始开始，所以它只是将这些变量赋值。然后苹果剩下的数量等于初始苹果减去使用的苹果加上购买的苹果，它确实给了我们准确的结果。

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
然后我们可以将输出带入另一个大语言模型中，然后以对话的方式重新表达它，这样它可以告诉你苹果剩下的数量是多少，或者我们可以直接从这个模块中获取输出。

