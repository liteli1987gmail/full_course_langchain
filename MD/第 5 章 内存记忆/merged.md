

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
