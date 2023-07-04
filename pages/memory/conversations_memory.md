

所以在这个文章中，我将讨论 LangChain 的记忆 (Memory)。

想一想，我们为什么需要记忆 (Memory)？

构建聊天机器人等等的一个重要原因是，人们对任何类型的聊天机器人或聊天代理都抱有人的期望，他们期望它具有人的品质和回应能力。

如果他们不了解这些东西通常只是进行一次调用并回应，他们会感到非常沮丧。所以你经常会发现人们想要缩写，当他们回顾先前的对话时，这可能涉及到地点、时间以及其他许多不同的事情。

另一个需要考虑的重要因素是跨整个对话达成共识的概念。如果某人开始谈论一个人，并且给出了一个名字，那么他们之后可能会简单地称呼该人为他或她。

为了让大型语言模型能够理解是谁，我们需要给它一些记忆 (Memory)，以便它可以回顾并确定他们正在谈论的是谁，或者我们一直在进行的对话是关于什么的。

### 大型语言模型本身没有记忆 (Memory)

正如我在第一个文章中提到的，大型语言模型本身没有记忆 (Memory)，你只需要传入提示，然后根据传入的提示得到有条件的回应。

过去人们尝试过在 Transformer 模型中加入某种记忆 (Memory)，但迄今为止还没有真正适用于大规模使用的最佳方案，希望将来可以有一种内置的记忆 (Memory)机制，使 Transformer 模型能够在内部保留信息。

目前，我们在大型语言模型中处理记忆 (Memory)有两种主要选择。

### 如何处理记忆 (Memory)问题？

第一种是将记忆 (Memory)放回到提示中，第二种方法是进行某种外部查找。

在本文章中，我们将简要介绍第二种方法，但在未来的一些文章中，我将更详细地介绍使用数据库和其他外部信息查找的方法。

第一种方法是将记忆 (Memory)直接放入提示中。

如果我们回顾之前提到的提示，我们有一个上下文，这里假设你是一个名叫 Kate 的数字助手，喜欢进行对话。
然后你可以看到，我们基本上只是告诉它当前的对话已经进行了多少。

用户说“嗨，我是 Sam”，
代理回应“嗨 Sam，你好吗”，
我说“我很好，你叫什么名字”，
代理回应“我的名字是 Kate”，
然后用户说“你今天好吗，Kate”。

你可以看到所有这些信息都会放入提示中。这种方式在这些事情中存在一些问题，因为当我们进行长时间对话时，没有办法将所有信息都适应语言模型可以接受的令牌范围内。

现在 OpenAI 的最新模型在一个范围内使用约 16k 个令牌，这是一个很大的信息量，但是无法保存完整的对话。这就带来了一系列问题，我们可以采取哪些方式来解决这个问题。

LangChain 内置了许多解决方案，其中一些只是非常简单的方式，就像之前我展示的那样，将其直接放入提示中。

我们还有其他一些方式，比如随着对话的进行进行摘要，还有一种方式是限制窗口，只记住最近的几次对话，还有一些方式，它将最近的几次对话进行摘要，但是对于其他的内容则进行总结。

摘要实际上是通过调用一个语言模型自身来完成的，它会询问“请给我总结一下这次对话”。这也是你将看到的内容。

我们还有一些更外部的方式来解决这个问题，比如制作某种知识图谱记忆 (Memory)，并将信息放入实体记忆 (Memory)中。最后，你可以自定义它，如果你有一个非常特殊的情况，你想要使用自己的记忆 (Memory)系统，你也可以在 LangChain 中编写自己的记忆 (Memory)系统。


## 代码实现

让我们进入代码，看看这些功能是如何工作的。

好的，首先我们需要安装 OpenAI LangChain 等常规设置。将你的 OpenAI 密钥放在这里。

```bash
!pip -q install openai langchain huggingface_hub transformers
```

```
import os

os.environ['OPENAI_API_KEY'] = ''
```

## 对话缓冲区记忆 (Memory) ConversationBufferMemory

我们将首先看一下对话缓冲区记忆 (Memory)。

在 LangChain 中，我们需要导入所需的记忆 (Memory)类型，然后实例化它，并将其传递给 Chain。

```python
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain import OpenAI
from langchain.chains import ConversationChain
```

这里我们使用的 Chain 只是一个简单的对话链 `ConversationChain`，允许我们跟 OpenAI 模型交互，并传递我们想要说的内容。

```python
llm = OpenAI(model_name='text-davinci-003', 
             temperature=0, 
             max_tokens = 256)
``` 

```python
memory = ConversationBufferMemory()
```

你会看到，我将 verbose 设置为 true，以便我们可以看到输入的提示以及得到的回应。

```python
conversation = ConversationChain(
    llm=llm, 
    verbose=True, 
    memory=memory
)
```

在这里，我们设置了最简单的记忆 (Memory)类型，即对话缓冲区 `ConversationBufferMemory`，它将跟踪用户和代理说的话，并随着对话的进行将其堆叠到提示中。

我们从我说“嗨，我是 Sam”开始，我们可以看到输入的提示以及当前对话，即“嗨，我是 Sam”。

```python
conversation.predict(input="Hi there! I am Sam")
```

AI 的回应是“嗨 Sam，我叫 AI，很高兴见到你，你今天来这里有什么事吗？”

```
        > Entering new ConversationChain chain...
Prompt after formatting:
The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.

Current conversation:

Human: Hi there! I am Sam
AI:

        > Finished chain.
 Hi Sam! My name is AI. It's nice to meet you. What brings you here today?
```

接下来，我们将对此进行回应，我要问它今天过得如何，所以我问了这个问题，我没有直接回答它。

```
conversation.predict(input="How are you today?")
```
我们看一下它是如何处理的，它对此没有任何问题，它回答说“我过得很好，谢谢你的关心，你呢？”然后我又说“我很好，谢谢你。你能帮我解决一些客户支持的问题吗？”

```
        > Entering new ConversationChain chain...
Prompt after formatting:
The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.

Current conversation:

Human: Hi there! I am Sam
AI:  Hi Sam! My name is AI. It's nice to meet you. What brings you here today?
Human: How are you today?
AI:

        > Finished chain.
 I'm doing great, thanks for asking! How about you?
```

你会注意到现在我们进入了这个对话，并将对话存储起来，这就是输入的提示，它只是在每一轮对话之后让提示变得更长。


我问它“你能帮我解决一些客户支持的问题吗？”。它回答说“当然可以，你有什么样的客户，需要什么帮助？”。

```
conversation.predict(input="I'm good thank you. Can you help me with some customer support?")
```

然后我可以继续这个对话，但是这将变得越来越长，所以在任何时候，我们可以查看对话记忆 (Memory)缓冲区，并看到实际上都说了些什么。这样我们就可以在任何时候保存对话，或者查看对话。

```
        > Entering new ConversationChain chain...
Prompt after formatting:
The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.

Current conversation:

Human: Hi there! I am Sam
AI:  Hi Sam! My name is AI. It's nice to meet you. What brings you here today?
Human: How are you today?
AI:  I'm doing great, thanks for asking! How about you?
Human: I'm good thank you. Can you help me with some customer support?
AI:

    > Finished chain.
 Absolutely! What kind of customer support do you need?
``` 

这样做可以让你看到我们之前对话的确切内容，这是 LangChain 中最简单的记忆 (Memory)形式，但仍然非常强大，特别是当你知道人们与你的 Bot 的互动次数有限，或者你实际上要在五次互动后关闭它，诸如此类的情况下，这种记忆 (Memory)将非常有效。

### 对话摘要记忆 (Memory) ConversationSummaryMemory

下一个记忆 (Memory)形式是对话摘要记忆。先导入对话摘要记忆。

```python
from langchain.chains.conversation.memory import ConversationSummaryMemory
from langchain import OpenAI
from langchain.chains import ConversationChain

llm = OpenAI(model_name='text-davinci-003', 
             temperature=0, 
             max_tokens = 256)
```
然后将其实例化到摘要记忆中，注入到当前对话链上 `ConversationChain`。

```python
summary_memory = ConversationSummaryMemory(llm=OpenAI())
conversation = ConversationChain(
    llm=llm, 
    verbose=True, 
    memory=summary_memory
)
```

现在的区别是，不再将我说的每一句话和机器人回应来回传递到实际提示中，而是要对其进行摘要。

我们把上一次的对话重复一遍：

```python
conversation.predict(input="Hi there! I am Sam")
conversation.predict(input="How are you today?")
conversation.predict(input="Can you help me with some customer support?")
```

打印 `conversation.memory.buffer` 看看：
```
print(conversation.memory.buffer)
```

所以你会看到，在开始时，我们会得到类似这样的东西：“好的，嗨，嗨，我是 Sam，对，Hi Sam，我叫 AI，很高兴见到你，你今天为什么来这儿？”我说我今天和之前一样，但是你注意到现在在第二次互动中，这是我说的第二件事，现在不再显示我第一次说的内容，而是对其进行摘要，所以你可以看到，人类介绍自己是 Sam，AI 介绍自己是 AI，然后 AI 问 Sam 今天来这里是为了什么，所以这就是不同之处。

```
The human introduces themselves as Sam, and the AI introduces itself as AI. The AI then asks what brings Sam here today, and Sam asks how the AI is doing. The AI responds that it is doing great, and when Sam asks for customer support, the AI responds positively and asks what kind of customer support Sam needs.
```

现在在这种情况下，我们实际上使用了更多的标记，但是随着我们的继续，你会看到，如果我说“好的，你能帮我解决一些客户支持的问题吗？”

当我们进入下一个摘要时，人类介绍自己是 Sam，然后说 AI 表现得很好，然后 AI 最后说的话，所以这是在开始时，也许它正在添加更多的标记，因为你得到了整个对话的摘要版本，但随着时间的推移，这肯定会使用更少的标记，所以你会看到，现在它正在回答，如果我继续这个对话，我将能够看到它实际上并没有存储我们 **逐字逐句** 地说的话，而是存储了对话的摘要。

我们的摘要基本上是整个对话的摘要，因此每次进行新的摘要调用时，我们就会有多个对大型语言模型的调用，我们有对响应的调用。

但我们还有对对话进行摘要的调用，因为在第一步之后，第一步不需要摘要，但在第一步之后，你对每一步的对话都有了摘要。我们可以简单地打印出来，看一下，我们可以看到是的，当 Sam 询问客户支持时，AI 的回应很好，并问 Sam 需要什么样的客户支持，你会注意到，这种情况下它在进行共同引用解析，而不是人类可能会说的是它对他问的回答，而在这里它坚持使用 AI 和 Sam，所以这是一个摘要，它也非常有用。

## ConversationBufferWindowMemory

下一个是第一个的另一种版本，所以这是一个对话缓冲窗口记忆 (ConversationBufferWindowMemory)，所以我们将做的是将对话内容简单地导入提示中。

首先还是导入和实例化 ConversationBufferWindowMemory。

```python
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain import OpenAI
from langchain.chains import ConversationChain

llm = OpenAI(model_name='text-davinci-003', 
             temperature=0, 
             max_tokens = 256)
```

但区别在于现在我们将设置我们在这里的最后几次互动，现在我将其设置得很低，只是为了让我们在这里看到它，但你实际上可以将其设置得比这个高得多，可以获取最后 5 次或 10 次互动，具体取决于你想使用多少个标记，这也与你想为这种情况花费多少钱有关，它实例化并在这里设置 k = 2，我们有一个对话：

```python
# We set a low k=2, to only keep the last 2 interactions in memory
window_memory = ConversationBufferWindowMemory(k=2)

conversation = ConversationChain(
    llm=llm, 
    verbose=True, 
    memory=window_memory
)
```

我们来依次执行以下对话，可以在 [codelab](https://colab.research.google.com/drive/1q3K5Rq_l58p9cHmP-yIApCnYE-eAtzDG?usp=sharing#scrollTo=afMEzKtGtXg7) 尝试，打印终端进程：

```python
conversation.predict(input="Hi there! I am Sam")

conversation.predict(input="I am looking for some customer support")

conversation.predict(input="My TV is not working.")

conversation.predict(input="When I turn it on it makes some weird sounds and then goes black")

```

这样我们就有了两个东西，所以现在它会再次问我，对此很抱歉，它要求提供更多信息，我提供了信息。

```
> Entering new ConversationChain chain...
Prompt after formatting:
The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.

Current conversation:
Human: I am looking for some customer support
AI:  Sure thing! I can help you with that. What kind of customer support do you need?
Human: My TV is not working.
AI:  I'm sorry to hear that. Can you tell me more about the issue you're having with your TV?
Human: When I turn it on it makes some weird sounds and then goes black
AI:

> Finished chain.
 That doesn't sound good. Have you tried unplugging the TV and plugging it back in? That can sometimes help reset the system.
```

但你会注意到，现在当我告诉它我打开后，它发出奇怪的声音，然后变黑，提示现在具有了最后的互动，即我们最后两次完整的互动加上这个新的互动，但它 **丢失** 了我说“嗨，我是 Sam”这句话，它只将最后两次互动传递给大型语言模型。

所以如果你有一些情况，比如我们将其设置为 k = 5，大多数对话可能不会有很大变化，它有时会回到对话早期的事情，但很多时候你可以用很短的记忆 (Memory)来愚弄人们，只要记住三到五个对话步骤，你可以看到，即使我们看一下对话记忆 (Memory)缓冲区，它仍然有完整的对话，所以如果我们想用它来查看或存储一些东西，我们可以这样做。

## ConversationSummaryBufferMemory

看单词拼写就知道这个则是前面两个的组合。跟之前一样，先导入和实例化。

```python
from langchain.chains.conversation.memory import ConversationSummaryBufferMemory
from langchain import OpenAI
from langchain.chains import ConversationChain

llm = OpenAI(model_name='text-davinci-003', 
             temperature=0, 
             max_tokens = 512)
```

现在我们将同时具有类似我们看过的第二个的摘要，但其中也将包含一个缓冲区，所以在这里，我们基本上是将其设置为缓冲区的标记数 `max_token_limit`，我们将标记限制为 40 个标记，所以我们可以以几种不同的方式设置它，我们这样实例化这个记忆 (Memory)。

```
# Setting k=2, will only keep the last 2 interactions in memory
# max_token_limit=40 - token limits needs transformers installed
memory = ConversationSummaryBufferMemory(llm=OpenAI(), max_token_limit=40)

conversation_with_summary = ConversationChain(
    llm=llm, 
    memory=memory, 
    verbose=True)
```

然后我们将再次按照以下代码顺序进行对话：

```
conversation_with_summary.predict(input="Hi there! I am Sam")

conversation_with_summary.predict(input="I need help with my broken TV")

conversation_with_summary.predict(input="It makes weird sounds when i turn it on and then goes black")

conversation_with_summary.predict(input="It seems to be Hardware")
```

这里没有什么不同的地方，我们有‘我需要帮助修理我的坏电视’，你可以看到它只是按照对话的步骤进行，然后问我为什么，我基本上告诉它是什么问题。


```
print(conversation_with_summary.memory.moving_summary_buffer)
```

我们还可以打印出移动的摘要缓冲区：

```

The human, Sam, introduces themselves to the AI. The AI responds and asks what brings Sam to them today, and Sam responds that they need help with their broken TV. The AI expresses sympathy and asks what the problem with the TV is, and Sam responds that it makes weird sounds when they turn it on and then goes black. The AI responds with "Hmm, that doesn't sound good. Do you know if it's a hardware or software issue?"
```

在这个互动中，我们现在走过了我们之前的地方，所以现在它在对之前的步骤进行摘要，所以人类 Sam 向 AI 介绍自己，AI 回应并问 Sam 为什么来找他们，然后给出了最后 K 个步骤或最后的一定数量的标记的实际对话，所以你可以看到，我输入了“听起来不太好，也许是硬件问题”，然后摘要也更新了，所以我们丢失了对话中的一步，但我们更新了摘要，将其包含在内，AI 表达同情并问问题。

所以这在某种程度上可以说是两全其美的东西。

如果我们想看看在对话的某个点上的摘要是什么样的，这在各种不同情况下可能很有用，如果你想让其他模块从对话中提取信息之类的，你可以尝试将其传递给那个模块，然后在其进行时尝试。

## 知识图谱记忆 Conversation Knowledge Graph Memory

接下来的两个有点不同，所以接下来的两个基本上是试图根据对话中的内容来表示对话并提取信息，以实体和知识图谱记忆的形式，所以这一个是知识图谱记忆，所以对话是用于知识图谱，其余部分相同，我们在其中放入了一个简单的提示，我只是为了让你能够看到它实际上在做什么，它现在添加了提示，只有当 AI 仅使用相关信息部分中包含的信息，并且不会产生幻觉时，才会使用它。

我们试图让它更专注于我们所说的内容，而不会以任何方式增加内容。

我们像通常一样实例化它。

```python
from langchain.chains.conversation.memory import ConversationKGMemory
from langchain import OpenAI
from langchain.chains import ConversationChain
from langchain.prompts.prompt import PromptTemplate

llm = OpenAI(model_name='text-davinci-003', 
             temperature=0, 
             max_tokens = 256)


template = """The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. 
If the AI does not know the answer to a question, it truthfully says it does not know. The AI ONLY uses information contained in the "Relevant Information" section and does not hallucinate.

Relevant Information:

{history}

Conversation:
Human: {input}
AI:"""
prompt = PromptTemplate(
    input_variables=["history", "input"], template=template)

conversation_with_kg = ConversationChain(
    llm=llm, 
    verbose=True, 
    prompt=prompt,
    memory=ConversationKGMemory(llm=llm)
)                 
```

我们的对话游戏在这里开始。

```
conversation_with_kg.predict(input="Hi there! I am Sam")

conversation_with_kg.predict(input="My TV is broken and I need some customer assistance")

conversation_with_kg.predict(input="It makes weird sounds when i turn it on and then goes black")

conversation_with_kg.predict(input="Yes it is and it is still under warranty. my warranty number is A512423")
```

但你会看到，每当它认为有相关信息时，它会传递这些信息，所以如果我继续说对，你知道这是什么问题，我给了一些信息，它实际上会接收这些信息并构建一个小型的知识图谱。

```python
# NetworkX Entity graph
import networkx as nx
import matplotlib.pyplot as plt

# nx.draw(conversation_with_kg.memory.kg, with_labels=True)
# plt.show()

print(conversation_with_kg.memory.kg)
print(conversation_with_kg.memory.kg.get_triples())
```

所以你实际上应该能够将其绘制出来，但只是向你展示这一点，而不是绘制它出来，我们看到我们已经有了这个小型的 NetworkX 图形，我们可以看到 Sam，好的，前两个是节点，第三个始终是连接它们的东西，所以好的，Sam 是人类，电视坏了，对的，电视发出奇怪的声音，电视变黑，电视在保修期内，是的，正确的设备在保修期内，这是我们在这里得到的保修号码，所以你可以看到它提取了一些非常有用的信息，这对于你想要触发其他提示或以不同方式响应的情况非常有用。

一旦你知道它是关于电视的，它就可以轻松地将其提取出来并将该信息传递给你的 Bot，然后以不同的方式回应，或者可以分开成不同的链条等等。

```
<langchain.graphs.networkx_graph.NetworkxEntityGraph object at 0x7f2bc65fe5e0>
[('Sam', 'Human', 'is'), ('TV', 'true', 'is broken'), ('TV', 'weird sounds', 'makes'), ('TV', 'black', 'goes'), ('TV', 'yes', 'is under warranty'), ('device', 'A512423', 'is under warranty')]
```

## 实体记忆 (Entity Memory)

这在做类似的事情时是类似的，只不过它现在正在寻找特定类型的实体。

是的，如果我们想要，我们甚至可以将一些示例放入提示中以进行此操作，所以阅读一下提示，提示的方式很有趣，我但基本上你有相同的对话链，我们传递这个提示，我们传递实例化的对话记忆 (Memory)。

还是跟以前一样先导入和实例化。

```
from langchain import OpenAI, ConversationChain
from langchain.chains.conversation.memory import ConversationEntityMemory
from langchain.chains.conversation.prompt import ENTITY_MEMORY_CONVERSATION_TEMPLATE
from pydantic import BaseModel
from typing import List, Dict, Any


llm = OpenAI(model_name='text-davinci-003', 
             temperature=0, 
             max_tokens = 256)

conversation = ConversationChain(
    llm=llm, 
    verbose=True,
    prompt=ENTITY_MEMORY_CONVERSATION_TEMPLATE,
    memory=ConversationEntityMemory(llm=llm)             
```

我们有相同的对话，我在这里进行了一些缩写，嗨，我是 Sam，我的电视坏了，但还在保修期内。

```
conversation.predict(input="Hi I am Sam. My TV is broken but it is under warranty.")

conversation.predict(input="How can I get it fixed. The warranty number is A512453")
```

所以现在我们可以看到它发现了实体 `['A512453']`。

```
# 打印 conversation.memory.entity_cache 的结果是 ['A512453']
```

就像哦，好的，它有 Sam，它有电视，对，然后它对此作出回应，你好，很抱歉听到你的电视坏了，有什么我可以帮你的吗？我如何修好它，保修号码是，然后我给了它。

所以你可以看到它从中提取了这个，并且在需要时它有了这个。


让我们继续进行下去，可以吗？

```python
conversation.predict(input="Can you send the repair person call Dave to fix it?.")
```

你能派人来维修吗，叫 Dave 来修吗？我们可以看到它弄清楚了。好的，那里有一个实体是 Dave，说可以，我可以派 Dave 来修你的电视，你能给我他的联系方式吗，所以它没有很明确地表示它应该是电视维修场所。

```
# 打印 conversation.memory.entity_cache 的结果是 ['Dave']
```

但无论如何最后，我们已经做了相当多的步骤，现在最后，我们可以看看，最后一个出现的实体是 ['Dave']。

我们也可以打印出整个内存存储在这里，它包含了所有的实体。

```
from pprint import pprint
pprint(conversation.memory.store)
```

这是 Sam 电视的保修号码，注意我从来没有说过这个，它重新表述了我所说的来做这个，它有 Dave，Dave 是一个维修人员，它有 Sam，Sam 拥有一台当前损坏但在保修期内的电视，然后我们有电视，电视在保修期内。

```
{'A512453': "A512453 is the warranty number for Sam's TV.",
 'Dave': 'Dave is a repair person.',
 'Sam': 'Sam owns a TV that is currently broken and under warranty.',
 'TV': 'TV is under warranty.'}
``` 

所以实体记忆 (Memory)对于提取关系之类的事物来说非常有用。

## 结论

也可以将不同的记忆 (Memory)结合在一起，放在这个之上，也许我们以后会看看这个，但这让你了解记忆 (Memory)是什么，以及它们在构建自己的对话代理时如何使用。

好的，如果这对你有帮助，请给个赞，我还有更多的 LangChain 文章，很快再更新。


## 参考资料

GitHub 仓库：https://github.com/liteli1987gmail/full_course_langchain/tree/main/code

## 学习交流
![](https://pic2.zhimg.com/80/v2-2307a70c1fcf93b9cd5e2a12d5114779_1440w.jpeg)


