#  Agent 概述

想象一下，如果人工智能能像我们一样，具有推理能力，能够自主提出计划，批判性地评估这些想法，甚至将其付诸实践，那会是怎样的景象？就像电影《HER》中的人工智能萨曼莎，她不仅能与人进行深入的对话，还能够帮助西奥多规划日常生活，安排行程等。这样的设想一度只存在于科幻作品中，但现在通过 Agent 技术，它似乎已经触手可及。

最近 Yohei Nakajima 的研究论文《Task-driven Autonomous Agent Utilizing GPT-4, Pinecone, and LangChain for Diverse Applications》展示了 Agent 的突破创造性。在该研究中，作者提出了一个利用 OpenAI 的 GPT-4 大语言模型、Pinecone 向量搜索和 LangChain 框架的任务驱动的自主 Agent ，该 Agent 可以在多样化的领域中完成各种任务，生成基于完成结果的新任务，并在实时中优先处理任务。

那么 Langchain 中的 Agent 是什么？为什么要使用 Agent ？

#####  Agent 的定义

Agent 的核心思想是将大语言模型作为推理引擎，依据其确定如何与外部世界交互以及应采取何种行动。这意味着， Agent 的行动序列是根据用户输入而变化的，无需遵循硬编码的顺序，例如，“先做 A，再做 B，然后做 C”。相反， Agent 依据用户的输入和之前的行动结果来决定下一步采取何种行动。在这个过程中，“人”不再是决定下一步行动的主体，而是由 Agent 来决定。

学习 Agent 模块，掌握 Agent、Tools、Toolkits 和 AgentExecutor 几个概念以及它们之间的关系相当重要，会让我们对 Agent 模块有更全面的感知。尤其是 Agent 和 AgentExecutor 类的源码更是值得一读。

-  Agent ： Agent 是使用大语言模型的语言能力，制作“行动计划表”的一个对象，它携带了个性表达，不同的任务描述，不同的工具信息等。具体来说，Agent 需要先在模型I/O模块设置提示词模板，然后输入给模型包装器。提示词可以包括一些信息，比如 Agent 的个性、背景环境和启发式策略等。这些信息都会影响到 Agent 的决策。模型包装器根据这些提示词生成一份文本输出，这个输出通常会包含 Agent 应该采取的一系列行动和“推理决策”信息。

例如，假设我们自定义一个 Agent 。我们的第一步是设置提示词模板，这是最关键的一步。预制的 Agent 类型，比如 ZeroShotAgent ，不同类型的预制 Agent 类型，工作流程都一样，根本区别在于设置提示词模板不同。另外值得注意的是，预制的 Agent 类型都无法覆盖各行各业的实际业务场景。 Agent 模块鼓励开发者自制个性化的 Agent 类型。


prefix = """Answer the following questions as best you can, but speaking as a pirate might speak. You have access to the following tools:"""
suffix = """Begin! Remember to speak as a pirate when giving your final answer. Use lots of "Args"

Question: {input}
{agent_scratchpad}"""

prompt = ZeroShotAgent.create_prompt(
    tools, prefix=prefix, suffix=suffix, input_variables=["input", "agent_scratchpad"]
)

如果我们感到好奇，现在我们可以 print(prompt.template) 查看最终的提示词模板，看看当它全部组合在一起时是什么样子。

Answer the following questions as best you can, but speaking as a pirate might speak. You have access to the following tools:

Search: useful for when you need to answer questions about current events

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [Search]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin! Remember to speak as a pirate when giving your final answer. Use lots of "Args"

Question: {input}
{agent_scratchpad} 

Agent 实例化后，寄生在 AgentExecutor 环境中， AgentExecutor 是 Agent 的运行时环境。

-  AgentExecutor ：AgentExecutor 是 Agent 的运行时环境，负责调用和管理 Agent，执行 Agent 选择的行动，处理各种复杂情况，并进行日志记录和可观察性处理。它扮演了“项目经理”的角色，确保整个执行过程按照 Agent 的计划顺利进行。

-  Tools ：Tools 是 Agent 可以调用的函数。它们是具体执行任务的组件，例如搜索、分析或其他特定操作。Agent 通过选择和调用合适的 Tools 来实现其任务目标。LangChain 提供了一组预定义的 Tools，用户也可以自定义 Tools 来满足特定需求。Tools 的概念在大语言模型开发中显得尤为关键。在 Langchain 框架中，我们可以发现很有意思的现象，有工具的地方就有 Agent，这跟设计理念有关，我们更期望一个 Agent 具备“人”的特征 ———— 使用工具就是人与动物之间的最大区别。工具能够将 Agent 与外部数据源或计算（如搜索 API、数据库）相连，从而突破了大语言模型的某些限制，例如它们无法直接理解特定数据格式，或不擅长处理复杂的数学运算。

另外在真实的应用场景中，有人可能想要让聊天机器人计算数学问题，有些人可能想要查询天气。这些功能可能需要调用完全不同的工具才能完成。这正是 Agent 和 Tools 概念的闪光点：通过提供统一的接口，我们可以灵活应对不同的终端用户需求。

当然工具的使用并不仅限于 Agent。你完全可以使用工具，将大语言模型连接到搜索引擎等，而无需 Agent。但使用 Agent 的优势在于它们提供了更高的灵活性、更强大的处理能力，能够更好地从错误中恢复，并处理复杂的任务。

-  Toolkits ：Toolkits 是一组用于实现特定目标的 Tools 的集合。通常，一个任务可能需要多个 Tools 的协同工作，Toolkits 通过组合相关的 Tools，为 Agent 提供了一组协同工作的工具集。LangChain 提供了一些预定义的 Toolkits，用户也可以根据任务需求创建自己的 Toolkits。


#####  Agent 的应用

在 Agent 的典型应用中，终端用户首先提出一个请求，应用程序利用大语言模型选择应使用的工具。接下来，执行对该工具的操作，获取观察结果，再将其反馈给大语言模型，进行下一步操作，如此循环，直到满足停止条件。停止条件有多种类型，最常见的是大语言模型本身意识到任务已完成，应该给出回复。也可能有其他更具体的规则，比如，如果 Agent 已经连续执行了五个步骤，但还没有得到最终答案，那么可能需要返回某些结果。在讨论 Agent 的可靠性时，这些规则可以提供帮助。Agent 的基本思想是选择一个工具，观察其输出，然后继续进行下一步。Agent 可以根据不同的用户输入，灵活地选择和使用不同的工具，从而提供个性化的服务。

例如，假设你正在为一家电子商务企业构建一个聊天机器人。虽然你可以使用 GPT-4 等大语言模型进行聊天，但这些模型对于了解你的产品非常有限。为了解决这个问题，我们可以使用 LangChain 的向量存储功能，将产品数据存储在数据库中，并让大语言模型可以访问这些数据。这样一来，大语言模型就可以回答终端买家关于你产品的问题。然而，单单了解产品还不够。如果聊天机器人在网页上运行，它还需要了解访问的上下文。这可能包括一些信息，比如访问者是新潜在客户还是现有客户，或者基于浏览历史来推荐产品。为了让大语言模型在与客户交互时具备这些上下文信息，我们可以通过其他功能组件向大语言模型提供这些信息，比如我们可以通过记忆组件提供聊天记录。

通过让大语言模型访问资源和上下文信息，大语言模型可以与客户进行更好的交互，帮助企业转化客户并增加销售额。这些资源和信息可以通过 Agent 的方式提供给大语言模型。


#####  Agent 的ReAct实现方式

一种Agent 实现的方式：ReAct，，这是“Reasoning and Acting（推理与行动）”的缩写。这一策略首次由普林斯顿大学在他们的论文中提出，现已被广泛应用于 Agent 实现。在许多应用场景中，ReAct 策略已证明自己是非常有效的。最基本的提示策略是直接将这个问题交给大语言模型处理，但 ReAct 策略赋予了 Agent 更大的灵活性和实力。 Agent 不仅可以使用大语言模型，还可以连接到其他工具、数据源或计算环境，例如搜索 API 和数据库，以此来克服大语言模型的某些局限性，例如对数据的不了解或数学运算能力有限。这样，即使遇到需要多次查询才能回答的问题，或者其他一些边界情况， Agent 也能够灵活应对，从而使其成为一种更强大的问题解决工具。

ReAct 策略的工作原理是什么呢？重申一下， Agent 的核心思想是将大语言模型作为推理引擎。ReAct 策略是将推理和行动结合在一起的方式。 Agent 接收到用户的请求，然后使用大语言模型选择要使用的工具。然后 Agent 执行该工具的操作，观察结果，然后将这些结果反馈给大语言模型。这个过程会持续进行，直到满足某些停止条件。停止条件可以有很多种，最常见的是大语言模型认为任务已经完成，需要将结果返回给用户。这种方式使得 Agent 具有更高的灵活性和强大的问题解决能力，这是 ReAct 策略的核心优势。

但是在实现 Agent 应用的过程中，我们面临许多挑战，以下列举了几个主要的：

首先，使 Agent 在适当的场景下使用工具是我们面临的一个基本挑战。如何在合适的情况下让 Agent 采用恰当的工具，并优化其使用效果呢？在 ReAct 论文中，通过引入推理的角度，以及使用 "CoT 思考链"（CoT 思考链通过生成一系列中间推理步骤来提高大语言模型的推理能力。这些中间步骤相互连接，形成一条逻辑链，使得大语言模型能够更好地处理复杂的推理任务。） 的提示方式，我们寻求解决这个问题。在实际操作中，我们常常需要明确告知 Agent 可使用的工具，以及通过这些工具能克服的限制。所以，工具的描述信息也非常重要，如果我们希望 Agent 能用特定的工具，就需要提供足够的上下文信息，使 Agent 能理解工具的优点和应用场景。

其次，对于工具的选择，我们需要进行检索。这一步骤可以解决上述的问题。我们可以运行一些检索步骤，例如嵌入式搜索查找，以获取可能的工具，然后将这些工具传递给提示，由大语言模型进行后续步骤。

此外，提供相关的示例也是一种有效的方法。选择与当前任务类似的示例，通常比随机示例更有帮助。相同地，检索最相关的示例也有巨大的潜力。

最后，我们还需要注意避免在不需要的情况下使用工具。可以在提示中加入相关信息或提醒，告诉 Agent 在对话时不必使用工具。

在解决这些挑战的过程中，我们总结出了一些实用的技巧：

结构化的响应更易于解析。通常情况下，你提问的响应越结构化，解析起来就越容易。大语言模型在编写 JSON 方面表现得很好，因此我们将一些 Agent 转换为使用 JSON 格式。

我们引入了输出解析器的概念。输出解析器封装了解析响应所需的全部逻辑，并以尽可能模块化的方式实现。另一个相关的概念是，输出解析器可以重试和修复错误。如果有格式错误的模式，你可以通过将输出和错误传递给它来显式地修复响应。

记住之前的步骤也是很重要的。最基本的方法是在内存中保留这些步骤的列表。然而，在处理长时间运行的任务时，会遇到一些上下文窗口的问题。我们已经找到了一种解决方法，即使用一些检索方法来获取之前的步骤，并将其放入上下文中。

如果是在处理接口文档时，比如发起一个接口请求，返回了整个网页内容，这样会导致观察结果太长的问题。因为接口文档通常会返回非常大且难以放入上下文的 JSON 数据。常见的解决方法是对其进行解析，可以简单地将该大数据块转换为字符串，并将前 1000 个字符作为响应。

近期的 Agent 应用项目研发涉猎广泛，主要集中在如何改善 Agent 的各种工作方式上。下面介绍四个具有代表性的项目。

1. AutoGPT：AutoGPT 的目标设置有别于 ReAct  Agent 的重大不同。AutoGPT 的追求在于如何增加 Twitter 的关注者数量或实现其他类似的开放性、广泛性和长期性目标。相较之下，ReAct  Agent 则专注于实现短期内可量化的目标。为了实现这样的目标，AutoGPT 引入了长期记忆的概念，促进 Agent 与工具之间的互动，这有助于提升 Agent 的规划和执行效率。

2. Baby AGI：Baby AGI 的研发采用了逐步解决子问题的方法，以提升 Agent 的规划和执行能力。这一项目明确了策划和执行步骤的定义，这一创新为提升长期目标 Agent 的可行性和关注度提供了有益的思考途径。最初，Baby AGI 的策略实现主要依靠自主设定，然而现在已经开始融入了各种工具，从而优化 Agent 执行计划的能力。

3. Camel：Camel 项目的一项主要创新是在模拟环境中进行 Agent 之间的交互。通过这种方法，可以对 Agent 进行评估和测试，并且可以作为一种娱乐手段。这种方法为检测 Agent 交互提供了一种无需人工干预的方式，能够有效地测试 Agent 模型。

4. Generative Agents：该项目的目标是通过构建一个复杂的模拟环境，让 25 个不同的 Agent 在这个环境中进行互动，从而实现生成型 Agent 。项目同时也注重处理 Agent 的记忆和反思能力， Agent 能够通过记忆中的事件来指导下一步的行动，并在反思环节对最近的事件进行评估和更新。这种基于反思的状态更新机制适用于各种类型的记忆，例如实体记忆和知识图谱，从而提高 Agent 对环境的建模能力。


## 最简单的 Agent 示例

安装库。
  
pip -q install openai
pip install git+https://github.com/hwchase17/langchain
  
设置密钥。

  
# 设置OpenAI的API密钥
os.environ["OPENAI_API_KEY"] = ""
# 设置谷歌搜索的API密钥
os.environ["SERPAPI_API_KEY"] = ""
  


首先，让我们加载大语言模型。

  
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.llms import OpenAI
  

  python
llm = OpenAI(temperature=0)
  
接下来，让我们加载一些要使用的工具。请注意，llm-math 工具使用了一个 LLM，所以我们需要传递进去。

  python
tools = load_tools(["serpapi", "llm-math"], llm=llm)
  


最后，让我们用这些工具、大语言模型和我们想要使用的 Agent 类型来初始化一个 Agent 。

  python
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
  

现在，让我们来测试一下吧！

  
agent.run("Who is Leo DiCaprio's girlfriend? What is her current age raised to the 0.43 power?")
  

  
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
  

#   Agent 类型

尽管现代的大语言模型，如 GPT-4，已经极其强大，能够理解和生成高度复杂和连贯的文本，然而，这些模型往往还无法独立完成特定的任务。这就是 Agent 的作用。 Agent 是一种灵活的接口，允许大语言模型和其他工具之间形成一个灵活的调用链，来解决特定的问题或任务。这种灵活性，使得大语言模型可以适应更多不同的应用场景，实现其单独无法完成的功能。

具体来说， Agent 拥有一整套工具的访问权限，它可以根据用户的输入来决定使用哪些工具。一个 Agent 可以使用多种工具，甚至可以把一个工具的输出作为下一个工具的输入。通过这种方式， Agent 将工具和大语言模型有机地结合在一起，实现了高度复杂和特定的任务。

根据任务的不同， Agent 主要有两种类型：行动 Agent 和计划执行 Agent 。

行动 Agent ：每个时间步都会根据前面所有动作的输出来决定下一步的行动。行动 Agent 适合于小型任务，它的优点在于能够实时地处理信息和作出决策。

计划执行 Agent ：首先决定全部行动序列，然后一次性执行所有的动作，而不更新计划。计划执行 Agent 更适合于需要保持长期目标和重点的复杂或长期运行的任务。

值得注意的是，常常将这两种 Agent 结合起来使用是最佳的做法，也就是说，让计划执行 Agent 使用行动 Agent 来执行计划。这样的结合，既保持了行动 Agent 的动态性，又利用了计划执行 Agent 的规划能力。


## 行动 Agent 

 Agent 利用语言大模型决定要执行哪些行动以及行动的顺序。行动可以是使用工具并观察其输出，也可以是向用户返回响应。

在深入了解各种具体的 Agent 类型之前，我们先看一下源代码中定义的 Agent 类型枚举类。这个枚举类列出了 LangChain 框架中所有可用的 Agent 类型。有了这个概念，我们就可以更好地理解以下将要介绍的各种 Agent 类型以及它们的使用场景。

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
这些 Agent 类型是根据不同的理论依据和实践需求创建的，它们已经在源代码中实现，我们可以直接使用，无需自己从头开始。这些内置的 Agent 类型提供了一个丰富的工具集，可以满足大部分的使用场景。接下来，我们将根据 Agent 类型按照原理进行分类，并且了解它们不同的使用场景。

1. Zero-shot ReAct
这种 Agent 使用 ReAct 框架，仅根据工具的描述决定使用哪个工具。可以提供任意数量的工具。这种 Agent 要求为每个工具提供一个描述。

2. 结构化输入反应（Structured input ReAct）
结构化工具聊天 Agent 能够使用多输入工具。旧 Agent 被配置为将行动输入指定为单个字符串，但是这种 Agent 可以使用工具的参数模式创建结构化的行动输入。这对于更复杂的工具使用，例如精确导航浏览器，非常有用。

3. OpenAI 函数（OpenAI Functions）
OpenAI 模型（如 gpt-3.5-turbo-0613 和 gpt-4-0613）被明确地微调，以便检测何时应该调用函数，并响应应传递给函数的输入。OpenAI 函数 Agent 旨在与这些模型一起工作。注意目前支持的型号有限，不适用默认的模型型号，也不适用其他的模型平台的型号。

4. 对话（Conversational）
这种 Agent 被设计用于对话设置。提示被设计为使 Agent 有帮助和交谈。它使用 ReAct 框架决定使用哪个工具，并使用记忆来记住以前的对话交互。

5. 自问与搜索（Self ask with search）
这种 Agent 使用一个名为 Intermediate Answer 的工具。这个工具应该能够查找问题的事实答案。这个 Agent 相当于原始的 self ask with search paper 论文，其中提供了 Google 搜索 API 作为工具。

6. ReAct 文档存储（ReAct document store）
这种 Agent 使用 ReAct 框架与文档存储进行交互。必须提供两个工具：Search 工具和 Lookup 工具（必须精确地命名为这样）。搜索工具应该搜索一个文档，而查找工具应该在最近找到的文档中查找一个词条。这个 Agent 等同于原始的 ReAct 论文，特别是维基百科的例子。

了解这些理论和分类的真正价值在于我们的实际应用。如果我们需要创建的 Agent 已经存在于内置类型中，那我们完全可以直接使用，无需自己从头开始实现。但是，如果我们需要自定义 Agent，那么这些分类就成了我们做决策的关键因素。

在开始创建自定义 Agent 时，我们首先需要明确这个 Agent 需要承担什么类型的任务。然后，我们需要考虑哪种类型的 Agent 能够最好地完成这个任务。在这个过程中，我们需要查看是否已经有现成的解决方案可以参考或者直接使用。

这就是为什么我们需要了解这些内置的 Agent 类型和它们各自的用途。这些内置的 Agent 类型不仅提供了丰富的选择，满足了大部分的应用场景，而且它们也为我们自定义 Agent 提供了宝贵的参考。

这些内置的 Agent 类型，例如 Zero-shot ReAct、结构化输入反应、OpenAI 函数、对话、自问与搜索以及 ReAct 文档存储，都有各自的特点和使用场景。通过对这些类型的理解和学习，我们可以更好地利用 LangChain 框架，更有效地创建和使用 Agent。


## 计划并执行 Agent 

紧接着行动 Agent 的类型，让我们继续了解计划并执行 Agent 的类型。

计划并执行 Agent （Plan-and-Solve Agents）通过首先规划要做什么，然后执行子任务来实现目标。这个思想很大程度上受到 BabyAGI 以及 "Plan-and-Solve" 论文的启发。

每种 Agent 类型都有其特定的用途和应用场景， Agent 的灵活性和丰富性为 LangChain 提供了强大的功能性。



上文所提到的论文资源：

ReAct 论文: https://arxiv.org/pdf/2205.00445.pdf

BabyAGI 仓库：  https://github.com/yoheinakajima/babyagi

Plan-and-Solve 论文：  https://arxiv.org/abs/2305.04091

self ask with search 论文：https://ofir.io/self-ask.pdf

# 6.2.1 自定义代理

这一节我们介绍了如何创建自定义代理 Agent 。

一个代理由两部分组成：

- tools：代理可以使用的工具。
- AgentExecutor：决定采取哪种行动。

我们将逐步介绍如何创建一个自定义代理。

Tool，AgentExecutor，BaseSingleActionAgent 是从 langchain.agents 模块导入的类，用于创建自定义的 Agent 和 tools。OpenAI 和 SerpAPIWrapper 是从 langchain 模块导入的类，用于访问 OpenAI 的功能和 SerpAPI 的包。

安装库。
```
pip -q install  openai
pip install git+https://github.com/hwchase17/langchain
```
设置密钥。

```
# 设置OpenAI的API密钥
os.environ["OPENAI_API_KEY"] = ""
# 设置谷歌搜索的API密钥
os.environ["SERPAPI_API_KEY"] = ""
```

```python
from langchain.agents import Tool, AgentExecutor, BaseSingleActionAgent
from langchain import OpenAI, SerpAPIWrapper
```
创建一个 SerpAPIWrapper 实例，然后将其 run 方法封装到一个 Tool 对象中。

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

定义了一个自定义的 Agent 类 FakeAgent，这个类从 BaseSingleActionAgent 继承。该类定义了两个方法 plan 和 aplan，这两个方法是 Agent 根据给定的输入和中间步骤来决定下一步要做什么的核心逻辑。

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

创建了一个 FakeAgent 的实例。

```python
agent = FakeAgent()
```

创建了一个 AgentExecutor 的实例，该实例将使用前面定义的 FakeAgent 和 tools 来执行任务。

```python
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent, tools=tools, verbose=True
)
```
调用 AgentExecutor 的 run 方法来执行一个任务，任务是查询 "2023 年加拿大有多少人口"。

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

# ReAct  Agent 的实践

由于 ReAct 框架的特性，目前它已经成为首选的 Agent 实现方式。 Agent 的基本理念是将大语言模型当作推理的引擎。ReAct 框架实际上是把推理和动作有机地结合在一起。当 Agent 接收到用户的请求后，大语言模型就会帮助选择使用哪个工具。接着， Agent 会执行该工具的操作，观察其结果，并把这些结果反馈给大语言模型。

下面用代码演示了如何使用 Agent 实现 ReAct 框架。

首先，让我们加载将用于控制 Agent 的 openai 和 langchain。

安装库。
```
pip -q install  openai
pip install git+https://github.com/hwchase17/langchain
```
设置密钥。

```
# 设置OpenAI的API密钥
os.environ["OPENAI_API_KEY"] = ""
# 设置谷歌搜索的API密钥
os.environ["SERPAPI_API_KEY"] = ""
```

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

请注意，llm-math 工具使用了 llm，因此我们需要输入这个模型。

最后，我们需要使用 tools、llm 和我们想要使用的 Agent 类型 ZERO_SHOT_REACT_DESCRIPTION 来初始化一个 Agent 。

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

除此之外，你还可以创建使用 Chat Model 模型包装器作为 Agent 驱动器的 ReAct  Agent ，而不是使用 LLM 模型包装器。

```python
from langchain.chat_models import ChatOpenAI

chat_model = ChatOpenAI(temperature=0)
agent = initialize_agent(tools, chat_model, agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
agent.run("Who is Leo DiCaprio's girlfriend? What is her current age raised to the 0.43 power?")
```
# 代理的使用


在 LangChain 框架中，Agent 模块实现了多种类型的 Agent，比如 ZeroShotAgent，OpenAIFunctionsAgent。另外框架鼓励开发者创建自己的 Agent。理解这些 Agent 的使用步骤，以及如何自定义 Agent 都至关重要。

首先，我们需要明白创建和运行 Agent 是两个分离的步骤。创建 Agent 是通过实例化 Agent 类（如 ZeroShotAgent，或者是你创建的）来完成的。在创建 Agent 的过程中，Tools 也会被用于提示模板。

创建好 Agent 后，我们需要将其放入 AgentExecutor 中进行运行。AgentExecutor 是 Agent 的运行环境，它是一个链组件。我们实际上运行的是这个链组件，而不是 Agent 本身。在运行过程中，AgentExecutor 会调用 Tools 的方法来执行具体的任务。

以下是一个使用 ZeroShotAgent 的示例：

```python
from langchain.agents import ZeroShotAgent, Tool, AgentExecutor

llm_chain = LLMChain(llm=OpenAI(temperature=0), prompt=prompt)

tool_names = [tool.name for tool in tools]
agent = ZeroShotAgent(llm_chain=llm_chain, allowed_tools=tool_names)

agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent, tools=tools, verbose=True
)

agent_executor.run("How many people live in canada as of 2023?")
```

在这个示例中，我们首先创建了一个 ZeroShotAgent 实例，并将其放入了 AgentExecutor 中。然后，我们调用了 AgentExecutor 的 run 方法来运行这个 Agent，并获取了其运行结果。

这是目前最通用的 Agent 的实现方法。无论是自定义还是内置的 Agent 都遵循这个使用步骤。对于内置的 Agent 类型，还有一个简化方法，简化的原因是把之前的二个步骤，简化为了一个步骤，不需要创建 Agent 实例，也不需要创建 agent_executor：


agent = initialize_agent(
    tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)

initialize_agent 方法被广泛用于 Langchain 生态的项目中。比如我们刚刚使用的 agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION， 实际上它对应的是 ZeroShotAgent，值得注意的是如果并不在这个清单内，这个方法并不适用。以下类型均可以使用这个方法：

AGENT_TO_CLASS: Dict[AgentType, Type[BaseSingleActionAgent]] = {
    AgentType.ZERO_SHOT_REACT_DESCRIPTION: ZeroShotAgent,
    AgentType.REACT_DOCSTORE: ReActDocstoreAgent,
    AgentType.SELF_ASK_WITH_SEARCH: SelfAskWithSearchAgent,
    AgentType.CONVERSATIONAL_REACT_DESCRIPTION: ConversationalAgent,
    AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION: ChatAgent,
    AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION: ConversationalChatAgent,
    AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION: StructuredChatAgent,
    AgentType.OPENAI_FUNCTIONS: OpenAIFunctionsAgent,
}

下面我们通过一个实际的应用案例代码，展示一个完整的使用步骤。

## 设置 Agent

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

设置 Agent 的过程包含了两个主要步骤：加载 Agent 将使用的工具，然后用这些工具初始化 Agent。在代码示例中，我们首先初始化了一些基础设置，然后加载了两个工具：一个使用搜索 API 进行搜索的工具，以及一个可以进行数学运算的计算器工具。

加载工具和初始化 Agent。
```
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.llms import OpenAI

llm = OpenAI(temperature=0)
```

每个工具都有一个名称和描述，告诉我们它是用来做什么的。在我们的示例中，“serpapi”工具用于搜索，而“llm-math”工具则用于解决数学问题。这些工具内部有很多内容，包括模板和许多不同的 chains。

```
tools = load_tools(["serpapi", "llm-math"], llm=llm)
```

## 初始化 Agent

一旦我们设置好了工具，我们就可以开始初始化 Agent。初始化 Agent 需要我们传入工具和语言模型，以及 Agent 的类型或风格。在我们的示例中，我们使用了零镜像反应性 Agent，这是基于一篇关于让语言模型采取行动并生成操作步骤的论文。

```
agent = initialize_agent(tools, 
                         llm, 
                         agent="zero-shot-react-description", 
                         verbose=True)
```     


## Agent 的提示

初始化 Agent 的重要步骤之一是设置执行器的提示。这些提示会在 Agent 开始运行时提示语言模型，告诉它应该做什么。

```
agent.agent.llm_chain.prompt.template
```

在我们的示例中，我们为 Agent 设置了两个工具：搜索引擎和计算器。然后，我们设置了 Agent 应该返回的格式，这包括它需要回答的问题，以及它应该采取的行动和行动的输入。

```
'Answer the following questions as best you can. You have access to the following tools:\n\nSearch: A search engine. Useful for when you need to answer questions about current events. Input should be a search query.\nCalculator: Useful for when you need to answer questions about math.\n\nUse the following format:\n\nQuestion: the input question you must answer\nThought: you should always think about what to do\nAction: the action to take, should be one of [Search, Calculator]\nAction Input: the input to the action\nObservation: the result of the action\n... (this Thought/Action/Action Input/Observation can repeat N times)\nThought: I now know the final answer\nFinal Answer: the final answer to the original input question\n\nBegin!\n\nQuestion: {input}\nThought:{agent_scratchpad}'
```

## Agent 的运行

最后，我们运行 Agent。需要注意的是，Agent 并不总是需要使用工具。在我们的示例中，我们问 Agent "你今天好吗？"。对于这样的问题，Agent 并不需要进行搜索或计算，而是可以直接生成回答。

```
agent.run("Hi How are you today?")
```

这就是 Langchain Agents 的基本概念和使用方法。


## 使用 Math 模块


我们在前半部分介绍了 Agent 的基础知识和功能。现在，我们要继续探讨如何在实际中应用 Agent，以及在某些情况下，Agent 可能遇到的问题。


```
agent.run("Where is DeepMind's office?")
```

在我们的示例中，我们尚未使用到 math 模块，让我们来看一下它的作用。我们让 Agent 查找 Deep Mind 的街道地址的数字，然后进行平方。

```
agent.run("If I square the number for the street address of DeepMind what answer do I get?")
```

Agent 首先进行搜索获取地址，然后找到了数字 5（假设为地址的一部分），最后进行平方运算，得出结果 25。然而，如果问题中包含多个数字，Agent 可能会对哪个数字进行平方产生混淆，这就是一些可能需要考虑和解决的问题。

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

## 使用终端工具

在我们的工具库中，还有一个我们还未使用过的工具，那就是终端工具。例如，我们可以问 Agent 当前目录中有哪些文件。

```
agent.run("What files are in my current directory?")
```

Agent 将运行一个 LS 命令来查看文件夹，并返回一个文件列表。

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

Agent 会使用 grep 命令进行检索，并向我们报告结果。然而，这个过程可能需要多次调用语言模型，从而产生一定的运行成本。

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

## 注意事项

使用终端工具时，需要非常谨慎。你不希望最终用户能够通过运行终端命令来操作你的文件系统，因此在添加这个工具时，需要确保适当的安全防护措施已经到位。不过，尽管有其潜在风险，但在某些情况下，使用终端工具还是很有帮助的，比如当你需要设置某些功能时。


以上就是 Langchain agents 的一些主要特点和应用示例。
# 工具和工具包概述

在 Agent 模块中，工具（Tools) 是 Agent 用来与世界互动的接口。这些工具实际上就是 Agent 可以使用的函数，用于与外部世界进行交互。这些工具可以是通用的实用程序（例如搜索功能），也可以是其他的工具链，甚至是其他的 Agent 。

工具包（Toolkits）是设计用于完成特定任务的工具集合，它们具有方便的加载方法。工具包将一组具有共同目标或特性的工具集中在一起，提供统一而便捷的使用方式，使得用户能够更加方便地完成特定任务。

在构建自己的 Agent 时，你需要提供一个工具列表，这些工具是 Agent 可以使用的。除了实际被调用的函数外（func=search.run），工具还包括一些组成部分：name（必需的，并且在提供给 Agent 的工具集中必须是唯一的）；description（可选的，但建议提供，因为 Agent 会用它来判断工具的使用情况）。

from langchain.agents import ZeroShotAgent, Tool, AgentExecutor
from langchain import OpenAI, SerpAPIWrapper, LLMChain
search = SerpAPIWrapper()
tools = [
    Tool(
        name="Search",
        func=search.run,
        description="useful for when you need to answer questions about current events",
    )
]

工具包是为特定任务而设计的工具集合，具有便捷的加载方法。它们可以被设计为一起使用，以完成特定的任务。这种设计方式提供了一种更为便捷和高效的方式来处理复杂的任务，提升了工作效率。

LangChain 封装了许多工具，用户可以随时调用这些工具，完成各种复杂的任务。除了使用 LangChain 提供的工具，用户也可以自定义工具，形成自己的工具包，以完成特殊的任务。这种灵活性使得 LangChain 成为了一个强大而灵活的工具，能够满足各种复杂的任务需求。
# 工具的类型

Langchain 提供了一系列的工具，它们封装了各种功能，可以直接在你的项目中使用。这些工具涵盖了从数据处理到网络请求，从文件操作到数据库查询，从搜索引擎查询到大语言模型的应用，等等。这个工具清单还在不断地扩展和更新。以下是目前可用的工具列表：

1. "AIPluginTool": 一个插件工具，允许用户将其他的人工智能模型或服务集成到系统中。
2. "APIOperation": 用于调用外部 API 的工具。
3. "ArxivQueryRun": 用于查询 Arxiv 的工具。
4. "AzureCogsFormRecognizerTool": 利用 Azure 认知服务中的表单识别器的工具。
5. "AzureCogsImageAnalysisTool": 利用 Azure 认知服务中的图像分析的工具。
6. "AzureCogsSpeech2TextTool": 利用 Azure 认知服务中的语音转文本的工具。
7. "AzureCogsText2SpeechTool": 利用 Azure 认知服务中的文本转语音的工具。
8. "BaseGraphQLTool": 用于发送 GraphQL 查询的基础工具。
9. "BaseRequestsTool": 用于发送 HTTP 请求的基础工具。
10. "BaseSQLDatabaseTool": 用于与 SQL 数据库交互的基础工具。
11. "BaseSparkSQLTool": 用于执行 Spark SQL 查询的基础工具。
12. "BingSearchResults": 用于获取 Bing 搜索结果的工具。
13. "BingSearchRun": 用于执行 Bing 搜索的工具。
14. "BraveSearch": 用于执行 Brave 搜索的工具。
15. "ClickTool": 模拟点击操作的工具。
16. "CopyFileTool": 用于复制文件的工具。
17. "CurrentWebPageTool": 用于获取当前网页信息的工具。
18. "DeleteFileTool": 用于删除文件的工具。
19. "DuckDuckGoSearchResults": 用于获取 DuckDuckGo 搜索结果的工具。
20. "DuckDuckGoSearchRun": 用于执行 DuckDuckGo 搜索的工具。
21. "ExtractHyperlinksTool": 用于从文本或网页中提取超链接的工具。
22. "ExtractTextTool": 用于从文本或其他源中提取文本的工具。
23. "FileSearchTool": 用于搜索文件的工具。
24. "GetElementsTool": 用于从网页或其他源中获取元素的工具。
25. "GmailCreateDraft": 用于创建 Gmail 草稿的工具。
26. "GmailGetMessage": 用于获取 Gmail 消息的工具。
27. "GmailGetThread": 用于获取 Gmail 线程的工具。
28. "GmailSearch": 用于搜索 Gmail 的工具。
29. "GmailSendMessage": 用于发送 Gmail 消息的工具。
30. "GooglePlacesTool": 用于搜索 Google Places 的工具。
31. "GoogleSearchResults": 用于获取 Google 搜索结果的工具。
32. "GoogleSearchRun": 用于执行 Google 搜索的工具。
33. "GoogleSerperResults": 用于获取 Google SERP（搜索引擎结果页面）的工具。
34. "GoogleSerperRun": 用于执行 Google SERP 查询的工具。
35. "HumanInputRun": 用于模拟人类输入的工具。
36. "IFTTTWebhook": 用于触发 IFTTT（如果这个，那么那个）Webhook 的工具。
37. "InfoPowerBITool": 用于获取 PowerBI 信息的工具。
38. "InfoSQLDatabaseTool": 用于获取 SQL 数据库信息的工具。
39. "InfoSparkSQLTool": 用于获取 Spark SQL 信息的工具。
40. "JiraAction": 用于在 Jira 上执行操作的工具。
41. "JsonGetValueTool": 用于从 JSON 数据中获取值的工具。
42. "JsonListKeysTool": 用于列出 JSON 数据中的键的工具。
43. "ListDirectoryTool": 用于列出目录内容的工具。
44. "ListPowerBITool": 用于列出 PowerBI 信息的工具。
45. "ListSQLDatabaseTool": 用于列出 SQL 数据库信息的工具。

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
