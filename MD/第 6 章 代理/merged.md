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
