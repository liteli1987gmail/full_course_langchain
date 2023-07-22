# 回调处理器

### 7.1.1 回调处理器的用途

你是否有过这样的经历：在开发一个复杂的链或者代理时，你需要在每个链的各个阶段都实时监控其状态，以便在出现问题时能迅速定位和解决。例如，你正在开发一个复杂的链，这个模型需要在给定的输入下，依次通过多个处理链条生成预期的输出。在这个过程中，模型每处理一个输入，都会消耗一定的令牌(token)。这时，希望能在执行过程中监控令牌消耗是很常见的业务需求, 你需要在各个阶段监控状态，出现问题时，快速定位解决。

没有合适的工具和方法，任务难以完成。可能要在链组件代码中添加监控令牌消耗的代码，过程繁琐，导致代码混乱难以维护。

LangChain 的回调处理器系统提供帮助。使用或自定义回调处理器，自动监控链条执行中的令牌消耗。令牌消耗超预期，处理器记录相关信息，可能触发其他操作，如发送警报或自动调整链的参数。方便有效管理模型和令牌消耗，避免大量的手动监控代码。

实际开发中，我们还想要记录模型输出到文件或数据库，数据分析和使用；或追踪模型运行时间，优化模型性能。没有合适的工具和方法，任务困难繁琐。

大语言模型开发中，回调处理器成为重要设计模式, 主要为我们的程序异常处理做了封装。在数据处理、网络请求、用户界面反馈，日志记录等场景下，回调处理器都能发挥作用。使用或自定义回调处理器，完成各种任务。例如，使用 FileCallbackHandler 记录模型输出到文件，或使用 TimeCallbackHandler 追踪模型运行时间。预设的回调处理器不满足需求，基于 BaseCallbackHandler 创建自定义的回调处理器也是可行的。

### 7.1.2 回调处理器的代码演示

例如，以下是一个使用 LangChain 回调处理器的代码片段：

```python
from langchain.callbacks import get_openai_callback
from langchain.llms import OpenAI

llm = OpenAI(temperature=0)
with get_openai_callback() as cb:
    llm("What is the square root of 4?")

total_tokens = cb.total_tokens
assert total_tokens > 0

# ...
```

这段代码中，首先创建了一个 OpenAI 模型实例，并通过 `get_openai_callback()` 函数获取一个回调处理器。然后，通过回调处理器，我们可以方便地获取模型执行的结果。

#### 异步的任务

但 LangChain 的回调处理器不仅可以处理同步的任务，也可以处理异步的任务，这大大增加了其灵活性。例如，以下的代码展示了如何使用回调处理器处理并发的任务：

```python
# ...

with get_openai_callback() as cb:
    await asyncio.gather(
        *[llm.agenerate(["What is the square root of 4?"]) for _ in range(3)]
    )

assert cb.total_tokens == total_tokens * 3

# ...
```

在这段代码中，我们创建了 3 个并发的任务，并使用同一个回调处理器来处理它们的结果。即使任务是并发的，回调处理器也能正确地处理每个任务的结果，并将它们汇总起来。

这就是 LangChain 回调处理器强大的功能。




## 7.2 集成应用

LangChain 回调处理器功能强大，实践应用广泛。和多服务商集成，扩展丰富。

ArgillaCallbackHandler 跟踪 LLM 输入输出，生成微调数据集。任务生成数据，问答，摘要，翻译等方面实用。

Context 理解用户与 AI 聊天产品交互，获得关键洞察，优化体验，最小化品牌风险。

PromptLayerCallbackHandler 可视化请求，版本提示，跟踪使用情况。

Streamlit 回调处理器帮助，展示 Agent 思考和行动在交互式的 Streamlit 应用中。

这些只是 LangChain 回调处理器实践中的例子，灵活性和功能强大，满足开发者处理各种复杂场景的需求。

LangChain 的回调处理器功能强大和灵活，提供方便的机制处理 AI 响应。同步任务，异步任务，通过回调处理器方便获取结果和处理错误。和多服务商集成，回调处理器在实践中发挥重要作用。
### 7.3.1 高级特性

LangChain 的回调处理器功能强大且灵活，它基于两个关键的基础类构建：`AsyncCallbackHandler` 和 `BaseCallbackHandler`。

各种特定类型的回调处理器，如 `FileCallbackHandler` 或 `StdOutCallbackHandler`，通常会继承 `BaseCallbackHandler` 并添加自己的特性。例如，`FileCallbackHandler` 就增加了将输出写入文件的功能。

`BaseCallbackHandler` 是一个基础类，提供了处理 AI 模型结果的基本功能。而 `AsyncCallbackHandler` 则是在 `BaseCallbackHandler` 的基础上，增加了异步处理的功能。这样，无论是同步的任务，还是异步的任务，都可以通过相应的回调处理器来处理。

### 7.3.2  自定义回调处理器

除了基本的功能之外，回调处理器还提供了一些高级特性，以满足开发者的需求。例如，你可以自定义回调处理器。以下是一个例子：

```python
import asyncio
from typing import Any, Dict, List

from langchain.chat_models import ChatOpenAI
from langchain.schema import LLMResult, HumanMessage
from langchain.callbacks.base import AsyncCallbackHandler, BaseCallbackHandler


class MyCustomSyncHandler(BaseCallbackHandler):
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        print(f"Sync handler being called in a `thread_pool_executor`: token: {token}")


class MyCustomAsyncHandler(AsyncCallbackHandler):
    """Async callback handler that can be used to handle callbacks from langchain."""

    async def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        """Run when chain starts running."""
        print("zzzz....")
        await asyncio.sleep(0.3)
        class_name = serialized["name"]
        print("Hi! I just woke up. Your llm is starting")

    async def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Run when chain ends running."""
        print("zzzz....")
        await asyncio.sleep(0.3)
        print("Hi! I just woke up. Your llm is ending")

chat = ChatOpenAI(
    max_tokens=25,
    streaming=True,
    callbacks=[MyCustomSyncHandler(), MyCustomAsyncHandler()],
)

await chat.agenerate([[HumanMessage(content="Tell me a joke")]])
```

在这个例子中，我们创建了两个个自定义的回调处理器，并在开始一个任务时，就打印一条消息 “Hi! I just woke up. Your llm is starting” 。任务结束之后，就打印另一条消息 “Hi! I just woke up. Your llm is ending”。

```
zzzz....
Hi! I just woke up. Your llm is starting
Sync handler being called in a `thread_pool_executor`: token: 
Sync handler being called in a `thread_pool_executor`: token: Why
Sync handler being called in a `thread_pool_executor`: token:  don
Sync handler being called in a `thread_pool_executor`: token: 't
Sync handler being called in a `thread_pool_executor`: token:  scientists
Sync handler being called in a `thread_pool_executor`: token:  trust
Sync handler being called in a `thread_pool_executor`: token:  atoms
Sync handler being called in a `thread_pool_executor`: token: ?
Sync handler being called in a `thread_pool_executor`: token:  


Sync handler being called in a `thread_pool_executor`: token: Because
Sync handler being called in a `thread_pool_executor`: token:  they
Sync handler being called in a `thread_pool_executor`: token:  make
Sync handler being called in a `thread_pool_executor`: token:  up
Sync handler being called in a `thread_pool_executor`: token:  everything
Sync handler being called in a `thread_pool_executor`: token: .
Sync handler being called in a `thread_pool_executor`: token: 
zzzz....
Hi! I just woke up. Your llm is ending
```
### 7.3.3  高级特性的应用

在构建 LangChain 程序时，其高级特性通常被应用在链组件和代理组件上。我们尝试将高级的回调处理器用在链和代理组件上。

现在假设你只想将 agent 的最终输出进行流式处理，那么你可以使用 `FinalStreamingStdOutCallbackHandler` 回调。因为我们通常不想代理在使用工具的时候，也进行流式处理，所以这个回调处理器很实用。

首先, 我们创建底层 LLM 时设置 ``streaming = True``，并传递一个新的 FinalStreamingStdOutCallbackHandler 实例。

```python
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.callbacks.streaming_stdout_final_only import (
    FinalStreamingStdOutCallbackHandler,
)
from langchain.llms import OpenAI

llm = OpenAI(
    streaming=True, callbacks=[FinalStreamingStdOutCallbackHandler()], temperature=0
)
```

在这个代码示例中，我们在创建 OpenAI 和 tools 时，都为其设置了同一个回调处理器 FinalStreamingStdOutCallbackHandler。而在创建 agent 时，我们又将这两个带有回调处理器的对象传递给了 agent。这样，无论是 agent，tools，还是 llm，都可以根据需要调用这个回调处理器。

```
tools = load_tools(["wikipedia", "llm-math"], llm=llm)
agent = initialize_agent(
    tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=False
)
agent.run(
    "It's 2023 now. How many years ago did Konrad Adenauer become Chancellor of Germany."
)
```
当我们通过 load_tools 函数加载工具时，我们将 llm 实例传递给了这些工具。因此，当这些工具在执行任务时，它们会使用 llm 实例，并因此也会调用这个回调处理器。

最后，当我们创建 agent 时，我们将这些工具和 llm 实例都传递给了 agent。因此，当 agent 执行任务时，它会使用这些工具和 llm 实例，从而也会调用这个回调处理器。

最后打印的结果是：

```
 Konrad Adenauer became Chancellor of Germany in 1949, 74 years ago in 2023.
```

以上就是 LangChain 回调处理器的一些高级特性和应用。可以看出，LangChain 回调处理器不仅提供了基本的处理功能，还提供了丰富的高级特性，以满足开发者的各种需求。这无疑增加了 LangChain 的使用价值，使其成为了构建和优化 AI 应用的强大工具。
