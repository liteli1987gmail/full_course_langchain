# 一文学会 OpenAI 的函数调用功能 Function calling

> 函数调用 （Function Calling)提供了一种将 GPT 的能力与外部工具和 API 相连接的新方法。

在这个文章中，我想向您展示OpenAI模型的函数调用 （Function Calling)能力，并向您展示如何将此新功能与 Langchain 集成。

我将通过以下代码详细介绍这个工作原理，开始吧！

### 安装包

```bash
!pip install --upgrade langchain
!pip install python-dotenv
!pip install openai
```
*注意：langchain的版本不低于0.0.200， 之前的版本尚不支持函数调用 （Function Calling)*

可以调用下面的`print_version`函数 （Function Calling) 看看自己目前的langchain的版本是否大于0.0.200。

```
import pkg_resources


def print_version(package_name):
    try:
        version = pkg_resources.get_distribution(package_name).version
        print(f"The version of the {package_name} library is {version}.")
    except pkg_resources.DistributionNotFound:
        print(f"The {package_name} library is not installed.")


print_version("langchain")

```

```
The version of the langchain library is 0.0.205.
```

### 连接 OpenAI

我们加载 API 密钥，并将OpenAI模块的 API, 密钥属性设置为环境变量的值，这样我们就可以连接到 OpenAI了。

```python
from dotenv import load_dotenv
import os
import openai
import json

load_dotenv()
openai.api_key = os.environ.get("OPENAI_API_KEY")

```

首先，为了使函数 （Function Calling) 工作，我们假设有一个函数 `get_pizza_info`可以获取披萨信息，传入一个字符串参数，它是披萨的名称 `pizza_name`，例如 Salami 披萨或其他任何披萨，然后您会得到一个固定的价格，本例中始终为 10.99，最后返回字符串JSON。

### 定义一个 Function Calling

```python
def get_pizza_info(pizza_name: str):
    pizza_info = {
        "name": pizza_name,
        "price": "10.99",
    }
    return json.dumps(pizza_info)
```
现在开始切入主题，我们来提供这个 `get_pizza_info` 函数的描述。

```python
functions = [
    {
        "name": "get_pizza_info",
        "description": "Get name and price of a pizza of the restaurant",
        "parameters": {
            "type": "object",
            "properties": {
                "pizza_name": {
                    "type": "string",
                    "description": "The name of the pizza, e.g. Salami",
                },
            },
            "required": ["pizza_name"],
        },
    }
]
```
### 为什么必须提供一个函数 （Function Calling) 描述？

必须提供一个描述，这对于 llm 非常重要， llm 用函数描述来识别函数 （Function Calling) 是否适合回答用户的请求。

在参数字典中，我们必须提供多个信息，例如在属性字典中，有关披萨名称的信息，并且还必须提供类型和描述。

这对于 llm 获取关于这个函数 （Function Calling) 的信息非常重要。

我们还必须将`pizza_name`参数设置为 `required`，因为我们这里没有默认值，所以这是 llm 了解如何处理函数 （Function Calling) 的信息。

### 运行 OpenAI 的 Function Calling

在提供了这种信息之后，我们必须使用它，我这里定义了一个名为 chat 的小助手函数 （Function Calling) 。

```python
def chat(query):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        messages=[{"role": "user", "content": query}],
        functions=functions,
    )
    message = response["choices"][0]["message"]
    return message
```    

 chat 的小助手函数 （Function Calling) 只接受一个名为 query 的参数，这将是一个字符串，这将是用户的请求，首先我们必须定义 API  Call，我们通过 `ChatCompletion`类从 OpenAI 进行调用，我们在这里调用 `create` 函数 。

*注意： 使用最新的模型非常重要，所有以 0613 结尾的模型，例如 `gpt-3.5-turbo-0613`, 如果是 gpt4 也是需要 0613 结尾的模型。*

然后，我们必须提供消息`messages`，这将是一个字典，其中包含角色`role`，角色当前为 `user`，并且我们提供内容`content`，该内容将被传递给此处的 chat 函数，然后作为附加参数，您必须提供这些函数的参数`functions=functions,` (functions变量是我们上一步定义的函数 描述列表，这个列表中我们暂时只定义了一个函数描述）。

通过调用这个 chat 函数  ，我们将获得一个响应，响应是一个更复杂的对象，我们可以通过调用响应来获取实际的消息，然后有一个名为 `choices` 的字典，这是一个我们尝试检索第一个元素的列表，这将是另一个字典，我们将检索消息，所以有很多东西，比如核心令牌等。

现在我们可以问模型法国的首都是什么，如果您运行，我们会得到一个答案。

```python
chat("What is the capital of france?")
```

```
<OpenAIObject at 0x166d47a42f0> JSON: {
  "role": "assistant",
  "content": "The capital of France is Paris."
}
```

这将给我们一个返回的 `openai` 对象，其中包含`content`，答案在这里，法国的首都是巴黎，角色是助手，系统始终是 AI 的角色，用户始终是发出请求的用户 `user` 的角色，这非常好，因为 API 已经提供了一个 JSON 对象。

那么当我们问披萨的价格是多少时会发生什么，这显然与比萨有关了。我们希望LLM 回答我们的是关于披萨的信息。

```python
query = "How much does pizza salami cost?"
message = chat(query)
message
```
我们还向 LLM 提供了应该使用 `get_pizza_info` 函数 （Function Calling) 的信息，这里的描述与比萨有关。如果您现在运行，它应该看起来不同，所以现在输出看起来非常不同。

运行代码后你会大吃一惊，内容中没有任何东西，只是 null。

```
<OpenAIObject at 0x166d47a43b0> JSON: {
  "role": "assistant",
  "content": null,
  "function_call": {
    "name": "get_pizza_info",
    "arguments": "{\n\"pizza_name\": \"Salami\"\n}"
  }
}
```

但是我们有此附加的 `function_call` 对象，它是一个字典，有参数 `"{\n\"pizza_name\": \"Salami\"\n}"`，所以这是函数 （Function Calling) 的参数，这是我们要传递给我们的函数 （Function Calling) 的值。

我们要传递给它的函数 （Function Calling) 的名称是 `get_pizza_info` 函数。

所以我们现在要使用这个信息，我们可以首先检查是否实际上我们有这个函数调用 （Function Calling)对象在这里。

如果是的话，我们知道 llm 要求我们调用函数 （Function Calling) ，并从这里提取函数 （Function Calling) 名称。

```
if message.get("function_call"):
    # 解析第一次调用的时候返回的 pizza 信息
    function_name = message["function_call"]["name"]
    pizza_name = json.loads(message["function_call"] ["arguments"]).get("pizza_name")
    print(pizza_name)
    # 这里将 chat 小助手函数的响应结果提取后，传递 function_response
    function_response = get_pizza_info(
        pizza_name=pizza_name 
    )

    second_response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        messages=[
            {"role": "user", "content": query},
            message,
            {
                "role": "function",
                "name": function_name,
                "content": function_response, # function calling 的 content 是 get_pizza_info 函数 
            },
        ],
    )

second_response
```

就像这样我们有这个函数调用 （Function Calling)对象，然后我们想要检索名称，所以这将是我们要调用的函数 （Function Calling) 名称，我们还希望得到 pizza 名称腊肠，这将是我们要传递给我们的函数 （Function Calling) 的参数。


我们得到函数调用 （Function Calling)和参数，然后我们有这个对象，我们使用 `JSON.loads()` 将其从 JSON 对象转换为字典，然后我们提取 `pizza_name` 键，这是我们要传递给我们的函数 （Function Calling) 的名称，然后我们可以通常进行 API  Call并检索信息，然后这
是函数 （Function Calling) 的响应，这也是一个 JSON 对象。


我们从这个函数 （Function Calling) 中检索到它，**通过这个结果**，我们想再次调用 API，我们在这里使用最新的模型，并且在之前我们还传递了消息，我们还传递了一个对象，其中有一个名为 `message["function_call"]["name"]` ，并在这里传递了函数 （Function Calling) 名称.

第一个函数 （Function Calling) 响应是具有名称和价格的披萨信息，我们在这里传递它给第二个函数  `second_response`，然后我们进行第二个响应 。

如果我们打印这个:

```
<OpenAIObject chat.completion id=chatcmpl-7Y9045lCV15L1psS5SNYclk4SGcDU at 0x166c574fa10> JSON: {
  "id": "chatcmpl-7Y9045lCV15L1psS5SNYclk4SGcDU",
  "object": "chat.completion",
  "created": 1688372104,
  "model": "gpt-3.5-turbo-0613",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "The cost of a pizza salami is $10.99."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 58,
    "completion_tokens": 13,
    "total_tokens": 71
  }
}
```

我们得到一个复杂的对象，我们可以在这里找到内容，腊肠的成本是 10.99，所以函数 （Function Calling) 响应将用于从我们的 API 中使用信息（也就是第一个函数 （Function Calling) ）创建了类似于人的答案。

这就是函数调用 （Function Calling)的全部内容，让 llm 决定是否要使用其他信息或外部信息。

只需让 LLM 自己回答问题，这是最基本的用法。

## 如何与 LangChain 一起使用?

我们来看看如何与 LangChain 一起使用。

首先导入 ChatOpenAI 类和 HumanMessage、AIMessage，还有 ChatMessage 类，这些类可以帮助我们创建这种功能，包括用户角色等。
我们可以不必要像之前那样，定义角色等，只需要传递 `content`。其他的都交给了 Langchain.

```python
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage, ChatMessage
```

我们只需要提供内容  `content`，可以看到我们有这个 `HumanMessage`，`AIMessage`。

然后在这里创建模型 LLM，通过实例化 ChatOpenAI，我们还在这里传入最新的模型 `gpt-3.5-turbo-0613`，然后运行预测消息函数 `predict_messages`，在这里可以提供一个额外的关键字参数，称为 `functions`，我们将提供我们之前定义的函数列表 `functions=[...]`。

```python
llm = ChatOpenAI(model="gpt-3.5-turbo-0613")
message = llm.predict_messages(
    [HumanMessage(content="What is the capital of france?")], functions=functions
)
```
运行后，我们可以看到:

```
AIMessage(content='The capital of France is Paris.', additional_kwargs={}, example=False)
```
 `AIMessage` 的 `content`,  法国的首都是巴黎。

使用 Langchain 后，过程变得很标准化。

现在我们再次运行查询披萨莎拉米在餐厅里的价格。
```python
llm = ChatOpenAI(model="gpt-3.5-turbo-0613")
message_pizza = llm.predict_messages(
    [HumanMessage(content="How much does pizza salami cost?")], functions=functions
)
message
```

```
AIMessage(content='', additional_kwargs={'function_call': {'name': 'get_pizza_info', 'arguments': '{\n"pizza_name": "Salami"\n}'}}, example=False)
```

现在我们预期我们不会得到内容 `content=''`，也不会得到任何东西，但是我们得到了 `message.additional_kwargs`。

`message.additional_kwargs` 中包括了函数调用 （Function Calling)字典，名称是披萨信息，因此我们可以看到 LLM 建议我们首先调用披萨信息函数 （Function Calling) ，这里的参数是 `Pizza_name` 和 `salami`，这看起来与我们之前做的非常相似，但是没有使用 LangChain。

现在如果我们看一下第一个响应中的 `message.additional_kwargs`，它们是空的，这只是一个空字典。

如果我们得到一个函数调用 （Function Calling)建议，我们可以在这里得到它， `additional_kwargs` 中有我们的函数调用 （Function Calling)，名称和参数。

因此我们现在可以使用这种功能来定义我们是否要进行额外的调用。

要获得披萨名称，我们与之前一样，只需提取额外的参数 `message.additional_kwargs`，并将其转换为具有`json.loads`的字典，并通过调用字典的 `get` 方法获得披萨名称，然后我们得到返回的披萨名称，即莎拉米.

然后我们再次调用这个函数 （Function Calling) ，就像这样再次调用函数 （Function Calling) ，我们得到名称莎拉米和价格为 10.99。

```python
import json

# 打印结果是 'Salami'
pizza_name = json.loads(message_pizza.additional_kwargs["function_call"]["arguments"]).get("pizza_name")
```

```
# 将'Salami'传参给 get_pizza_info 函数
pizza_api_response = get_pizza_info(pizza_name=pizza_name)
```

返回的结果是：
```
'{"name": "Salami", "price": "10.99"}'
```

现在我们可以使用这个 API 响应并创建我们的新 API Call，使用预测消息函数 `llm.predict_messages`，在这里我们只有 HumanMessage，我们将提供我们的查询，而 AIMessage 我们将只提供额外的关键字参数 `additional_kwargs` 的字符串，而不是提供这个函数 ，我们将只使用 `ChatMessage`，在这里提供角色是 `role="function`。

`additional_kwargs` 我们将提供名称 `name`，这个是上一次调用的 API 返回结果 `message_pizza.additional_kwargs["function_call"]["name"]`。

因此，如果我们再次运行这个，我们应该得到与之前类似的响应，`AIMessage` 内容是在餐厅里的披萨价格为 `10.99`，因此非常容易使用.

```
second_response = llm.predict_messages(
    [
        HumanMessage(content=query), # query = "How much does pizza salami cost?"
        AIMessage(content=str(message_pizza.additional_kwargs)),
        ChatMessage(
            role="function",
            additional_kwargs={
                "name": message_pizza.additional_kwargs["function_call"]["name"]
            },
            # pizza_api_response = get_pizza_info(pizza_name=pizza_name)
            content=pizza_api_response
        ),
    ],
    functions=functions,
)
# second_response
```

运行结果：
```
<OpenAIObject at 0x166d47a43b0> JSON: {
  "role": "assistant",
  "content": null,
  "function_call": {
    "name": "get_pizza_info",
    "arguments": "{\n\"pizza_name\": \"Salami\"\n}"
  }
}
Salami
<OpenAIObject chat.completion id=chatcmpl-7Y9045lCV15L1psS5SNYclk4SGcDU at 0x166c574fa10> JSON: {
  "id": "chatcmpl-7Y9045lCV15L1psS5SNYclk4SGcDU",
  "object": "chat.completion",
  "created": 1688372104,
  "model": "gpt-3.5-turbo-0613",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "The cost of a pizza salami is $10.99."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 58,
    "completion_tokens": 13,
    "total_tokens": 71
  }
}
AIMessage(content='The capital of France is Paris.', additional_kwargs={}, example=False)
AIMessage(content='', additional_kwargs={'function_call': {'name': 'get_pizza_info', 'arguments': '{\n"pizza_name": "Salami"\n}'}}, example=False)
'Salami'
'{"name": "Salami", "price": "10.99"}'
AIMessage(content='The pizza Salami costs $10.99.', additional_kwargs={}, example=False)
```

### 使用 LangChain 的 tools

但是使用这种 `message.additional_kwargs` 工作仍然感觉有点复杂。

LangChain 已经提供了与外部世界交互的另一种标准化方法，以进行请求或其他操作，这些称为工具 tools，工具 tools 是由 Chain 提供的类，您也可以创建自己的工具，我将向您展示如何做到这一点。

```
from typing import Optional
from langchain.tools import BaseTool
from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)


class StupidJokeTool(BaseTool):
    name = "StupidJokeTool"
    description = "Tool to explain jokes about chickens"

    def _run(
        self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        return "It is funny, because AI..."

    async def _arun(
        self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("joke tool does not support async")
```



首先，我们必须导入基类，您可以通过继承基类工具来创建自定义类或自定义工具，然后必须提供工具的名称和描述，这是一个非常简单和非功能性的工具，只是为了向您提供一些语法，您必须定义一个下划线 `_run` 函数和一个下划线 `_arun` 函数，这提供了异步支持和同步支持。

这里函数 （Function Calling) 返回的东西是没有实际功能。

当然您可以在这里查询数据库或者做任何您想做的事情，但是我只是向您展示您可以做什么。

如果您有了自己的工具与类一起使用，您可以轻松将自己的类转换为格式化的工具: `format_tool_to_openai_function`，我还在这里导入了 `MoveFileTool` 工具，它允许您在计算机上移动文件。

```python
from langchain.tools import format_tool_to_openai_function, MoveFileTool


tools = [StupidJokeTool(), MoveFileTool()]
# 将自己的 tools 转换为格式化的 function
functions = [format_tool_to_openai_function(t) for t in tools]
# functions 是之前定义的一个变量：一个函数列表

```

```python
query = "Why does the chicken cross the road? To get to the other side"
output = llm.predict_messages([HumanMessage(content=query)], functions=functions)
# output
```
我们看看 output 运行结果：

```
AIMessage(content='', additional_kwargs={'function_call': {'name': 'StupidJokeTool', 'arguments': '{\n"__arg1": "To get to the other side"\n}'}}, example=False)
```

我们现在将 output 的结果传递进 second_response （注意 `additional_kwargs`)。

我在这里创建了一个工具列表，其中实例化了我的自定义类和 MoveFile 工具类，因此我有两个工具。


如果我们看一下它们，它们有一个名称 `name`，如愚蠢的笑话工具 `StupidJokeTool`，它们有一个描述，这是直接从这些类参数 `arguments` 中获取的，`arguments`里的属性是 `__arg1`。


现在我们可以再次像这样使用它，例如为什么鸡过马路？

```python
query = "Why does the chicken cross the road? To get to the other side"

second_response = llm.predict_messages(
    [
        HumanMessage(content=query),
        AIMessage(content=str(output.additional_kwargs)),
        ChatMessage(
            role="function",
            additional_kwargs={
                "name": output.additional_kwargs["function_call"]["name"]
            },
            content="""
                {tool_response}
            """,
        ),
    ],
    functions=functions,
)
# second_response
```


现在我可以运行预测消息函数，并在这里提供这两个函数 （Function Calling) ，并让 LLM 决定是否使用工具，因此，正如我们所看到的，我们不会得到任何内容。

因此，AI 希望我们调用一个函数，函数调用 （Function Calling)是我们的 `StupidJokeTool`工具类或我们将类转换为的函数 （Function Calling) 。

因此现在我们实际上没有函数 （原生openai案例，我们是定义了函数的描述) ，我们只是将工具本身转换为函数 （Function Calling) 定义，但不是函数 （Function Calling) 本身。

运行后返回的：
```
AIMessage(content='', additional_kwargs={'function_call': {'name': 'StupidJokeTool', 'arguments': '{\n  "__arg1": "To get to the other side"\n}'}}, example=False)
```

我们必须提取 `additional_kwargs`，然后我们将它传递给我们的函数 （Function Calling) ，然后我们得到我们的工具响应。

现在我们可以再次使用这两个响应来进行其他请求，仍然是与之前相同的模式，我们通过传入一个聊天消息，这里的角色是函数 （`role="function"`) ，内容是完全自动的响应，因此这可能甚至不会像预期的那样工作。

它不像预期的那样工作，因为我们的工具 `StupidJokeTool` 没有提供任何功能，但是我认为您会看到模式，进行初始调用，如果 LLM 要求您调用函数 （Function Calling) ，则调用函数 （Function Calling) 并提供函数 （Function Calling) 的输出到另一个 LLM 调用。

这就是目前与普通 LLM 链的工作方式。

老实说，对于代理，这个功能已经实现得更好。

### Langchain Agent 如何实现 Function Calling ？

我将向您展示它是如何工作的，首先我们导入一些链 Chain，例如 `LLMMathChain`，还有一个 `chat_models`，聊天模型在这里使用 `ChatOpenAI` 创建我们的 LLM。

```
from langchain import LLMMathChain
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI


llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")
llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)
tools = [
    Tool(
        name="Calculator",
        func=llm_math_chain.run,
        description="useful for when you need to answer questions about math"
    ),
]
```

此代理能够回答普通问题并进行一些计算，因此例如我们像这样使用它，我们使用初始化代理函数 （Function Calling) ，现在我们在这里使用它与我们的工具一起。

```
agent = initialize_agent(tools, llm, agent=AgentType.OPENAI_FUNCTIONS, verbose=True)
```

现在我们有了新的代理类型 `OPENAI_FUNCTIONS`，它属于 openai 函数类型，所以如果运行这个代理，*我们不需要传递任何关键字参数或额外的参数，只需要像这样使用它。*

现在我们运行“法国的首都是什么”，我们得到的结果是法国的首都：巴黎。

```
agent.run("What is the capital of france?")
```

我们可以得到：
```
> Entering new  chain...
The capital of France is Paris.

> Finished chain.
'The capital of France is Paris.'
```

如果我们想知道 100 除以 25 等于多少，这时候计算器被调用，我们得到最终答案 100 除以 25 等于 4。

```
agent.run("100 除以 25 等于多少?")
```

我们可以得到：
```
> Entering new  chain...

Invoking: `Calculator` with `100 / 25`




> Entering new  chain...
100 / 25```text
100 / 25
```
...numexpr.evaluate("100 / 25")...

Answer: 4.0
> Finished chain.
Answer: 4.0100 除以 25 等于 4。
```


所以对于代理 Langchain Agent 来说，它的工作非常流畅，我相信很快它也会与其他的 llm 链一起工作。

如果你喜欢这个文章，请随时订阅我并给这个文章点赞，非常感谢，再见。

## 参考资料

- Github 代码仓库：https://github.com/Coding-Crashkurse/Langchain-Full-Course

- langchain 中文教程： https://python.langchain.com.cn

