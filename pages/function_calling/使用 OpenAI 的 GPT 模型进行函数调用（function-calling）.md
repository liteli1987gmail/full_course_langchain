
# 使用 OpenAI 的 GPT 模型进行函数调用（function-calling）

## OpenAI 发布函数调用（function-calling）

最近，OpenAI 发布了一项令人振奋的更新，他们在 API 中引入了函数调用（function-calling）的功能，这非常酷，因为到目前为止，我们一直在猜测，我也在猜测他们实际上是如何选择插件的工具的。

现在，他们开始逐层剥去洋葱的皮，让我们知道如何做到这一点。

因此，我在这个视频教程中要做的是，浏览他们的公告博客文章，向你展示一些我感到非常兴奋的关键点，然后我想通过这个教程，展示如何在你自己的应用中使用函数调用（function-calling）的简单、中级和更高级的选项。

我们看到的一个有趣的趋势是，语言模型正在增强它们的能力，不仅仅是普通的文本生成。

### 没有函数调用之前存在什么问题？

当 Chat GPT 首次发布时，关于你的猫的诗歌确实很有趣，但有许多系统也需要做出 **决策**。

当你使用语言模型作为推理引擎时，自由形式的文本并不是与其他计算机通话的最佳方式，如果你能用 **JSON 格式** 进行通话，那就更好了，这就是函数调用（function-calling）开始引导我们的方向。

### 使用限制

OpenAI 最近公布了一些重要的 [更新](https://openai.com/blog/function-calling-and-other-api-updates)，包括更可控的 API 模型、函数调用（function-calling）能力、更长的上下文和更低的价格。

新的函数调用（function-calling）能力允许开发者向 `gpt-4-0613` 和 `gpt-3.5-turbo-0613` 描述函数，并让模型智能地选择输出包含调用这些函数的参数的 JSON 对象。

这是一种更可靠地将 GPT 的能力与外部工具和 API 连接的新方法。这些模型已经进行了微调，既可以侦测到需要调用函数的时刻（取决于用户的输入），也可以响应符合函数签名的 [JSON](https://www.freecodecamp.org/news/how-to-use-rest-api/)。

OpenAI 提供了函数调用（function-calling）的 [示例](https://openai.com/blog/function-calling-and-other-api-updates)，步骤包括：使用函数和用户输入调用模型，使用模型的响应调用你的 API，将响应发送回模型进行总结。

OpenAI 提醒开发者，他们正在努力减少从工具的输出中获取不可信数据的风险，并建议开发者只从可信的工具中获取信息，并在执行有实际影响的操作（如发送电子邮件、在线发布或购买）之前，包含用户确认步骤。


我想通过这两个例子来展示我们将如何在 Langchain 中引入这个功能。

## OpenAI 的基础示例


现在我们先来做 OpenAI 的基础示例，然后我们将进行 Langchain 示例，然后是一个复杂的示例。

所以我们要在这里导入我们的包，确保我们的 API 密钥都设置好了。然后我们首先要做的就是 OpenAI 的基础示例。
```python
# !pip install langchain --upgrade
# Version: 0.0.199 Make sure you're on the latest version

import langchain
import openai
import json

# Environment Variables
import os
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv('OPENAI_API_KEY', 'YourAPIKeyIfNotSet')
```

所以我要做的就是刚才我们做过的那个，但是我要定义一个名为获取当前天气的函数。所以我们要给它一个名字，叫做获取当前天气，我们要给它一个描述，这其实就是模型会知道该如何选择哪个工具的指令，获取给定位置的当前天气。
```python
function_descriptions = [
            {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "unit": {
                            "type": "string",
                            "description": "The temperature unit to use. Infer this from the users location.",
                            "enum": ["celsius", "fahrenheit"]
                        },
                    },
                    "required": ["location", "unit"],
                },
            }
        ]
```        

好，然后我们要给它一些参数，然后我们要给它一些属性。所以在这里，我们要给它一个位置和一个单位，并且你要定义类型、描述，甚至在这个案例中，你可以找到枚举值，如果你只希望指定数量的值来返回。然后你要返回的参数，这些参数不应该返回。好，在这个案例中，我们希望位置和单位返回。让我们继续运行这个。

所以对于用户查询，波士顿的天气怎么样？
```python
user_query = "波士顿的天气怎么样?"
```


让我们混合一下，因为这只是之前的内容，旧金山的天气怎么样？所以请再次记住，这是基础的 OpenAI，我们还没有使用 Lang chain。我们要调用 openai chat completion create，我们要使用 gpt4，记住，这是我们需要使用的 `gpt-4-0613` 的模型，我们要给它一系列消息。所以在这个案例中，我们只给它一个消息，这是一个用户消息，内容就是我们在上面选择的用户查询。
```python
response = openai.ChatCompletion.create(
        model="gpt-4-0613",
        
        # This is the chat message from the user
        messages=[{"role": "user", "content": user_query}],
    
        
        functions=function_descriptions,
        function_call="auto",
    )
```


我们将传入函数参数，这是我们上面的函数描述，我们在这里只选择了一个工具，但是这是一个列表，你可以选择多个工具，我们马上就会讲到这个。

然后对于函数调用（function-calling），我们要做的是自动 `function_call="auto"`。现在这个函数调用（function-calling），它指定了机器人是否应该使用函数返回，或者是否应该首先选择。

当你有自动选择时，它会为你自动选择。

如果你把这个设置为 None，那么你就不希望使用函数，那么函数就不会被使用。

我们来调用一下，看看接口返回内容：
```python
ai_response_message = response["choices"][0]["message"]
print(ai_response_message)
```

打印结果：
```text
{
  "content": null,
  "function_call": {
    "arguments": "{\n  \"location\": \"San Francisco, CA\",\n  \"unit\": \"fahrenheit\"\n}",
    "name": "get_current_weather"
  },
  "role": "assistant"
}
```

所以我们得到的是，我们有我们的 AI 回应消息。现在的内容是空的 `"content": null,`，因为没有消息本身，这不是实际的内容消息，但我们确实得到了一个函数调用（function-calling）的回应 `"function_call"`。

在这里，我们在参数中有位置，我们解析出了旧金山，然后我们解析出了华氏度作为单位，名字就是获取当前天气的函数。

所以让我们继续处理这个，我们将得到用户的位置 `user_location` 和用户的单位 `user_unit`。

```python
user_location = eval(ai_response_message['function_call']['arguments']).get("location")
user_unit = eval(ai_response_message['function_call']['arguments']).get("unit")
```

我们在这里定义了一个简单的函数，这只是一个静态函数，它将返回一些数据给我们，它将给我们这个位置和这个单位的天气。

```python
def get_current_weather(location, unit):
    
    """Get the current weather in a given location"""
    
    weather_info = {
        "location": location,
        "temperature": "72",
        "unit": unit,
        "forecast": ["sunny", "windy"],
    }
    return json.dumps(weather_info)
```

```
function_response = get_current_weather(
    location=user_location,
    unit=user_unit,
)
```
让我们运行这个，然后对于我们的函数回应，让我们得到这个。返回的结果是：
```
'{"location": "San Francisco, CA", "temperature": "72", "unit": "fahrenheit", "forecast": ["sunny", "windy"]}'
```


好，所以我们在这里有的是，这是我们从 API 调用得到的消息，这通常会是一个已经更新的更改的确认，或者可能是一个成功的消息或者类似的东西，但无论如何，这是 API 的回应。

然后我们要做的是，我实际上要替换那个，我们要得到第二个回应。所以我们实际上将我们的回应发送回给 OpenAI。所以我们要使用 gpt4，对于这个用户，这将是我们之前有的消息历史。

所以我们有一个用户消息，内容就是用户查询，也就是旧金山的天气怎么样？然后我们要传递我们的 AI 回应消息 `ai_response_message`，这代表了它最初给我们返回的东西。

然后我们要给它 API 的回应，所以我们说，嘿，你告诉我们去调用这个函数，我们确实调用了这个函数，因为我们调用了这个函数 `get_current_weather` 获取当前天气，这就是我们从函数中得到的回应。

```python
second_response = openai.ChatCompletion.create(
    model="gpt-4-0613",
    messages=[
        {"role": "user", "content": user_query},
        ai_response_message,
        {
            "role": "function",
            "name": "get_current_weather",
            "content": function_response,
        },
    ],
)
```

它将是所有这些非常好的信息，它会注意到我们得到了天气和预报。然后让我们打印出下一个消息。

```
print (second_response['choices'][0]['message']['content'])
```

```
Currently in San Francisco, CA the weather is sunny and windy with a temperature of 72°F.
```

旧金山的天气目前是晴朗且多风，温度为 72 华氏度。现在这很酷，因为它得到了我们的 API 回应，它知道它在处理什么格式，然后它给了我们自然语言的回应。好，这就是在 OpenAI 中如何做到这一点的基础示例。

## Langchain 示例

让我们去检查一下在 Lang chain 中如何做到这一点。

所以在这里需要记住的关键部分是，对于 OpenAI 引入的任何新技术，我们都需要花一点时间来理解我们如何最好地与之合作，对于 Langchain 也是如此。

我已经做了一些变通办法，但也利用了他们到目前为止做的一些工作，他们的团队非常棒，他们实际上在新框架发布一个小时之内就推出了更新，并已经为此提供了支持。

因此，我想象随着我们的继续，将会有越来越多的支持。好，所以我们要导入 chat openAI，因为我们要使用 gpt4，我们要导入一个人类消息，这将代表人类在说什么，AI 消息。

```python
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage, ChatMessage
from langchain.tools import format_tool_to_openai_function, YouTubeSearchTool, MoveFileTool
```

为了模拟我们上面看到的函数消息，我只是做了一个通用的聊天消息，然后我们会给它一个自定义的角色，我们将导入一些可能有趣的工具，让我们继续并更新这些 gpt4。

```
llm = ChatOpenAI(model="gpt-4-0613")
```

所以我们要使用移动工具，这个移动工具就是告诉你如何移动一些文件，这其实是 Harrison 之前展示的例子，然后我们将把每一个工具格式化为 OpenAI 想要看到的函数模式或函数调用（function-calling）模式。

```
tools = [MoveFileTool()]
functions = [format_tool_to_openai_function(t) for t in tools]
```


让我们做这个，然后我们看看我们现在有什么函数。
```
print(f"{functions}")
```

```
[{'name': 'move_file',
  'description': 'Move or rename a file from one location to another',
  'parameters': {'title': 'FileMoveInput',
   'description': 'Input for MoveFileTool.',
   'type': 'object',
   'properties': {'source_path': {'title': 'Source Path',
     'description': 'Path of the file to move',
     'type': 'string'},
    'destination_path': {'title': 'Destination Path',
     'description': 'New path for the moved file',
     'type': 'string'}},
   'required': ['source_path', 'destination_path']}}]
```   



所以你可以看到我们有我们的名字，我们有我们的描述，我们还有我们的参数，这将是它需要知道如何使用这些工具的不同信息。

然后我们要做的是，我们要传一个人类消息，我们要说，嘿，请把文件 Foo 移动到 bar。

```
message = llm.predict_messages([HumanMessage(content='move file foo to bar')], functions=functions)

```
```
message.additional_kwargs['function_call']
```
```
{'name': 'move_file',
 'arguments': '{\n  "source_path": "foo",\n  "destination_path": "bar"\n}'}
```


但重要的部分是，我们要在这里传入函数，这个函数将是它可以使用的工具列表。所以让我们运行这个，然后我们看看我们这里有什么。所以在额外的关键字参数 `additional_kwargs['function_call']` 中，我们有我们的函数调用（function-calling），这将是我们从 OpenAI 得到的函数调用（function-calling），这是用 Lang chain 的方式来做的，它说我们需要使用移动文件的工具，这里的参数源路径是 Foo，目标路径是 bar，这是我们预期的。

## 真实世界的复杂例子

现在我想做一个更复杂的例子，所以我想做多个工具，但我也想在一个用户查询中做多个请求。

我将创建一个新的函数描述 `function_descriptions`，用于更新财务模型。

它将接受三个参数，分别是要更新的年份 `year`，要更新的类别 `category`，以及要更新的金额 `amount`。

```python
function_descriptions = [
            {
                "name": "edit_financial_forecast",
                "description": "Make an edit to a users financial forecast model",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "year": {
                            "type": "integer",
                            "description": "The year the user would like to make an edit to their forecast for",
                        },
                        "category": {
                            "type": "string",
                            "description": "The category of the edit a user would like to edit"
                        },
                        "amount": {
                            "type": "integer",
                            "description": "The amount of units the user would like to change"
                        },
                    },
                    "required": ["year", "category", "amount"],
                },
            },
            {
                "name": "print_financial_forecast",
                "description": "Send the financial forecast to the printer",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "printer_name": {
                            "type": "string",
                            "description": "the name of the printer that the forecast should be sent to",
                            "enum": ["home_printer", "office_printer"]
                        }
                    },
                    "required": ["printer_name"],
                },
            }
        ]
```        
OpenAI 新的函数调用（function-calling）的一个酷炫之处在于，LLM 会决定它应该向用户返回一个正常的响应，还是再次调用函数。让我们通过用户在同一查询中的两个不同请求来测试这一点。

```
user_request = """
Please do three things add 40 units to 2023 headcount
and subtract 23 units from 2022 opex
then print out the forecast at my home
"""
```

我们将自己跟踪消息历史。随着对函数对话的更多支持的加入，我们将不再需要这样做。

首先，我们将将用户的消息 `HumanMessage` 与我们的函数 `function_descriptions` 调用一起发送给语言模型。

### 第一个请求

```
first_response = llm.predict_messages(
    [HumanMessage(content=user_request)],
    functions=function_descriptions)

```
打印 `first_response` 返回的结果是：
```
AIMessage(content='', additional_kwargs={'function_call': {'name': 'edit_financial_forecast', 'arguments': '{\n  "year": 2023,\n  "category": "headcount",\n  "amount": 40\n}'}}, example=False)
```

打印 `first_response.additional_kwargs`：

```python
{'function_call': {
   'name': 'edit_financial_forecast',
  'arguments': '{\n  
  "year": 2023,\n  
  "category": "headcount",\n  
  "amount": 40\n}'}}
```
我们最开始是在 prompt 中输入了指令。此时已经把指令中的更新的年份 `year`，要更新的类别 `category`，以及要更新的金额 `amount`，全部提取作为 `arguments` 参数。

```
function_name = first_response.additional_kwargs["function_call"]["name"]
# 打印 function_name
```
打印 function_name 的结果是：
```
'edit_financial_forecast'
```
再打印返回, 第一次调用的函数 `first_response.additional_kwargs`, 给我们的参数是什么：
```
print (f"""
Year: {eval(first_response.additional_kwargs['function_call']['arguments']).get('year')}
Category: {eval(first_response.additional_kwargs['function_call']['arguments']).get('category')}
Amount: {eval(first_response.additional_kwargs['function_call']['arguments']).get('amount')}
""")
```
打印的结果是：
```
Year: 2023
Category: headcount
Amount: 40
```

### 第二个请求

但我们还没有完成！用户查询中有第二个请求，让我们将第一次函数调用（function-calling）的获取的信息，传回模型中，看看会发生什么？

`AIMessage(content=str(first_response.additional_kwargs)),` 这里是灵魂代码。

```
second_response = llm.predict_messages(
    [HumanMessage(content=user_request),
    AIMessage(content=str(first_response.additional_kwargs)),
        ChatMessage(role='function',
        additional_kwargs = {'name': function_name},
        content = "Just updated the financial forecast for year 2023, category headcount amd amount 40"
                                                   )
                                       ],
       functions=function_descriptions)
```
我们来看看返回，此时的三个参数结果变了。
`{\n  "year": 2022,\n  "category": "opex",\n  "amount": -23\n}`

```
{'function_call': {'name': 'edit_financial_forecast',
  'arguments': '{\n  "year": 2022,\n  "category": "opex",\n  "amount": -23\n}'}}
```

我们再来溯源一次，看看哪个函数被调用的。

```
function_name = second_response.additional_kwargs['function_call']['name']
function_name
```

打印结果是:
```
'edit_financial_forecast'
```
### 第三个请求

我们再来挑战更复杂的情况。将前面两次的函数返回结果，传给 LLM 模型，再次调用。

```
third_response = llm.predict_messages(
    [HumanMessage(content=user_request),
    AIMessage(content=str(first_response.additional_kwargs)),
    AIMessage(content=str(second_response.additional_kwargs)),
    ChatMessage(role='function',
    additional_kwargs = {'name': function_name},
    content = """
        Just made the following updates: 2022, opex -23 and
        Year: 2023
        Category: headcount
        Amount: 40
        """)
        ],
    functions=function_descriptions)
```

我们看看这次会有什么意外收获？

```python
# 打印 third_response.additional_kwargs
{'function_call': {'name': 'print_financial_forecast',
  'arguments': '{\n  "printer_name": "home_printer"\n}'}}
```

```
function_name = third_response.additional_kwargs['function_call']['name']
```
```
# 打印 function_name 的结果
'print_financial_forecast'
```

太好了！所以它知道它已经完成了财务预测（因为我们告诉了它），然后将我们的预测发送到了我们的家用打印机。让我们结束这个会话吧。

```
forth_response = llm.predict_messages(
    [HumanMessage(content=user_request),
    AIMessage(content=str(first_response.additional_kwargs)),
    AIMessage(content=str(second_response.additional_kwargs)),
    AIMessage(content=str(third_response.additional_kwargs)),
    ChatMessage(role='function',
    additional_kwargs = {'name': function_name},
    content = """
        just printed the document at home
        """
        )],
    functions=function_descriptions)
```

最后看看它给我们返回了什么：

```
'I have updated the financial forecast as per your instructions. It has also been printed at your home.'
```

## 结论

我们刚刚串联了一大堆不同的命令，它展示了应该使用哪个工具，不仅仅是选择了使用哪个工具，而且还选择了何时不需要使用工具，而需要给我们一个自然语言的回应。

我非常期待看到你们如何使用这个聊天模型，我认为这里将会有一些非常酷的功能。

这没有什么革命性的，但它卸载了一个我们之前没有的，非常复杂和混乱的预测过程的部分。


