# 7.5 本章小结

本章从学术的角度探索了两个当前备受关注的库——AutoGPT 和 BabyAGI，这两个库代表着人工智能的新趋势，而 LangChain 的基本组件和高级技巧则是理解和应用这两个库的关键。

我们首先分析了 AutoGPT 和 BabyAGI 的工作原理，从中我们可以看到，它们利用 LangChain 的基本组件，构建了复杂的模型。但是，读者需注意，尽管 AutoGPT 和 BabyAGI 引人注目，它们目前仍处于实验阶段，暂时不能用于生产环境。因此，深入了解和应用它们需要足够的谨慎和准备。

除了 AutoGPT 和 BabyAGI，我们还研究了一个简单实用的对话式表单，以及流行的产品应用 Chat GPT。通过实例代码，我们不仅可以了解这些应用如何利用 LangChain 的基本组件进行工作，也可以学习到一些高级的编程技巧。

然而，在开始学习本章内容之前，我们强烈建议读者先全面了解 LangChain 的所有组件。因为本章的所有应用和实例都是基于这些组件的深度组合和运用。

# 7.1 使用 Lang Chain 实现 BabyAGI

现在我们将利用 Lang Chain 进行 BabyAGI 实现。这将让我们更加直观地看到每一步骤的发生情况，并且，您也可以在自己的环境中进行实验。

![](https://lh5.googleusercontent.com/ka3XyjN55jUh_7H4HJSrVJt9ctOdaaXRBvHFgC2fgCq05Mp0C8WmVowQmF0IKp0wd7ewn7nfcSwa6E8mW0nV60_eb01ioCEGusW9ql4oX25tDmS-TCOnlUNlVDkfjMrPBd-YQqjjcSdrx062uUljvtc)

### 7.1.1 环境与工具

对于此次实验，我们会需要两个主要工具：OpenAI 以及一个搜索引擎 API。这两者将会协同完成 BabyAGI 的构建。有两个版本的 BabyAGI，一个不依赖任何外部调用，而另一个则利用搜索引擎进行外部调用。

安装库：
```
!pip -q install langchain huggingface_hub openai google-search-results tiktoken cohere faiss-cpu
```
设置密钥：
```
import os

os.environ["OPENAI_API_KEY"] = ""
os.environ["SERPAPI_API_KEY"] = ""
```
导入工具：
```
import os
from collections import deque
from typing import Dict, List, Optional, Any

from langchain import LLMChain, OpenAI, PromptTemplate
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import BaseLLM
from langchain.vectorstores.base import VectorStore
from pydantic import BaseModel, Field
from langchain.chains.base import Chain
```
### 7.1.2  向量存储

在此实验中，我们使用了 FAISS 向量存储，这是一种内存存储技术，使我们无需进行任何外部调用，例如向 Pinecone 请求。但如果你愿意，你完全可以改变一些设定，将其连接到 Pinecone。向量存储是利用 OpenAI 的嵌入进行的。

导入 FAISS 向量库：
```
from langchain.vectorstores import FAISS
from langchain.docstore import InMemoryDocstore
```


在构建一个特定的嵌入模型，生成向量索引，并存储这些向量时，我们可以按照以下步骤来操作。

首先，我们需要选择一个适当的嵌入模型。这种模型可以是词嵌入模型，如 Word2Vec 或 GloVe，也可以是句子嵌入模型，如 BERT 或者 Doc2Vec。这些模型通过将词或句子映射到高维度的向量空间，实现了对词或句子语义的捕捉。选择哪种嵌入模型主要取决于我们处理的任务特性和数据的特点。

这里我们使用的是 OpenAI 的嵌入模型, OpenAI 的文本嵌入模型可以精确地嵌入大段文本，具体而言，8100 个标记，根据它们的词对标记比例 0.75，大约可以处理 6143 个单词。它输出 1536 维的向量。

```
# Define your embedding model
embeddings_model = OpenAIEmbeddings()
```

其次，对我们的文本数据进行处理，生成相应的嵌入向量。生成向量后，我们需要构建一个索引，以便能够高效地查询和比较向量。

```
# Initialize the vectorstore as empty
import faiss
embedding_size = 1536
index = faiss.IndexFlatL2(embedding_size)
```


最后，我们需要将生成的向量和构建的索引进行存储。

```
vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})
```


### 7.1.3 构建链

LangChain 的好处在于，我们可以清楚看到链条在执行哪些操作，以及它们的提示是什么。在这里，我们有三个主要链条：任务创建、任务优先级和执行。这些链条都在为达成整体目标而工作，它们会生成一系列任务。

#### 任务创建链

这个链条基本上是说：作为一个任务创建的 AI，你要利用执行代理的结果，来创建具有一定目标的新任务。此处的目标就是你想要 AI 实现的东西。最后完成的任务有结果，这个结果是基于任务描述生成的。这些是未完成的任务，如果有一系列尚未完成的任务，就会将其输入到这个链条中。最后，根据结果创建新任务，这些任务由 AI 系统完成，并且不与未完成的任务重叠。

```
class TaskCreationChain(LLMChain):
    """Chain to generates tasks."""

    @classmethod
    def from_llm(cls, llm: BaseLLM, verbose: bool = True) -> LLMChain:
        """Get the response parser."""
        task_creation_template = (
            "You are an task creation AI that uses the result of an execution agent"
            " to create new tasks with the following objective: {objective},"
            " The last completed task has the result: {result}."
            " This result was based on this task description: {task_description}."
            " These are incomplete tasks: {incomplete_tasks}."
            " Based on the result, create new tasks to be completed"
            " by the AI system that do not overlap with incomplete tasks."
            " Return the tasks as an array."
        )
        prompt = PromptTemplate(
            template=task_creation_template,
            input_variables=["result", "task_description", "incomplete_tasks", "objective"],
        )
        return cls(prompt=prompt, llm=llm, verbose=verbose)
```        

这些步骤看起来很简单，但这里就是你可以进行修改，从而使 AI 更符合你需求的地方。在接下来的博客中，我们将进一步深入解释向量存储和 LangChain 的运作细节。

#### 任务优先级链

这个链条的主要职责是将传入的任务进行清理，重新设置它们的优先级，以便于按照您的团队的最终目标进行排序。任务优先级链条不会删除任何任务，而是将任务以编号列表的形式返回。

```
class TaskPrioritizationChain(LLMChain):
    """Chain to prioritize tasks."""

    @classmethod
    def from_llm(cls, llm: BaseLLM, verbose: bool = True) -> LLMChain:
        """Get the response parser."""
        task_prioritization_template = (
            "You are an task prioritization AI tasked with cleaning the formatting of and reprioritizing"
            " the following tasks: {task_names}."
            " Consider the ultimate objective of your team: {objective}."
            " Do not remove any tasks. Return the result as a numbered list, like:"
            " #. First task"
            " #. Second task"
            " Start the task list with number {next_task_id}."
        )
        prompt = PromptTemplate(
            template=task_prioritization_template,
            input_variables=["task_names", "next_task_id", "objective"],
        )
        return cls(prompt=prompt, llm=llm, verbose=verbose)
```        

####  执行链

执行链条对于两个版本的 BabyAGI 来说有所不同。对于无工具版本，执行过程非常简单；而对于有工具版本，执行过程会相对复杂一些。

在这个过程中，我们定义了一个执行代理，并传递了一些工具给它。这个执行代理是一个计划者，能够为给定的目标制定一个待办事项清单。我们传递了搜索和待办事项这两种工具给它，以便它能够在需要的时候进行搜索或者制定待办事项清单。

```
from langchain.agents import ZeroShotAgent, Tool, AgentExecutor
from langchain import OpenAI, SerpAPIWrapper, LLMChain
todo_prompt = PromptTemplate.from_template("You are a planner who is an expert at coming up with a todo list for a given objective. Come up with a todo list for this objective: {objective}")
todo_chain = LLMChain(llm=OpenAI(temperature=0), prompt=todo_prompt)
search = SerpAPIWrapper()
tools = [
    Tool(
        name = "Search",
        func=search.run,
        description="useful for when you need to answer questions about current events"
    ),
    Tool(
        name = "TODO",
        func=todo_chain.run,
        description="useful for when you need to come up with todo lists. Input: an objective to create a todo list for. Output: a todo list for that objective. Please be very clear what the objective is!"
    )
]


prefix = """You are an AI who performs one task based on the following objective: {objective}. Take into account these previously completed tasks: {context}."""
suffix = """Question: {task}
{agent_scratchpad}"""
prompt = ZeroShotAgent.create_prompt(
    tools, 
    prefix=prefix, 
    suffix=suffix, 
    input_variables=["objective", "task", "context","agent_scratchpad"]
)
from langchain.agents import ZeroShotAgent, Tool, AgentExecutor
from langchain import OpenAI, SerpAPIWrapper, LLMChain
todo_prompt = PromptTemplate.from_template("You are a planner who is an expert at coming up with a todo list for a given objective. Come up with a todo list for this objective: {objective}")
todo_chain = LLMChain(llm=OpenAI(temperature=0), prompt=todo_prompt)
search = SerpAPIWrapper()
tools = [
    Tool(
        name = "Search",
        func=search.run,
        description="useful for when you need to answer questions about current events"
    ),
    Tool(
        name = "TODO",
        func=todo_chain.run,
        description="useful for when you need to come up with todo lists. Input: an objective to create a todo list for. Output: a todo list for that objective. Please be very clear what the objective is!"
    )
]


prefix = """You are an AI who performs one task based on the following objective: {objective}. Take into account these previously completed tasks: {context}."""
suffix = """Question: {task}
{agent_scratchpad}"""
prompt = ZeroShotAgent.create_prompt(
    tools, 
    prefix=prefix, 
    suffix=suffix, 
    input_variables=["objective", "task", "context","agent_scratchpad"]
```
我们可以看到，这个执行器不是使用 "react" 方式，而是使用 ZeroShotAgent 代理，将提示语、前缀后缀以及输入变量一并输入。通过这种方式，我们可以更清楚地看到在执行过程中，这些部分如何组合在一起工作。

###  整合所有链

现在，我们有了一些函数，它们的作用是定义任务，运行任务，并设置一个循环使得任务能够持续运行。重要的是，这个系统并不是只运行一次任务就结束，而是通过一个循环，让系统不断地获取和执行任务。

在这个过程中，代码将所有的部分结合在一起，无论你是使用无工具版本的 BabyAGI，还是使用有工具版本，过程都是一样的。我们可以看到任务列表，对于每个任务，都有对应的链条在执行。

```
def get_next_task(task_creation_chain: LLMChain, result: Dict,                task_description:        str, task_list: List[str], objective: str) -> List[Dict]:
    """Get the next task."""
    incomplete_tasks = ", ".join(task_list)
    response = task_creation_chain.run(result=result, task_description=task_description, incomplete_tasks=incomplete_tasks, objective=objective)
    new_tasks = response.split('\n')
    return [{"task_name": task_name} for task_name in new_tasks if task_name.strip()]
``` 

```
def prioritize_tasks(task_prioritization_chain: LLMChain, this_task_id: int, task_list: List[Dict], objective: str) -> List[Dict]:
    """Prioritize tasks."""
    task_names = [t["task_name"] for t in task_list]
    next_task_id = int(this_task_id) + 1
    response = task_prioritization_chain.run(task_names=task_names, 
                                             next_task_id=next_task_id, 
                                             objective=objective)
    new_tasks = response.split('\n')
    prioritized_task_list = []
    for task_string in new_tasks:
        if not task_string.strip():
            continue
        task_parts = task_string.strip().split(".", 1)
        if len(task_parts) == 2:
            task_id = task_parts[0].strip()
            task_name = task_parts[1].strip()
            prioritized_task_list.append({"task_id": task_id, "task_name": task_name})
    return prioritized_task_list
```

```
def _get_top_tasks(vectorstore, query: str, k: int) -> List[str]:
    """Get the top k tasks based on the query."""
    results = vectorstore.similarity_search_with_score(query, k=k)
    if not results:
        return []
    sorted_results, _ = zip(*sorted(results, key=lambda x: x[1], reverse=True))
    return [str(item.metadata['task']) for item in sorted_results]

def execute_task(vectorstore, execution_chain: LLMChain, objective: str, task: str, k: int = 5) -> str:
    """Execute a task."""
    context = _get_top_tasks(vectorstore, query=objective, k=k)
    return execution_chain.run(objective=objective, context=context, task=task)
```    

LangChain 创建 BabyAGI 类

为了使这个过程更便于管理，我们为 BabyAGI 创建了一个类。在这个类中，我们可以添加任务，打印任务列表，打印下一个任务，打印任务结果。这些函数将能够与语言模型一起使用，使得所有的内容都能够同时运行。

实际的运行过程是在一个 While 循环中进行的。它会在获取到某个结果后退出，并根据这个结果进行下一步操作。我们可以看到，整个过程中发生的各种事情，包括创建新任务，重新设置优先级等等。
```
class BabyAGI(Chain, BaseModel):
    """Controller model for the BabyAGI agent."""

    task_list: deque = Field(default_factory=deque)
    task_creation_chain: TaskCreationChain = Field(...)
    task_prioritization_chain: TaskPrioritizationChain = Field(...)
    execution_chain: AgentExecutor = Field(...)
    task_id_counter: int = Field(1)
    vectorstore: VectorStore = Field(init=False)
    max_iterations: Optional[int] = None
        
    class Config:
        """Configuration for this pydantic object."""
        arbitrary_types_allowed = True

    def add_task(self, task: Dict):
        self.task_list.append(task)

    def print_task_list(self):
        print("\033[95m\033[1m" + "\n*****TASK LIST*****\n" + "\033[0m\033[0m")
        for t in self.task_list:
            print(str(t["task_id"]) + ": " + t["task_name"])

    def print_next_task(self, task: Dict):
        print("\033[92m\033[1m" + "\n*****NEXT TASK*****\n" + "\033[0m\033[0m")
        print(str(task["task_id"]) + ": " + task["task_name"])

    def print_task_result(self, result: str):
        print("\033[93m\033[1m" + "\n*****TASK RESULT*****\n" + "\033[0m\033[0m")
        print(result)
        
    @property
    def input_keys(self) -> List[str]:
        return ["objective"]
    
    @property
    def output_keys(self) -> List[str]:
        return []

    def _call(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Run the agent."""
        objective = inputs['objective']
        first_task = inputs.get("first_task", "Make a todo list")
        self.add_task({"task_id": 1, "task_name": first_task})
        num_iters = 0
        while True:
            if self.task_list:
                self.print_task_list()

                # Step 1: Pull the first task
                task = self.task_list.popleft()
                self.print_next_task(task)

                # Step 2: Execute the task
                result = execute_task(
                    self.vectorstore, self.execution_chain, objective, task["task_name"]
                )
                this_task_id = int(task["task_id"])
                self.print_task_result(result)

                # Step 3: Store the result in Pinecone
                result_id = f"result_{task['task_id']}"
                self.vectorstore.add_texts(
                    texts=[result],
                    metadatas=[{"task": task["task_name"]}],
                    ids=[result_id],
                )

                # Step 4: Create new tasks and reprioritize task list
                new_tasks = get_next_task(
                    self.task_creation_chain, result, task["task_name"], [t["task_name"] for t in self.task_list], objective
                )
                for new_task in new_tasks:
                    self.task_id_counter += 1
                    new_task.update({"task_id": self.task_id_counter})
                    self.add_task(new_task)
                self.task_list = deque(
                    prioritize_tasks(
                        self.task_prioritization_chain, this_task_id, list(self.task_list), objective
                    )
                )
            num_iters += 1
            if self.max_iterations is not None and num_iters == self.max_iterations:
                print("\033[91m\033[1m" + "\n*****TASK ENDING*****\n" + "\033[0m\033[0m")
                break
        return {}

    @classmethod
    def from_llm(
        cls,
        llm: BaseLLM,
        vectorstore: VectorStore,
        verbose: bool = False,
        **kwargs
    ) -> "BabyAGI":
        """Initialize the BabyAGI Controller."""
        task_creation_chain = TaskCreationChain.from_llm(
            llm, verbose=verbose
        )
        task_prioritization_chain = TaskPrioritizationChain.from_llm(
            llm, verbose=verbose
        )
        llm_chain = LLMChain(llm=llm, prompt=prompt)
        tool_names = [tool.name for tool in tools]
        agent = ZeroShotAgent(llm_chain=llm_chain, allowed_tools=tool_names)
        agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)
        return cls(
            task_creation_chain=task_creation_chain,
            task_prioritization_chain=task_prioritization_chain,
            execution_chain=agent_executor,
            vectorstore=vectorstore,
            **kwargs
        )
```        
在这个系统中，我们并没有使用 Pinecone 进行存储，而是选择在本地进行存储。这样，我们可以更直观地看到在这个过程中发生的每一件事。

虽然系统已经可以正常运行，但我认为如果添加一个额外的链条，用于生成一些摘要，例如一个最终报告，将会使结果更好。目前，虽然系统可以执行所有的任务，但是在最后得出结论的时候，可能会有一些不足。

在实际运行过程中，我们使用了 OpenAI 的语言模型，并将温度设为零。

```
llm = OpenAI(temperature=0)
```

例如，我们可以设置一个目标，就是找到在网上购买 yubikey 5C 的最便宜的价格和网站，然后将结果提供给我。我们可以看到，通过这样的设置，我们可以实现一些特定的目标。

```
OBJECTIVE = "Find the cheapest price and site to buy a Yubikey 5c online and give me the URL"
```


### 7.1.4  实例化 BabyAGI

我们开始实例化 BabyAGI 类并运行它。

```
llm = OpenAI(temperature=0)
```

首先，我们需要把语言模型和向量存储器传入，然后我们设置了一个最大的迭代次数，这是这个版本相比于先前版本的改进之处。在早前的版本中，程序会无限循环下去，而在这个版本中，我们可以通过设置迭代次数上限来限制循环的次数 (max_iterations: Optional [int] = 7)。


```
# Logging of LLMChains
verbose=False
# If None, will keep on going forever
max_iterations: Optional[int] = 7
# 实例化 BabyAGI
baby_agi = BabyAGI.from_llm(
    llm=llm,
    vectorstore=vectorstore,
    verbose=verbose,
    max_iterations=max_iterations
)
```

接下来，我们将目标输入到程序中，程序会制定一个待办事项列表并开始执行。例如，我们希望找到网上购买 YubiKey 5C 的最便宜的价格和网站，并获取 URL。程序则会生成一个待办事项列表，包括搜索在线零售商，比较不同零售商的价格，查找折扣或促销活动，以及阅读每个零售商的客户评论。

```
baby_agi({"objective": OBJECTIVE})
```

### 7.1.5  BabyAGI 执行结果

然后，程序会根据待办事项列表开始执行任务。对于每个任务，程序会进行一些搜索，比较不同在线商店中的 YubiKey 5C 价格，检查是否有折扣代码和促销活动，等等。

在整个过程中，程序会生成观察结果，比如它在哪些地方看到了 YubiKey，它找到的最便宜的价格是多少。如果在执行过程中遇到问题或者需要做出选择，程序也会返回相应的任务，并根据这些任务调整待办事项列表。

最后，程序会返回一个 URL (这个地址可能不能访问到商品），告诉我们可以在哪个网站以最便宜的价格购买 YubiKey 5C。但是，我们发现返回的 URL 并不总是有效的。例如，程序返回的 URL 可能会导致 404 错误，或者返回的价格可能和网站上显示的价格不一致。这些问题可能是由于程序运行的位置和我们实际的位置不同，或者可能是因为程序没有能力检查 URL 的有效性。

```
*****TASK LIST*****

3: Compare the price of Yubikey 5c at other online retailers to Yubico.com/store.
4: Check customer reviews of [Retailer Name] for Yubikey 5c.
5: Find out if [Retailer Name] offers any discounts or promotions for Yubikey 5c.
6: Research the return policy of [Retailer Name] for Yubikey 5c.
7: Determine the shipping cost for Yubikey 5c from [Retailer Name].
8: Check customer reviews of other online retailers for Yubikey 5c.
9: Find out if other online retailers offer any discounts or promotions for Yubikey 5c.
10: Research the return policy of other online retailers for Yubikey 5c.
11: Determine the shipping cost for Yubikey 5c from other online retailers.

*****NEXT TASK*****

3: Compare the price of Yubikey 5c at other online retailers to Yubico.com/store.


> Entering new AgentExecutor chain...
Thought: I should compare the prices of Yubikey 5c at other online retailers.
Action: Search
Action Input: Prices of Yubikey 5c at other online retailers
Observation: [{'position': 1, 'block_position': 'top', 'title': 'YubiKey 5C - OEM Official', 'price': '$55.00', 'extracted_price': 55.0, 'link': 'https://www.yubico.com/product/yubikey-5c', 'source': 'yubico.com/store', 'thumbnail': 'https://serpapi.com/searches/64ba2ffc49ecdb86973e7b26/images/ebce3fc64f92f22d58e2ea0dae58f9a2419c119482504cffef43e39a06787765.webp', 'extensions': ['45-day returns (most items)']}, {'position': 2, 'block_position': 'top', 'title': 'Yubico YubiKey 5C - USB security key', 'price': '$3,256.99', 'extracted_price': 3256.99, 'link': 'https://www.cdw.com/product/yubico-yubikey-5c-usb-security-key/7493450?cm_ven=acquirgy&cm_cat=google&cm_pla=NA-NA-Yubico_NY&cm_ite=7493450', 'source': 'CDW', 'shipping': 'Get it by 7/26', 'thumbnail': 'https://serpapi.com/searches/64ba2ffc49ecdb86973e7b26/images/ebce3fc64f92f22d58e2ea0dae58f9a2b4a2cbdc8de9b340a34c7f35661e9f75.webp'}, {'position': 3, 'block_position': 'top', 'title': 'YubiKey 5C NFC - OEM Official', 'price': '$55.00', 'extracted_price': 55.0, 'link': 'https://www.yubico.com/product/yubikey-5c-nfc', 'source': 'yubico.com/store', 'thumbnail': 'https://serpapi.com/searches/64ba2ffc49ecdb86973e7b26/images/ebce3fc64f92f22d58e2ea0dae58f9a2f0f2eed4b19c6081b5000768a9cc1878.webp', 'extensions': ['45-day returns (most items)']}]
Thought:
```

虽然这个系统还不完美，但是它确实为我们提供了一个基于链条的自动化流程，用来获取信息、制定待办事项列表，并执行任务。这个系统给我们展示了如何用简单的链条模型来处理复杂的问题。这是一个不断学习和思考的过程，我们可以根据需要调整提示，添加新的链条，或者改进现有的链条。




# 7.2 实现 AutoGPT


Auto-GPT 是一个开源的实验性应用程序，它展示了 GPT-4 语言模型的能力。这个由 GPT-4 驱动的程序，通过串联起 LLM（大型语言模型）的 "思考"，以实现你设定的任何目标。作为 GPT-4 全自主运行的首批示例之一，AutoGPT 拓宽了人工智能可能达到的边界。

想象一下，你正在编写一个复杂的报告，这需要从多个方面进行深入研究。这时候，你可以设定一个目标，让 Auto-GPT 帮你完成这项任务。你会看到 Auto-GPT 如何通过串联起大型语言模型的 "思考"，自动收集相关的信息，形成一个结构化的报告，这就如同你的个人助手在帮你分担 "工作的压力"。

在所有这些中，Auto-GPT 所展示的能力，无疑推动了人工智能的可能性。例如，它可以自动写作文章，分析大量数据，甚至参与到复杂的决策过程中，这些都是我们以前认为只有人类才能做的事情。正因为如此，Auto-GPT 无疑正在挑战我们对于人工智能的想象。

### 7.2.1 使用 LangChain 实现 AutoGPT

这一节将介绍如何使用 LangChain 的基本组件（如大型语言模型，提示模板，向量存储，嵌入，工具等）来实现一个 AutoGPT 模型。AutoGPT 是一个自动化的大型语言模型项目，它可以进行多种任务，包括文件读取、写入和搜索等。

安装库。

```bash
pip -q install  openai tiktoken
pip install git+https://github.com/hwchase17/langchain
```

设置密钥。

```
import os

os.environ["OPENAI_API_KEY"] = ""
```
我们来设置一些工具，这包括搜索工具、写文件工具和读文件工具，这些都是为了让 AutoGPT 能够更好地完成任务。

```
from langchain.utilities import SerpAPIWrapper
from langchain.agents import Tool
from langchain.tools.file_management.write import WriteFileTool
from langchain.tools.file_management.read import ReadFileTool

search = SerpAPIWrapper()
tools = [
    Tool(
        name="search",
        func=search.run,
        description="useful for when you need to answer questions about current events. You should ask targeted questions",
    ),
    WriteFileTool(),
    ReadFileTool(),
]
```

设置内存，这是用于存储模型在执行任务过程中的中间步骤。这对于理解模型的工作过程和改进模型的性能都非常重要。

```
from langchain.vectorstores import FAISS
from langchain.docstore import InMemoryDocstore
from langchain.embeddings import OpenAIEmbeddings
```
我们使用 OpenAI 的嵌入模型, 向量库使用 FAISS。

```
# Define your embedding model
embeddings_model = OpenAIEmbeddings()
# Initialize the vectorstore as empty
import faiss

embedding_size = 1536
index = faiss.IndexFlatL2(embedding_size)
vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})
```

在设置好工具和内存后，初始化模型和 AutoGPT。我们选择的模型是 ChatOpenAI，这是一个专为对话设计的大语言模型。

```
from langchain.experimental import AutoGPT
from langchain.chat_models import ChatOpenAI
```

```
agent = AutoGPT.from_llm_and_tools(
    ai_name="Tom",
    ai_role="Assistant",
    tools=tools,
    llm=ChatOpenAI(temperature=0),
    memory=vectorstore.as_retriever(),
)
# Set verbose to be true
agent.chain.verbose = True
```

接下来，运行一个示例，让 AutoGPT 为旧金山编写一份天气报告。这展示了 AutoGPT 在实际应用中的能力。

```
agent.run(["write a weather report for SF today"])
```

### 7.2.2 设置聊天历史记录的内存

最后，设置聊天历史记录的内存。这是除了用于存储模型中间步骤的内存之外的另一种内存。这种内存默认使用'ChatMessageHistory'，但可以改变。当你希望使用不同类型的内存，例如'FileChatHistoryMemory'时，这将非常有用。

```
from langchain.memory.chat_message_histories import FileChatMessageHistory

agent = AutoGPT.from_llm_and_tools(
    ai_name="Tom",
    ai_role="Assistant",
    tools=tools,
    llm=ChatOpenAI(temperature=0),
    memory=vectorstore.as_retriever(),
    chat_history_memory=FileChatMessageHistory("chat_history.txt"),
)
```
我们运行询问上海现在的天气如何, 测试这种方式是否生效。

```
agent.run(["write a weather report for Shanghai today"])
```
我的文件夹里多了 weather_report.txt 和 chat_history.txt 两个文件
```
{
  "thoughts": {
    "text": "Since I have completed the task of writing a weather report for San Francisco, I can now move on to the next task. One possible task could be to write a weather report for Shanghai today. This will allow me to provide the user with up-to-date weather information for Shanghai.",
    "reasoning": "Writing a weather report is a simple and straightforward task that I can easily accomplish. It will provide useful information to the user and help them plan their activities accordingly.",
    "plan": "- Use the 'search' command to search for the weather in Shanghai today.\n- Analyze the search results and extract the relevant information.\n- Write a weather report for Shanghai today.",
    "criticism": "I need to ensure that I accurately extract the weather information from the search results and provide a clear and concise weather report to the user.",
    "speak": "I will search for the weather in Shanghai today and provide you with a weather report."
  },
  "command": {
    "name": "search",
    "args": {
      "tool_input": "weather in Shanghai today"
    }
  }
}
```
这就是如何使用 LangChain 的基本组件来实现 AutoGPT。

# 用 LangChain 自制 chatPDF

我们这个 ChatPDF 项目是引入外部数据集对大语言模型进行微调，以生成更准确的回答的程序。

试想你是一个航天飞机设计师，你需要了解最新的航空材料技术，你可以将这个需求输入到我们的模型中，模型就会根据最新的数据集给出准确的答案。

我们的界面呈现的是人类与文档问答的聊天，但实质上，我们仍然是在与大语言模型交流，只不过这个模型现在被赋予了接入外部数据集的能力。就像你在与一位熟悉你公司内部文档的同事交谈，尽管他可能并未参与过这些文档的编写，但他可以准确地回答你的问题。

大语言模型之前，我们不能像聊天一样与文档交流，我们只能依赖于搜索。例如, 你正在为一项重要的报告寻找资料，你必须知道你需要查找的关键词，然后在大量的信息中筛选出你需要的部分。而现在，我们可以通过聊天的方式，即使不知道具体的关键词，也可以让模型根据我们的问题告诉我们答案。就好像你在问一位专业的图书馆员，哪些书籍可以帮助你完成这份报告。

那为什么我们要引入文档的外部数据集呢？这是因为大语言模型的训练数据截止到 2021 年 9 月，之后产生的知识和信息并未被包含进去。就像我们的模型是一个生活在过去的时间旅行者，他只能告诉你他离开的那个时刻之前的所有信息，对之后的事情一无所知。

我们的模型训练数据不仅包含参数，也引入了外部数据集进行训练。就好像我们通过电话，向一个在远方的朋友讲述最新的新闻，这样他就可以了解到最新的事情。我们的大语言模型也一样，通过外部数据集的输入，我们可以使它了解到最新的信息，从而进行更好的检索增强和微调，适应我们的实时需求。

引入外部数据集还有一个重要的目的，那就是修复大语言模型的 "机器幻觉"，避免给出错误的回答。试想一下，如果你向一个只知道过去信息的人询问未来的趋势，他可能会基于过去的信息进行推断，但这样的答案未必正确。所以我们通过引入最新的数据，让我们的模型能够更准确地回答问题，避免因为信息过时产生的误导。

我们现在使用的数据文档形式包括 pdf、json、word、excel 等，这些都是我们获取实时知识和数据的途径。这类程序现在非常受欢迎，比如最著名的 chatpdf 和 chatdoc, 还有针对各种特定领域的程序，如针对法律文档的程序。就像你在阅读各种格式的书籍一样，不同的程序能够提供不同的知识和信息。



### 7.3.1 程序流程

我们的实现方式是利用 Langchain 已实现的向量存储、嵌入以及使用查询和检索相关的链，来获取外部数据集，处理文档，进行相关性检索后合并处理，置入大语言模型的提示模板中，实现与 PDF 文件交流的目的。

我们选定的文档是 Reid Hoffman 写的一本关于 GPT-4 和人工智能的书，我们将下载这本 PDF 并将其转化为可查询和交互的形式。

#### 加载文件且切分块

首先，我们要加载 PDF 文件，并将其分割成块。接下来，我们将创建一个向量存储，并对其进行查询。这个过程很像使用搜索引擎，我们在向量存储中查找相关内容，但与基于关键词的搜索不同，我们是基于语义的。

#### 嵌入和向量存储

为每个块创建嵌入，这个嵌入会是一个向量，代表该块中的所有信息。当我们需要查询某个信息时，我们会将问题嵌入为一个向量，然后发送给向量存储器。向量存储器会返回最相关的信息块，然后将其与问题一起传递给语言模型。

#### 语言模型 I/O

语言模型会阅读这些信息，并根据这些信息决定答案是什么。这种基于语义搜索的概念并不新，甚至使用大语言模型来查询文本的想法也已经存在很长时间了。

#### LangChain 创建链

我们需要做的关键步骤包括：加载文档，分割文档，创建嵌入，将嵌入放入向量存储。接下来，我们将使用 LangChain 创建我们的链，以便我们可以进行查询，查找向量存储中的信息，将其带入链中，然后与问题结合，最终得出一个答案。



### 7.3.2 环境和工具

#### 安装包
```
!pip -q install langchain openai tiktoken PyPDF2 faiss-cpu

```
#### 设置密钥
```

import os

os.environ["OPENAI_API_KEY"] = ""
```

#### 下载 PDF 文本 
```
# 可以复制到浏览器后，直接保存在本地电脑上。
!wget -q https://www.impromptubook.com/wp-content/uploads/2023/03/impromptu-rh.pdf
```

### 导入库和 PDF 加载器

首先，我们需要一个 PDF 阅读器。虽然我们这次只使用了一个基础的 PDF 阅读器，你也可以根据需要选择更合适的 PDF 阅读器。我们通过 PDF 阅读器将 PDF 文档读取成一个长字符串。这个过程可能会遇到一些格式问题，比如奇怪的空格等，但每个项目都有自己独特的数据处理方法。不论是使用基础的处理方式，还是使用诸如'unstructured'库、AWS 或 Google Cloud 的 API，都有可能。


先导入阅读器 `PdfReader`, 嵌入模型 `OpenAIEmbeddings`, 文档切分器 `CharacterTextSplitter`, 向量存储库 `FAISS`  ： 

```
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS 
```

加载之前准备好的 PDF 素材：

```
# 括号内填入当前电脑保存pdf文件的位置. 
doc_reader = PdfReader('/content/impromptu-rh.pdf')
```
打印 `doc_reader` 看看 :
```
<PyPDF2._reader.PdfReader at 0x7f119f57f640>
```
将 PDF 文档转化为可用的 `raw_text` 格式：

```
# read data from the file and put them into a variable called raw_text
raw_text = ''
for i, page in enumerate(doc_reader.pages):
    text = page.extract_text()
    if text:
        raw_text += text
```     
我们可以打印 `raw_text` 的结果字符串长度，看看是不是转换成功了：

```
len(raw_text) # 得到 356710 的结果
```


### 7.3.3 文本拆分

我们需要将获取到的长字符串拆分为适合分析的小段落。

我们的方法很简单，就是将这个长字符串按照字符数拆分。比如我们可以设定每 1000 个字符为一个块, `chunk_size = 1000`。


```

# Splitting up the text into smaller chunks for indexing
text_splitter = CharacterTextSplitter(        
    separator = "\n",
    chunk_size = 1000,
    chunk_overlap  = 200, #striding over the text
    length_function = len,
)
texts = text_splitter.split_text(raw_text)

```

我们总共切了 448 个块：

```
len(texts) # 448
``` 

*注意*： 在这个代码片段中，`chunk_overlap` 参数用于指定文本切分时的重叠量（overlap）。它表示在切分后生成的每个分块之间重叠的字符数。具体来说，这个参数表示每个分块的前后，两个分块之间会有多少个字符是重复的。举例来说 chunkA 和 chunkB, 他们有 200 个字符是重复的。

然后，我们采用滑动窗口的方法来拆分文本。即每个块之间会有部分字符重叠，比如在每 1000 个字符的块上，我们让前后两块有 200 个字符重叠。这样做的目的是避免关键信息被切分，而且即使有些信息出现在了多个块中，因为我们是在获取整体语义，所以这些重叠的块在语义上也会有所区别。

我们可以随机打印一块的内容：

```
texts[20]
```

输出是：

```
'million registered users. \nIn late January 2023, Microsoft1—which had invested $1 billion \nin OpenAI in 2019—announced that it would be investing $10 \nbillion more in the company. It soon unveiled a new version of \nits search engine Bing, with a variation of ChatGPT built into it.\n1 I sit on Microsoft’s Board of Directors. 10Impromptu: Amplifying Our Humanity Through AI\nBy the start of February 2023, OpenAI said ChatGPT had \none hundred million monthly active users, making it the fast-\nest-growing consumer internet app ever. Along with that \ntorrent of user interest, there were news stories of the new Bing \nchatbot functioning in sporadically unusual ways that were \nvery different from how ChatGPT had generally been engaging \nwith users—including showing “anger,” hurling insults, boast-\ning on its hacking abilities and capacity for revenge, and basi-\ncally acting as if it were auditioning for a future episode of Real \nHousewives: Black Mirror Edition .'
```



### 7.3.4  创建嵌入和检索

有了分好的小块文本，我们就可以为这些文本创建嵌入了。在这个步骤，我们使用了 OpenAI 的嵌入技术。

```
# Download embeddings from OpenAI
embeddings = OpenAIEmbeddings()
```
我们先把文本传给嵌入制造器，然后通过 FAISS 库创建向量存储本身。

```
docsearch = FAISS.from_texts(texts, embeddings)
```

至此，我们已经将原本的 PDF 文档转化为了可以进行机器学习的向量数据。

就是这么简单，接下来我们就可以向这个 PDF 问问题了。

### 相似度检索

现在，我们可以使用这些向量数据来进行搜索匹配了。我们以一个实际的查询为例：“GPT-4 如何改变社交媒体？”。

```
query = "how does GPT-4 change social media?"
docs = docsearch.similarity_search(query) # 这是搜索匹配的文字结果数组。
```

我们把这个查询传给文档搜索系统，使用相似度匹配搜索 `similarity_search`，在向量数据中寻找与查询最匹配的文档。

我们的搜索结果将包括与查询最接近的四个文档，而这些文档都是通过我们的嵌入函数进行嵌入的。

```

len(docs)  # 结果为 4 代表有 4 处地方跟问题有关系

```

我们尝试打印第一个 `docs[0]`
在我们的搜索结果中，首位的文档中多次提到了“社交媒体”，看来我们的查询效果还是很好的。

```
Document(page_content='rected ways that tools like GPT-4 and DALL-E 2 enable.\nThis is a theme I’ve touched on throughout this travelog, but \nit’s especially relevant in this chapter. From its inception, social \nmedia worked to recast broadcast media’s monolithic and \npassive audiences as interactive, democratic communities, in \nwhich newly empowered participants could connect directly \nwith each other. They could project their own voices broadly, \nwith no editorial “gatekeeping” beyond a given platform’s terms \nof service.\nEven with the rise of recommendation algorithms, social media \nremains a medium where users have more chance to deter -\nmine their own pathways and experiences than they do in the \nworld of traditional media. It’s a medium where they’ve come \nto expect a certain level of autonomy, and typically they look for \nnew ways to expand it.\nSocial media content creators also wear a lot of hats, especially \nwhen starting out. A new YouTube creator is probably not only', metadata={})
```

这就是如何利用 OpenAI 技术处理 PDF 文档，将海量的信息提炼为可用的数据的全部步骤。是不是很简单，赶紧动手做起来吧~

我们现在只有一个 PDF 文档，实现代码也很简单，Langchain 给了很多组件，我们完成得很快。接下来，我们处理多文档的提问，现实是我们要获取到真实的信息，通过会跨越多个文档，才能提取有用的信息。比如读取金融研报，新闻综合报道等等。


### 7.3.5  进阶 Stuff 链

在上一节中，我们加载了一个 PDF 文档，转化格式，切分字符后，创建向量数据来进行搜索匹配获得了问题的答案。一旦我们有了已经处理好的文档，我们就可以开始构建一个简单的问答链。现在我们看看 如何使用 Langchain 构建问答链。 

在这个过程中，我们使用了 OpenAI 的模型，并选择了 Langchain 的现有文档处理链中 一种被称为 "stuff" 的链类型。在这种模式下，我们只是将所有内容都放在一个调用中，理想情况下，我们放入的内容应该少于 4000 个令牌。

除了 "stuff" 之外，Langchain 文档处理链还有 精化（Refine）、Map reduce 、重排（Map re-rank）。后面我们会再次用到。

```

from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
chain = load_qa_chain(OpenAI(), 
                      chain_type="stuff") # we are going to stuff all the docs in at once
```

### 运行链，构建查询

下一步，我们要构建我们的查询。首先，我们使用向量存储中返回的内容作为上下文片段来回答我们的问题。然后，我们将这个查询传给语言模型链。语言模型链会回答这个查询，给出相应的答案。

例如，我们可能会问 "这本书的作者是谁？"，然后将该查询传递给向量存储进行相似性搜索。系统会返回最相似的四个文档，我们将这些文档传递给语言模型链并给出查询，然后系统会给出一个答案。

```
query = "who are the authors of the book?"
docs = docsearch.similarity_search(query)
chain.run(input_documents=docs, question=query)

```
看看他回答了什么:

```
' The authors of the book are Reid Hoffman and Ben Casnocha.'
```


#### 选择返回的文档数量

我们可以设置返回的文档数量。默认情况下，系统会返回四个最相关的文档，但我们可以更改这个数字。

例如，我们可以设置返回前六个或更多的搜索结果。

```
query = "who is the book authored by?"
docs = docsearch.similarity_search(query,k=6)
chain.run(input_documents=docs, question=query)
```

然而，我们需要注意的是，如果我们设置返回的文档数量过多, 比如设置 `k=20`，那么总的令牌数可能会超过模型的最大上下文长度，导致错误。例如，你使用的模型的最大上下文长度为 4097，但如果我们请求的令牌数超过了 5000，系统就会报错。

设置返回的文档数量为 `k=6`，我们获取的结果是：
```
' The book is authored by Reid Hoffman and Aria Finger.'
```

### 7.3.6  进阶 map_rerank 链

为了解决这个问题，我们可以更改链类型。我们在之前的文章中看过许多不同的链类型。

"stuff" 类型优势是把所有内容都放在一起的地方。任何时候我们可以使用 "stuff"，最好就使用它，通用且节省成本。

我们还可以使用 "map_reduce" 在并行计算中对每个文档进行操作，但这可能会导致对 API 进行过多的调用，增加成本。

继续我们的讨论，我们将深入了解如何通过 Langchain 技术从 PDF 文档中提取有用的信息，特别是我们将重点讨论如何处理多个查询和理解返回结果。

第一种方式是使用不同类型的查询链类型。这里我们使用 `map_rerank` 这种类型，提高查询的质量。

##### `map_rerank` 优化查询质量

让我们从提出更复杂的查询开始。比如说，我们想要知道 "OpenAI 是什么"，并且我们想要获取前 10 个最相关的查询结果。在这种情况下，OpenAI 会返回多个答案，而不仅仅是一个。我们可以看到它不只返回一个答案，而是根据我们的需求返回了每个查询的答案和相应的评分。

```

from langchain.chains.question_answering import load_qa_chain

chain = load_qa_chain(OpenAI(), 
                      chain_type="map_rerank",
                      return_intermediate_steps=True
                      ) 

query = "who are openai?"
docs = docsearch.similarity_search(query,k=10)
results = chain({"input_documents": docs, "question": query}, return_only_outputs=True)
results
```

重要的参数是 `return_intermediate_steps=True`, 设置这个参数我们可以看到 `map_rerank` 是如何对检索到的文档进行打分的。


#####  理解评分系统

OpenAI 技术对返回的每个查询结果进行了评分。比如说，OpenAI 在这本书中被多次提及，因此它的评分可能会有 80 分，90 分甚至 100 分。我们可以假设 OpenAI 可能选择了评分为 100 分的两个或三个查询，然后将它们合并，最终给出了我们的输出。

```
{'intermediate_steps': [{'answer': ' OpenAI is an organization that released text-to-image generation tool DALL-E 2 and ChatGPT in April 2022 and are giving millions of users hands-on access to these AI tools.',
   'score': '80'},
  {'answer': ' OpenAI is a research laboratory whose founding goal is to develop technologies that put the power of AI directly into the hands of millions of people.',
   'score': '80'},
  {'answer': ' OpenAI is a research organization that develops and shares artificial intelligence tools for the benefit of humanity.',
   'score': '100'},
  {'answer': ' OpenAI is a technology company focused on using artificial intelligence to solve real-world problems.',
   'score': '80'},
  {'answer': ' OpenAI is a company co-founded by Sam Altman that is developing AI technologies and allowing individuals to participate in the development process.',
   'score': '90'},
  {'answer': ' OpenAI is a research laboratory that focuses on artificial intelligence technologies.',
   'score': '80'},
  {'answer': ' OpenAI is a non-profit artificial intelligence research company that was founded in 2015 with the goal of advancing digital intelligence in the way that is most likely to benefit humanity as a whole. ',
   'score': '90'},
  {'answer': ' OpenAI is a nonprofit artificial intelligence (AI) research organization. Microsoft invested $1 billion in OpenAI in 2019 and announced in late January 2023 that it would be investing an additional $10 billion.',
   'score': '100'},
  {'answer': ' OpenAI is an artificial intelligence research laboratory founded by Elon Musk and Sam Altman.',
   'score': '100'},
  {'answer': ' OpenAI is an artificial intelligence research laboratory founded in December 201It is a non-profit organization with the mission to ensure that artificial general intelligence (AGI) benefits all of humanity. ',
   'score': '90'}],
 'output_text': ' OpenAI is a research organization that develops and shares artificial intelligence tools for the benefit of humanity.'}
```

评分后，模型输出一个最终的答案, `'score': '100'` 得分 100 的那个答案：

```
results['output_text'] 
```

```
' OpenAI is a research organization that develops and shares artificial intelligence tools for the benefit of humanity.'
```

为了搞清楚为什么模型会评分，做出判断，我们可以打印 prompt 提示模板：

```
# check the prompt
chain.llm_chain.prompt.template
```
看完模板的一刻，你肯定有所顿悟：

```
"Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.\n\nIn addition to giving an answer, also return a score of how fully it answered the user's question. This should be in the following format:\n\nQuestion: [question here]\nHelpful Answer: [answer here]\nScore: [score between 0 and 100]\n\nHow to determine the score:\n- Higher is a better answer\n- Better responds fully to the asked question, with sufficient level of detail\n- If you do not know the answer based on the context, that should be a score of 0\n- Don't be overconfident!\n\nExample #1\n\nContext:\n---------\nApples are red\n---------\nQuestion: what color are apples?\nHelpful Answer: red\nScore: 100\n\nExample #2\n\nContext:\n---------\nit was night and the witness forgot his glasses. he was not sure if it was a sports car or an suv\n---------\nQuestion: what type was the car?\nHelpful Answer: a sports car or an suv\nScore: 60\n\nExample #3\n\nContext:\n---------\nPears are either red or orange\n---------\nQuestion: what color are apples?\nHelpful Answer: This document does not answer the question\nScore: 0\n\nBegin!\n\nContext:\n---------\n{context}\n---------\nQuestion: {question}\nHelpful Answer:"
```


### 7.3.7  RetrievalQA 链

除了单个查询，我们还可以使用链式查询。我们可以开始将这些查询放在一个链条中。

链式查询指的是查询向量存储链和语言模型的链。

例如我们可以有一个查询向量存储的链，以及一个查询语言模型的链。然后，我们可以将这些链条组合在一起，创建一个检索 QA 链条 。


#### 使用 RetrievalQA 链

RetrievalQA 链是 Langchain 已经封装好的索引查询问答链。实例化之后，我们可以直接把问题扔给它，而不需要 `chain.run()`。简化了很多步骤，获得了比较稳定的查询结果。

为了创建这样的链，我们需要一个检索器。我们可以使用之前设置好的 docsearch, 作为检索器，并且我们可以设置返回的文档数量 `"k":4`。

```
docsearch = FAISS.from_texts(texts, embeddings) # 检索器是向量库数据
```

我们可以将这些参数传递给链条类型 "stuff"，它会为我们返回源文档。（选择 "stuff" 类型的原因：跟第一个 "stuff" 类型 和 `map_reduce` 类型对比答案的质量）。


```
from langchain.chains import RetrievalQA

# set up FAISS as a generic retriever 
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":4})  # as_retriever方法是构建检索器 

# create the chain to answer questions 
rqa = RetrievalQA.from_chain_type(llm=OpenAI(), 
                                  chain_type="stuff", 
                                  retriever=retriever, 
                                  return_source_documents=True)
```

#### 返回结果和源文档

当我们查询 "OpenAI 是什么" 时，我们不仅会得到一个答案，还会得到源文档 `source_documents`。源文档是返回结果的参考文档，它可以帮助我们理解答案是如何得出的。

```
rqa("What is OpenAI?")
```

```
{'query': 'What is OpenAI?',
 'result': ' OpenAI is a research organization that develops and shares artificial intelligence tools for the benefit of humanity.',
 'source_documents': [Document(page_content='thing that has largely been happening to individuals rather \nthan for them—an under-the-radar force deployed by Big Tech \nwithout much public knowledge, much less consent, via tech -\nnologies like facial recognition and algorithmic decision-mak-\ning on home loans, job applicant screening, social media rec-\nommendations, and more.\nA founding goal of OpenAI was to develop technologies that put \nthe power of AI directly into the hands of millions of people..... #😊后面我省略了。
```

#### 直接返回结果设置

如果我们不需要中间步骤和源文档，只需要最终答案，那么我们可以直接请求返回结果。将代码:

```
return_source_documents=True
```
改为
```
return_source_documents=False
```

比如说，我们问 "What does gpt-4 mean for creativity?"

```
query = "What does gpt-4 mean for creativity?"
rqa(query)['result']
```
它会直接返回结果，不包括源文档。

```
' GPT-4 can amplify the creativity of humans by providing contextualized search, versatile brainstorming and production aid, and the ability to generate dialogue and branching narratives for interactive characters.'
```






## 7.4 对话式表单

这一节，我们一起探索这个由大语言模型驱动的提问和用户回答的程序。它并不是我们常见的 AI 程序，这样的程序并非人类提出问题，AI 进行回答。角色发生了转变，AI 主动提出问题，人类进行回答。

这类程序已经被广泛地应用到各种生活场景中。想象一下，你正在参加一个公司的招聘，面试的过程全由这个程序负责。它会向你提出一系列关于岗位的问题，让你来回答，如同真实的面试官。或者，你每天要通过几百个人好友申请、打招呼、了解需求等，这个程序会自动跟新好友聊天，根据他们的回答来更新信息。还有一种情况是，你正在填写一个报名表，这个程序会根据你之前的回答，逐步引导你完成报名。这些都是具体生活中这类程序的使用案例，可以看出其实用性。

一个典型的这类程序需要完成两个主要任务。首先，我们需要让语言模型只负责提问，而不进行回答，同时限制问题的范围。以招聘程序为例，程序只会提出关于岗位认识的问题，让面试者进行回答。

其次，程序需要根据用户的回答来更新数据库和下一个问题。例如，有个用户回答 "我叫美丽"，程序就能够识别出这个用户的名字是 "美丽"，并将其保存到数据库中。然后，程序会检查是否还有其他信息缺失，比如用户的居住城市或邮箱地址等，如果有缺失的信息，它就会选择相应的问题进行提问，如 "你住在哪里？"。一旦所有需要的信息都收集齐全，程序就会结束这一次的对话。

### 7.4.1 OpenAI 函数的标记链

这里将介绍如何创建一个对话式表单，实现用户以自然对话的方式填写表单信息。

在网页上，我们经常见到表单，用户需要填写详细信息。在网页上处理这些表单非常容易，因为信息可以很容易地解析和处理。但是，如果我们将表单放入一个聊天机器人中，并且希望用户能够以自然对话的方式回答。

我们将使用 OpenAI 函数的标记链来给用户的信息做“标记”。

标记链是使用 OpenAI 函数参数来指定一个标记文档的模式。这有助于我们确保模型输出我们想要的精确标签，以及它们对应的类型。

比如我们正在处理一个大量的文本数据，我们希望分析每一段文本的情绪是积极的还是消极的。在这种情况下，我们就可以使用标记链来实现这个功能。我们需要的不仅仅是模型的输出结果，更重要的是，这些结果必须是我们想要的，比如具有情绪类型的标签。

标记链需要在我们想要给文本标注特定属性的时候使用。例如，我们可能会问：“这条信息的情绪是什么？”在这个例子中，“情绪”就是我们想要标注的特定属性，而标记链就可以帮助我们实现这个目标。

通过这种方式，我们不仅可以标注出文本的情绪，还可以标注出文本的其他属性，如主题，作者的观点，等等。这个过程就好像给文本贴上了一张张的标签，让我们可以更快更准确地理解和分析文本。


### 7.4.2 标记链的使用

安装包和设置密钥：

```bash
pip -q install  openai tiktoken
pip install git+https://github.com/hwchase17/langchain
```
注意，因为标记链需要 langchain 新版本才支持，所以我们安装最新版本的 Langchain。

```python
import os
os.environ["OPENAI_API_KEY"] = ""
```
导入类和方法。

```
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from enum import Enum
from langchain.chains.openai_functions import (
    create_tagging_chain,
    create_tagging_chain_pydantic,
)
```

我们创建一个 Pydantic 类，该类用于获取用户的姓名、城市、电子邮件等信息。

``` python
class PersonalDetails(BaseModel):
    # 定义数据的类型
    name: str = Field(
        ...,
        description = "这是用户输入的名字"
    )
    city: str = Field(
        ...,
        description = "这是用户输入的居住城市"
    )
    email: str = Field(
        ...,
        description = "这是用户输入的邮箱地址"
    )
```

我们使用 OpenAI 的聊天模型。

``` python
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")
```

我们还创建了一个标记链，用于检查过滤链的响应，并将用户详细信息填写到相应的字段中。


``` python
chain = create_tagging_chain_pydantic(PersonalDetails,llm)
```

然后，我们实例化一个空的用户记录，并创建一个标记链来处理用户输入的对话。

#### 示例运行

我们通过一个示例演示了对话式表单的运行过程。用户需要提供姓名、城市和电子邮件。

``` python
test_str1 = "你好，我是美丽，我住在上海浦东，我的邮箱是： liteli1987@gmail.com"
test_res1 = chain.run(test_str1)
```

用户的对话成功被标记到 Pydantic 数据类：

```
PersonalDetails(name='美丽', city='上海浦东', email='liteli1987@gmail.com')
```

我们还可以做一些测试，查看标记链的工作效果。例如我们并不告诉它，我的名字是什么，用人称代词“我”。

```
test_str2 = "我的邮箱是： liteli1987@gmail.com"
test_res2 = chain.run(test_str2)
test_res2
```

它仍然会记录到邮箱这一项。

```
PersonalDetails(name='', city='', email='liteli1987@gmail.com')
```

我们还可以加入一些干扰信息，比如我告诉他，我的邮箱，顺带告诉他我弟弟的邮箱。

```
test_str3 = "我叫美丽，我弟弟的邮箱是：1106968391@qq.com"
test_res3 = chain.run(test_str3)
test_res3
```
它并不会把我弟弟的邮箱记录到我的信息里。

```
PersonalDetails(name='美丽', city='', email='')
```

### 7.4.3 创建提示模板

还记得我们这个程序需要完成的二大任务吗？第一个任务便是我们需要让语言模型只负责提问，而不进行回答，同时限制问题的范围。我们可以设置提示模板, 运行一个大语言模型链完成这一目标。

```
def ask_for_info(ask_for=["name","city","email"]):
    # 定义一个提示模板
    first_prompt = ChatPromptTemplate.from_template(
        """
        假设你现在是一名前台，你现在需要对用户进行询问他个人的具体信息。
        不要跟用户打招呼！你可以解释你需要什么信息。不要说“你好！”！
        接下来你和用户之间的对话都是你来提问，凡是你说的都是问句。
        你每次随机选择{ask_for}列表中的一个项目，向用户提问。
        比如["name","city"]列表，你可以随机选择一个"name", 你的问题就是“请问你的名字是？”
        """
    )
    info_gathering_chain = LLMChain(llm=llm, prompt=first_prompt)
    chat_chain = info_gathering_chain.run(ask_for=ask_for)
    return chat_chain
```

运行这个函数，我们便可以让大语言模型，只提问不回答。我们初始化一个问题。

```
ask_for_info(ask_for=["name","city","email"])
```
直接运行后，大语言模型发起了第一个提问。

```
'请问你的名字是？'
```


### 7.4.4 数据更新和检查

我们定义一个函数，用于检查数据是否填写完整。

```
def check_what_is_empty(user_personal_details):
    ask_for = []
    # 检查项目是否为空
    for field,value in user_personal_details.dict().items():
        if value in [None, "", 0]: 
            print(f"Field '{field}' 为空" )
            ask_for.append(f'{field}')
    return ask_for
```
我们假设 007 用户, 初始数据都没有填写：

```
user_007_personal_details = PersonalDetails(name="",city="",email="")
```

运行函数，查看哪些数据没有填写：

```
ask_for = check_what_is_empty(user_007_personal_details)
ask_for
```
函数调用后，显示 007 的姓名、城市和邮箱都没有填写。

```
Field 'name' 为空
Field 'city' 为空
Field 'email' 为空
['name', 'city', 'email']
```

我们再来定义一个函数，用于获取用户输入信息并且更新用户的信息。

```
def add_non_empty_details(current_details:PersonalDetails, new_details:PersonalDetails):
    # 这是已经填好的用户信息
    non_empty_details = {k:v for k,v in new_details.dict().items() if v not in [None, "", 0]}
    update_details = current_details.copy(update=non_empty_details)
    return update_details
```    

AI 每次提问，用户回答，这个函数根据这个用户的回答更新内存中的用户信息。

```
res = chain.run("我的名字007") 
user_007_personal_details = add_non_empty_details(user_007_personal_details,res)
user_007_personal_details
```

运行标记链后，更新一条数据。

```
PersonalDetails(name='007', city='', email='')
```

更新后，程序需要知道哪些数据没有填写。

```
ask_for = check_what_is_empty(user_007_personal_details)
ask_for
```

调用检查函数后，我们可以看到。

```
["city","email"]
```

刚刚我们定义函数 “check_what_is_empty” 检查哪几项没填，定义函数 “add_non_empty_details” 更新用户信息。有了这两个函数，结合我们的提示模板，实现机器人发起提问。

```
def decide_ask(ask_for=["name","city","email"]):
    if ask_for:
        ai_res = ask_for_info(ask_for=ask_for)
        print(ai_res)
    else:
        print("全部填写完整")
decide_ask(ask_for)  
```
我们定义一个函数，根据 “check_what_is_empty” 检查的结果，决定是否运行 “ask_for_info” 函数。 “ask_for_info” 函数内实现了调用我们的提示模板，运行 LLMChain 链。

我们假设 999 用户过来做测试。

```
user_999_personal_details = PersonalDetails(name="",city="",email="")
```

启动程序，开始提问。

```
decide_ask(ask_for)
```
AI 开始问 999 用户。

```
请问你的名字是？
```

999 用户回答后，AI 更新了该用户的信息。

```
str999 = "我的名字是999"
user_999_personal_details, ask_for_999 = filter_response(str999,user_999_personal_details)
decide_ask(ask_for_999)
```
检查完邮箱地址仍然为空，AI 继续问“请问您的电子邮件地址是多少？”。
```
Field 'email' 为空
请问您的电子邮件地址是多少？
```
999 用户回答自己的邮箱。

```
str999 = "XX@XX.com"
user_999_personal_details, ask_for_999 = filter_response(str999,user_999_personal_details)
decide_ask(ask_for_999)
```
AI　停止提问。

```
＇全部填写完整＇
```




