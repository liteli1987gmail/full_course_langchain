LLM class that expect subclasses to implement a simpler call method.

    The purpose of this class is to expose a simpler interface for working
    with LLMs, rather than expect the user to implement the full _generate method.

- **class InMemoryCache(BaseCache)**：在内存中存储内容的缓存。
- **class FullLLMCache(Base)**：用于完整的LLM缓存（所有世代）的SQLite表。
- **class SQLAlchemyCache(BaseCache)**：使用SQAlchemy作为后端的缓存。
- **class RedisCache(BaseCache)**：使用Redis作为后端的缓存。
- **class RedisSemanticCache(BaseCache)**：使用Redis作为向量存储后端的缓存。
- **class GPTCache(BaseCache)**：使用GPTCache作为后端的缓存。
- **class MomentoCache(BaseCache)**：使用Momento作为后端的缓存。参见 [https://gomomento.com/](https://gomomento.com/)
- **class EmbeddingsRedundantFilter(BaseDocumentTransformer, BaseModel)**：通过比较它们的嵌入来删除冗余文档的过滤器。
- **class Requests(BaseModel)**：对请求的包装器，用于处理身份验证和异步请求。此包装器的主要目的是处理身份验证（保存标头）并在同一基本对象上启用简单的异步方法。
- **class TextRequestsWrapper(BaseModel)**：请求库的轻量级文本输出包装器。此包装器的主要目的是始终返回文本输出。
- **class HumanMessage(BaseMessage)**：人类发出的消息类型。
- **class AIMessage(BaseMessage)**：AI发出的消息类型。- **class SystemMessage(BaseMessage)**：系统消息的类型。
- **class FunctionMessage(BaseMessage)**：用于序列化的消息类型。
- **class ChatMessage(BaseMessage)**：具有任意发言者的消息类型。
- **class RunInfo(BaseModel)**：包含运行的所有相关元数据的类。
- **class ChatResult(BaseModel)**：包含聊天结果的所有相关信息的类。
- **class LLMResult(BaseModel)**：包含LLM结果的所有相关信息的类。
- **class FileChatMessageHistory(BaseChatMessageHistory)**：将历史记录存储在本地文件中的聊天消息历史记录。参数：file_path：用于存储消息的本地文件的路径。
- **class BaseOutputParser(BaseLLMOutputParser, ABC, Generic[T])**：解析LLM调用的输出的类。输出解析器有助于结构化语言模型的响应。
- **class NoOpOutputParser(BaseOutputParser[str])**：仅返回原始文本的输出解析器。
- **class TextSplitter(BaseDocumentTransformer, ABC)**：将文本分割为块的接口。
- **class BaseSingleActionAgent(BaseModel)**：基本代理类。
- **class BaseMultiActionAgent(BaseModel)**：基本代理类。
- **class AgentOutputParser(BaseOutputParser)**：将文本解析为代理动作/完成的类。
- **class LLMSingleActionAgent(BaseSingleActionAgent)**：返回代理的字典表示。
- **class Agent(BaseSingleActionAgent)**：负责调用语言模型并决定动作的类。这由LLMChain驱动。在LLMChain中的提示必须包含一个名为“agent_scratchpad”的变量，代理可以在其中存储其中间工作。
- **class ExceptionTool(BaseTool)**：由使用工具的代理组成。
- **class InvalidTool(BaseTool)**：遇到无效工具名称时运行的工具。
- **class BaseToolkit(BaseModel)**：负责定义一组相关工具的类。
- **class AzureCognitiveServicesToolkit(BaseToolkit)**：用于Azure认知服务的工具包。
- **class FileManagementToolkit(BaseToolkit)**：用于与本地文件交互的工具包。
- **class GmailToolkit(BaseToolkit)**：用于与Gmail交互的工具包。
- **class JiraToolkit(BaseToolkit)**：Jira工具包。
- **class JsonToolkit(BaseToolkit)**：用于与JSON规范交互的工具包。
- **class NLAToolkit(BaseToolkit)**：自然语言API工具包定义。
- **class RequestsGetToolWithParsing(BaseRequestsTool, BaseTool)**：将控制器公开为工具。使用计划从规划器调用工具，并动态创建一个仅包含相关文档的控制器代理以限制上下文。
- **class RequestsToolkit(BaseToolkit)**：用于进行请求的工具包。
- **class OpenAPIToolkit(BaseToolkit)**：用于与OpenAPI API交互的工具包。
- **class PlayWrightBrowserToolkit(BaseToolkit)**：用于Web浏览器工具的工具包。
- **class PowerBIToolkit(BaseToolkit)**：用于与PowerBI数据集交互的工具包。
- **class SparkSQLToolkit(BaseToolkit)**：用于与Spark SQL交互的工具包。
- **class SQLDatabaseToolkit(BaseToolkit)**：用于与SQL数据库交互的工具包。
- **class VectorStoreInfo(BaseModel)**：关于向量存储的信息。
- **class VectorStoreToolkit(BaseToolkit)**：用于与向量存储交互的工具包。
- **class VectorStoreRouterToolkit(BaseToolkit)**：用于在向量存储之间进行路由的工具包。
- **class ZapierToolkit(BaseToolkit)**：Zapier工具包。
- **class OpenAIFunctionsAgent(BaseSingleActionAgent)**：由OpenAI函数驱动的代理。参数：llm：应为ChatOpenAI的实例，具体为支持使用`functions`的模型。tools：此代理可以访问的工具。prompt：此代理的提示，应支持`agent_scratchpad`作为其中的变量。要简单构造此提示，请使用`OpenAIFunctionsAgent.create_prompt(...)`。
- **class AimCallbackHandler(BaseMetadataCallbackHandler, BaseCallbackHandler)**：用于将日志记录到Aim的回调处理程序。参数：repo（可选）：Aim仓库路径或与Run对象绑定的Repo对象。如果省略，默认使用默认Repo。experiment_name（可选）：设置Run的“experiment”属性。默认为“default”。稍后可用于查询运行/序列。system_tracking_interval（可选）：设置系统使用度量的跟踪间隔（例如CPU、内存等）以秒为单位。设置为`None`以禁用系统度量跟踪。log_system_params（可选）：启用/禁用系统参数的日志记录，如安装的包、git信息、环境变量等。此处理程序将利用所关联的回调方法，调整每个回调函数的输入，并将响应记录到Aim。
- **class ArgillaCallbackHandler(BaseCallbackHandler)**：用于将日志记录到Argilla的回调处理程序。参数：dataset_name：Argilla中的`FeedbackDataset`的名称。请注意，它必须事先存在。如果需要有关如何在Argilla中创建`FeedbackDataset`的帮助，请访问https://docs.argilla.io/en/latest/guides/llms/practical_guides/use_argilla_callback_in_langchain.html。workspace_name：Argilla中指定的`FeedbackDataset`所在的工作区的名称。默认为`None`，这意味着将使用默认工作区。api_url：要使用的Argilla服务器的URL，以及其中的`FeedbackDataset`所在的位置。默认为`None`，这意味着将使用`ARGILLA_API_URL`环境变量或默认的http://localhost:6900。api_key：与Argilla服务器建立连接的API密钥。默认为`None`，这意味着将使用`ARGILLA_API_KEY`环境变量或默认的`argilla.apikey`。举例：from langchain.llms import OpenAI from langchain.callbacks import ArgillaCallbackHandler argilla_callback = ArgillaCallbackHandler(dataset_name="my-dataset", workspace_name="my-workspace", api_url="http://localhost:6900", api_key="argilla.apikey") llm = OpenAI(temperature=0, callbacks=[argilla_callback], verbose=True, openai_api_key="API_KEY_HERE") llm.generate(["What is the best NLP-annotation tool out there? (no bias at all)"]) "Argilla, no doubt about it."
- **class ArizeCallbackHandler(BaseCallbackHandler)**：用于将日志记录到Arize的回调处理程序。
- **class AsyncCallbackHandler(BaseCallbackHandler)**：可用于处理来自LangChain的异步回调的异步回调处理程序。
- **class ClearMLCallbackHandler(BaseMetadataCallbackHandler, BaseCallbackHandler)**：将日志记录到ClearML的回调处理程序。参数：job_type（str）：clearml任务类型，例如“inference”、“testing”或“qc”。project_name（str）：clearml项目名称。tags（list）：要添加到任务的标签。task_name（str）：clearml任务的名称。visualize（bool）：是否可视化运行。complexity_metrics（bool）：是否记录复杂度指标。stream_logs（bool）：是否将回调操作流式传输到ClearML。此处理程序将使用关联的回调方法，并使用关于LLM运行状态的元数据格式化每个回调函数的输入，并将响应添加到{method}_records和action的记录列表中。然后将响应记录到ClearML控制台。
- **class CometCallbackHandler(BaseMetadataCallbackHandler, BaseCallbackHandler)**：将日志记录到Comet的回调处理程序。参数：job_type（str）：comet_ml任务类型，例如“inference”、“testing”或“qc”。project_name（str）：comet_ml项目名称。tags（list）：要添加到任务的标签。task_name（str）：comet_ml任务的名称。visualize（bool）：是否可视化运行。complexity_metrics（bool）：是否记录复杂度指标。stream_logs（bool）：是否将回调操作流式传输到Comet。此处理程序将使用关联的回调方法，并使用关于LLM运行状态的元数据格式化每个回调函数的输入，并将响应添加到{method}_records和action的记录列表中。然后将响应记录到Comet。
- **class FileCallbackHandler(BaseCallbackHandler)**：将日志记录到文件的回调处理程序。
- **class HumanApprovalCallbackHandler(BaseCallbackHandler)**：手动验证值的回调处理程序。
- **class RunManager(BaseRunManager)**: 同步运行管理器。
- **class AsyncRunManager(BaseRunManager)**: 异步运行管理器。
- **class CallbackManager(BaseCallbackManager)**: 用于处理来自LangChain的回调的回调管理器。
- **class AsyncCallbackManager(BaseCallbackManager)**: 用于处理来自LangChain的异步回调的异步回调管理器。
- **class MlflowCallbackHandler(BaseMetadataCallbackHandler, BaseCallbackHandler)**: 将指标和数据记录到MLflow服务器的回调处理程序。参数：name（str）：运行的名称。experiment（str）：实验的名称。tags（str）：要附加到运行的标签。tracking_uri（str）：MLflow跟踪服务器URI。此处理程序将利用关联的回调方法，并使用关于LLM运行状态的元数据格式化每个回调函数的输入，并将响应添加到{method}_records和action的记录列表中。然后，将响应记录到MLflow服务器。
- **class OpenAICallbackHandler(BaseCallbackHandler)**: 跟踪OpenAI信息的回调处理程序。
- **class StdOutCallbackHandler(BaseCallbackHandler)**: 输出到标准输出的回调处理程序。
- **class StreamingStdOutCallbackHandler(BaseCallbackHandler)**: 用于流式传输的回调处理程序。仅适用于支持流式传输的LLM。
- **class StreamlitCallbackHandler(BaseCallbackHandler)**: 记录到Streamlit的回调处理程序。
- **class WandbCallbackHandler(BaseMetadataCallbackHandler, BaseCallbackHandler)**: 记录到Weights and Biases的回调处理程序。参数：job_type（str）：任务类型。project（str）：要记录到的项目。entity（str）：要记录到的实体。tags（list）：要记录的标签。group（str）：要记录到的分组。name（str）：运行的名称。notes（str）：要记录的注释。visualize（bool）：是否可视化运行。complexity_metrics（bool）：是否记录复杂度指标。stream_logs（bool）：是否将回调操作流式传输到Weights and Biases。此处理程序将利用关联的回调方法，并使用关于LLM运行状态的元数据格式化每个回调函数的输入，并将响应添加到{method}_records和action的记录列表中。然后，使用run.log()方法将响应记录到Weights and Biases。
- **class WhyLabsCallbackHandler(BaseCallbackHandler)**: WhyLabs的回调处理程序。
- **class BaseTracer(BaseCallbackHandler, ABC)**: 跟踪器的基本接口。
- **class LangChainTr    - **class LangChainTracer(BaseTracer)**: POST到langchain端点的SharedTracer的实现。
- **class LangChainTracerV1(BaseTracer)**: POST到langchain端点的SharedTracer的实现。
- **class RunCollectorCallbackHandler(BaseTracer)**: 收集所有嵌套运行的回调处理程序，将其存储在列表中。用于检查和评估。
- **class TracerSessionV1Base(BaseModel)**: TracerSessionV1的基类。
- **class BaseRun(BaseModel)**: Run的基类。
- **class LLMRun(BaseRun)**: LLMRun的类。
- **class ChainRun(BaseRun)**: ChainRun的类。
- **class ToolRun(BaseRun)**: ToolRun的类。
- **class Run(BaseRunV2)**: Tracer中V2 API的Run模式。
- **class BasePromptSelector(BaseModel, ABC)**: 获取语言模型的默认提示。
- **class ConditionalPromptSelector(BasePromptSelector)**: 经过条件处理的提示收集。
- **class APIRequesterOutputParser(BaseOutputParser)**: 解析请求和错误标签。
- **class APIResponderOutputParser(BaseOutputParser)**: 解析响应和错误标签。
- **class MapReduceDocumentsChain(BaseCombineDocumentsChain)**: 通过映射链条对文档进行组合，然后合并结果。
- **class MapRerankDocumentsChain(BaseCombineDocumentsChain)**: 通过映射链条对文档进行组合，然后对结果进行重新排序。
- **class RefineDocumentsChain(BaseCombineDocumentsChain)**: 通过首次处理然后在更多文档上进行优化来组合文档。
- **class StuffDocumentsChain(BaseCombineDocumentsChain)**: 通过将文档塞入上下文来组合的链。
- **class ConstitutionalPrinciple(BaseModel)**: 表示宪法原则的类。
- **class ConversationalRetrievalChain(BaseConversationalRetrievalChain)**: 用于与索引进行聊天的链。
- **class ChatVectorDBChain(BaseConversationalRetrievalChain)**: 用于与向量数据库进行聊天的链。
- **class FinishedOutputParser(BaseOutputParser[Tuple[str, bool]])**: 使用任何相关的上下文来回应用户信息。如果提供了上下文，你应该在那个上下文中给出你的回答。一旦你完成回应，返回FINISHED。
- **class BashOutputParser(BaseOutputParser)**: bash输出的解析器。
- **class FactWithEvidence(BaseModel)**: 表示单一声明的类。每个事实都有一个主体和一系列来源。如果有多个事实，确保将它们分开，这样每一个都只使用一组与之相关的来源。
- **class QuestionAnswer(BaseModel)**: 一个问题及其答案作为一系列事实，每个事实都应该有一个来源。每个句子包含一个主体和一系列来源。
- **class AnswerWithSources(BaseModel)**: 一个有来源的问题的答案。
- **class QAWithSourcesChain(BaseQAWithSourcesChain)**: 文档上的有来源的问题回答。
- **class RetrievalQAWithSourcesChain(BaseQAWithSourcesChain)**: 索引上的有来源的问题回答。
- **class VectorDBQAWithSourcesChain(BaseQAWithSourcesChain)**: 向量数据库上的有来源的问题回答。
- **class StructuredQueryOutputParser(BaseOutputParser[StructuredQuery])**: 解析字典到查询语言内部表示的函数。
- **class Expr(BaseModel)**: 一个过滤表达式。
- **class AttributeInfo(BaseModel)**: 关于数据源属性的信息。
- **class RetrievalQA(BaseRetrievalQA)**: 针对索引的问题回答链。
- **class VectorDBQA(BaseRetrievalQA)**: 针对向量数据库的问题回答链。
- **class RouterOutputParser(BaseOutputParser[Dict[str, str]])**: 多提示链中路由器链输出的解析器。
- **class ChatAnthropic(BaseChatModel, _AnthropicCommon)**: 作为Anthropic的大型语言模型的包装器。使用时，你应该已经安装了``anthropic`` Python包，并设置了环境变量``ANTHROPIC_API_KEY``，或者将其作为命名参数传递给构造函数。
- **class BaseChatModel(BaseLanguageModel, ABC)**: 是否打印出响应文本。
- **class SimpleChatModel(BaseChatModel)**: 更简单的接口。
- **class ChatGooglePalm(BaseChatModel, BaseModel)**: 作为Google的PaLM Chat API的包装器。使用它，你必须已经安装了google.generativeai Python包，并且设置了``GOOGLE_API_KEY``环境变量，或者通过ChatGoogle构造函数的google_api_key kwarg传递你的API密钥。
- **class ChatOpenAI(BaseChatModel)**: 作为OpenAI Chat大型语言模型的包装器。使用它，你应该已经安装了``openai`` Python包，并且设置了``OPENAI_API_KEY``环境变量。任何可以传递给openai.create调用的参数，即使这个类没有明确保存，也可以传递进去。
- **class AcreomLoader(BaseLoader)**: 用路径进行初始化。
- **class AirbyteJSONLoader(BaseLoader)**: 加载本地airbyte json文件的加载器。
- **class AirtableLoader(BaseLoader)**: 加载本地airbyte json文件的加载器。
- **class ApifyDatasetLoader(BaseLoader, BaseModel)**: 从Apify数据集加载文档的逻辑。
- **class ArxivLoader(BaseLoader)**: 将arxiv.org的查询结果加载到一个文档列表中。每个文档表示一个文档。加载器将原始PDF格式转换成文本。
- **class AzureBlobStorageContainerLoader(BaseLoader)**: 从Azure Blob Storage加载文档的逻辑。
- **class AzureBlobStorageFileLoader(BaseLoader)**: 从Azure Blob Storage加载文档的逻辑。
- **class BibtexLoader(BaseLoader)**: 将bibtex文件加载到一个文档列表中。每个文档表示bibtex文件中的一个条目。如果在`file` bibtex字段中存在PDF文件，原始PDF将被加载到文档文本中。如果没有这样的文件条目，将使用`abstract`字段。
- **class BigQueryLoader(BaseLoader)**: 将BigQuery的查询结果加载到一个文档列表中。每个文档代表结果的一行。`page_content_columns`被写入文档的`page_content`中。`metadata_columns`被写入文档的`metadata`中。默认情况下，所有列都被写入`page_content`，没有被写入`metadata`。
- **class BiliBiliLoader(BaseLoader)**: 加载bilibili的字幕的加载器。
- **class ChatGPTLoader(BaseLoader)**: 从导出的ChatGPT数据加载对话的加载器。
- **class CoNLLULoader(BaseLoader)**: 加载CoNLL-U文件。
- **class CSVLoader(BaseLoader)**: 将CSV文件加载到一个文档列表中。每个文档表示CSV文件的一行。每行被转换成一个键/值对，并输出到文档的page_content的新一行中。每个从csv加载的文档的源都设置为`file_path`参数的值。你可以通过设置`source_column`参数来覆盖这个，将其设置为CSV文件中的列名。每个文档的源将被设置为`source_column`中指定的列名的值。
- **class DataFrameLoader(BaseLoader)**: 加载Pandas DataFrames。
- **class DiffbotLoader(BaseLoader)**: 加载Diffbot文件json的加载器。
- **class DirectoryLoader(BaseLoader)**: 从目录加载文档的逻辑。
- **class DiscordChatLoader(BaseLoader)**: 加载Discord聊天日志。
- **class DocugamiLoader(BaseLoader, BaseModel)**: 加载来自Docugami的处理过的文档的加载器。使用时，你应该已经安装了``lxml`` Python包。
以下是我根据您给出的格式进行翻译和排列的类描述：

- **class DuckDBLoader(BaseLoader)**: 加载 DuckDB 查询结果到一系列文档中。每个文档代表结果中的一行。`page_content_columns` 写入文档的 `page_content`。`metadata_columns` 写入文档的 `metadata`。默认情况下，所有列都写入 `page_content`，无写入 `metadata`。

- **class OutlookMessageLoader(BaseLoader)**: 使用 extract_msg 加载 Outlook Message 文件的加载器。详情请参见：https://github.com/TeamMsgExtractor/msg-extractor。

- **class BaseEmbaasLoader(BaseModel)**: embaas 文档提取 API 的 URL。

- **class EmbaasBlobLoader(BaseEmbaasLoader, BaseBlobParser)**: embaas 文档字节加载服务的封装。要使用，你应该有环境变量 `EMBAAS_API_KEY` 设为你的 API 密钥，或作为命名参数传递给构造器。例子见原文。

- **class EmbaasLoader(BaseEmbaasLoader, BaseLoader)**: embaas 文档加载服务的封装。要使用，你应该有环境变量 `EMBAAS_API_KEY` 设为你的 API 密钥，或作为命名参数传递给构造器。例子见原文。

- **class EverNoteLoader(BaseLoader)**: EverNote 加载器。加载 EverNote 笔记本导出文件（例如：my_notebook.enex）到 Documents 中。产生这种文件的指导可以在这里找到：https://help.evernote.com/hc/en-us/articles/209005557-Export-notes-and-notebooks-as-ENEX-or-HTML。详细参数和描述见原文。

- **class FacebookChatLoader(BaseLoader)**: 加载 Facebook 消息 JSON 目录导出的加载器。

- **class FaunaLoader(BaseLoader)**: 具有特定属性的加载器，这些属性包括 FQL 查询字符串，每个页面的内容字段，用于认证 FaunaDB 的密钥，以及包含在元数据中的字段名列表。

- **class FigmaFileLoader(BaseLoader)**: 加载 Figma 文件 JSON 的加载器。

- **class GCSDirectoryLoader(BaseLoader)**: 从 GCS 加载文档的逻辑。

- **class GCSFileLoader(BaseLoader)**: 从 GCS 加载文档的逻辑。

- **class GenericLoader(BaseLoader)**: 通用文档加载器。允许将任意 blob 加载器与 blob 解析器组合在一起。具体示例见原文。

- **class GitLoader(BaseLoader)**: 将 Git 仓库的文件加载到一系列文档中。仓库可以是本地磁盘上的 `repo_path`，也可以是 `clone_url` 的远程地址，该地址将被克隆到 `repo_path`。目前仅支持文本文件。详细参数和描述见原文。

- **class BaseGitHubLoader(BaseLoader, BaseModel, ABC)**: 加载 GitHub 仓库的问题。

- **class GitHubIssuesLoader(BaseGitHubLoader)**: 如果为 True，将结果中包含 Pull Requests，否则忽略它们。

- **class GoogleDriveLoader(BaseLoader, BaseModel)**: 从 Google Drive 加载 Google Docs 的加载器。

- **class GutenbergLoader(BaseLoader)**: 使用 urllib 加载 .txt 网络文件的加载器。

- **class BSHTMLLoader(BaseLoader)**: 使用 beautiful soup 解析 HTML 文件的加载器。

- **class HuggingFaceDatasetLoader(BaseLoader)**: 从 Hugging Face Hub 加载文档的逻辑。

- **class IFixitLoader(BaseLoader)**: 加载 iFixit 修理指南，设备 wiki 和答案。详情见原文。

- **class ImageCaptionLoader(BaseLoader)**: 加载图像标题的加载器。

- **class JoplinLoader(BaseLoader)**: 从 Joplin 获取笔记的加载器。要使用此加载器，你需要有 Joplin 运行，并启用 Web Clipper（在应用设置中查找 "Web Clipper"）。详细使用和获取 access token 的方法见原文。

- **class JSONLoader(BaseLoader)**: 加载 JSON 文件，并引用提供的 jq schema 将文本加载到文档中。具体示例见原文。

- **class MastodonTootsLoader(BaseLoader)**: Mastodon toots 加载器。

- **class MaxComputeLoader(BaseLoader)**: 将阿里云 MaxCompute 表的查询结果加载到文档中。

- **class MWDumpLoader(BaseLoader)**: 从 XML 文件中加载 MediaWiki dump。

"class NotebookLoader(BaseLoader) is Loader that loads .ipynb notebook files.",
"class NotionDirectoryLoader(BaseLoader) is Loader that loads Notion directory dump.",
"class NotionDBLoader(BaseLoader) is Notion DB Loader.  Reads content from pages within a Noton Database.  Args:   integration_token (str)**: Notion integration token.   database_id (str)**: Notion database id.   request_timeout_sec (int)**: Timeout for Notion requests in seconds. ",
"class ObsidianLoader(BaseLoader) is Loader that loads Obsidian files from disk.",
"class _OneDriveSettings(BaseSettings) is    Authenticates the OneDrive API client using the specified   authentication method and returns the Account object.    Returns:    Type[Account]: The authenticated Account object.  ",
"class OneDriveFileLoader(BaseLoader, BaseModel) is Load Documents",
"class BasePDFLoader(BaseLoader, ABC) is Base loader class for PDF files.   Defaults to check for local file, but if the file is a web path, it will download it  to a temporary file, and use that, then clean up the temporary file after completion ",
"class OnlinePDFLoader(BasePDFLoader) is Loader that loads online PDFs.",
"class PyPDFLoader(BasePDFLoader) is Loads a PDF with pypdf and chunks at character level.   Loader also stores page numbers in metadatas. ",
"class PyPDFium2Loader(BasePDFLoader) is Loads a PDF with pypdfium2 and chunks at character level.",
"class PyPDFDirectoryLoader(BaseLoader) is Loads a directory with PDF files with pypdf and chunks at character level.   Loader also stores page numbers in metadatas. ",
"class PDFMinerLoader(BasePDFLoader) is Loader that uses PDFMiner to load PDF files.",
"class PDFMinerPDFasHTMLLoader(BasePDFLoader) is Loader that uses PDFMiner to load PDF files as HTML content.",
"class PyMuPDFLoader(BasePDFLoader) is Loader that uses PyMuPDF to load PDF files.",
"class MathpixPDFLoader(BasePDFLoader) is Loader that uses pdfplumber to load PDF files.",
"class PsychicLoader(BaseLoader) is Loader that loads documents from Psychic.dev.",
"class PySparkDataFrameLoader(BaseLoader) is Load PySpark DataFrames",
"class ReadTheDocsLoader(BaseLoader) is Loader that loads ReadTheDocs documentation directory dump.",
"class RedditPostsLoader(BaseLoader) is Reddit posts loader.  Read posts on a subreddit.  First you need to go to  https://www.reddit.com/prefs/apps/  and create your application ",
"class RoamLoader(BaseLoader) is Loader that loads Roam files from disk.",
"class S3DirectoryLoader(BaseLoader) is Loading logic for loading documents from s3.",
"class S3FileLoader(BaseLoader) is Loading logic for loading documents from s3.",
"class SlackDirectoryLoader(BaseLoader) is Loader for loading documents from a Slack directory dump.",
"class SnowflakeLoader(BaseLoader) is Loads a query result from Snowflake into a list of documents.   Each document represents one row of the result. The `page_content_columns`  are written into the `page_content` of the document. The `metadata_columns`  are written into the `metadata` of the document. By default, all columns  are written into the `page_content` and none into the `metadata`.  ",
"class SRTLoader(BaseLoader) is Loader for .srt (subtitle) files.",
"class TelegramChatFileLoader(BaseLoader) is Loader that loads Telegram chat json directory dump.",
"class TelegramChatApiLoader(BaseLoader) is Loader that loads Telegram chat json directory dump.",
"class TextLoader(BaseLoader) is Load text files.    Args:   file_path: Path to the file to load.    encoding: File encoding to use. If `None`, the file will be loaded   with the default system encoding.    autodetect_encoding: Whether to try to autodetect the file encoding    if the specified encoding fails. ",
"class ToMarkdownLoader(BaseLoader) is Loader that loads HTML to markdown using 2markdown.",
"class TomlLoader(BaseLoader) is   A TOML document loader that inherits from the BaseLoader class.   This class can be initialized with either a single source file or a source  directory containing TOML files. ",
"class TrelloLoader(BaseLoader) is Trello loader. Reads all cards from a Trello board.",
"class TwitterTweetLoader(BaseLoader) is Twitter tweets loader.  Read tweets of user twitter handle.   First you need to go to  `https://developer.twitter.com/en/docs/twitter-api  /getting-started/getting-access-to-the-twitter-api`  to get your token. And create a v2 version of the app. ",
"class UnstructuredBaseLoader(BaseLoader, ABC) is Loader that uses unstructured to load files.",
"class UnstructuredURLLoader(BaseLoader) is Loader that uses unstructured to load HTML files.",
"class PlaywrightURLLoader(BaseLoader) is Loader that uses Playwright and to load a page and unstructured to load the html.  This is useful for loading pages that require javascript to render.   Attributes:   urls (List[str])**: List of URLs to load.   continue_on_failure (bool)**: If True, continue loading other URLs on failure.   headless (bool)**: If True, the browser will run in headless mode. ",
"class SeleniumURLLoader(BaseLoader) is Loader that uses Selenium and to load a page and unstructured to load the html.  This is useful for loading pages that require javascript to render.   Attributes:   urls (List[str])**: List of URLs to load.   continue_on_failure (bool)**: If True, continue loading other URLs on failure.   browser (str)**: The browser to use, either 'chrome' or 'firefox'.   binary_location (Optional[str])**: The location of the browser binary.   executable_path (Optional[str])**: The path to the browser executable.   headless (bool)**: If True, the browser will run in headless mode.   arguments [List[str]]: List of arguments to pass to the browser. ",
"class WeatherDataLoader(BaseLoader) is Weather Reader.   Reads the forecast & current weather of any location using OpenWeatherMap's free  API. Checkout 'https://openweathermap.org/appid' for more on how to generate a free  OpenWeatherMap API. ",
"class WebBaseLoader(BaseLoader) is Loader that uses urllib and beautiful soup to load webpages.",
"class WhatsAppChatLoader(BaseLoader) is Loader that loads WhatsApp messages text file.",
"class WikipediaLoader(BaseLoader) is Loads a query result from www.wikipedia.org into a list of Documents.  The hard limit on the number of downloaded Documents is 300 for now.   Each wiki page represents one Document. ",
"class Docx2txtLoader(BaseLoader, ABC) is Loads a DOCX with docx2txt and chunks at character level.   Defaults to check for local file, but if the file is a web path, it will download it  to a temporary file, and use that, then clean up the temporary file after completion ",
"class YoutubeLoader(BaseLoader) is Loader that loads Youtube transcripts.",
"class GoogleApiYoutubeLoader(BaseLoader) is Loader that loads all Videos from a Channel   To use, you should have the ``googleapiclient,youtube_transcript_api``  python package installed.  As the service needs a google_api_client, you first have to initialize  the GoogleApiClient.   Additionally you have to either provide a channel name or a list of videoids  \"https://developers.google.com/docs/api/quickstart/python\"  Example:   .. code-block:: python     from langchain.document_loaders import GoogleApiClient    from langchain.document_loaders import GoogleApiYoutubeLoader    google_api_client = GoogleApiClient(     service_account_path=Path(\"path_to_your_sec_file.json\")    )    loader = GoogleApiYoutubeLoader(     google_api_client=google_api_client,     channel_name = \"CodeAesthetic\"    )    load.load()  ",
"class Blob(BaseModel) is A blob is used to represent raw data by either reference or value.   Provides an interface to materialize the blob in different representations, and  help to decouple the development of data loaders from the downstream parsing of  the raw data.   Inspired by: https://developer.mozilla.org/en-US/docs/Web/API/Blob ",
"class OpenAIWhisperParser(BaseBlobParser) is Transcribe and parse audio files.  Audio transcription is with OpenAI Whisper model.",
"class MimeTypeBasedParser(BaseBlobParser) is A parser that uses mime-types to determine how to parse a blob.   This parser is useful for simple pipelines where the mime-type is sufficient  to determine how to parse a blob.   To use, configure handlers based on mime-types and pass them to the initializer.   Example:    .. code-block:: python    from langchain.document_loaders.parsers.generic import MimeTypeBasedParser    parser = MimeTypeBasedParser(    handlers={     \"application/pdf\": ...,    },    fallback_parser=...,   ) ",
"class PyPDFParser(BaseBlobParser) is Loads a PDF with pypdf and chunks at character level.",
"class PDFMinerParser(BaseBlobParser) is Parse PDFs with PDFMiner.",
"class PyMuPDFParser(BaseBlobParser) is Parse PDFs with PyMuPDF.",
"class PyPDFium2Parser(BaseBlobParser) is Parse PDFs with PyPDFium2.",
"class PDFPlumberParser(BaseBlobParser) is Parse PDFs with PDFPlumber.",
"class TextParser(BaseBlobParser) is Lazily parse the blob.",
"class BS4HTMLParser(BaseBlobParser) is Parser that uses beautiful soup to parse HTML files.",
"class AlephAlphaAsymmetricSemanticEmbedding(BaseModel, Embeddings) is   Wrapper for Aleph Alpha's Asymmetric Embeddings  AA provides you with an endpoint to embed a document and a query.  The models were optimized to make the embeddings of documents and  the query for a document as similar as possible.  To learn more, check out: https://docs.aleph-alpha.com/docs/tasks/semantic_embed/   Example:   .. code-block:: python     from aleph_alpha import AlephAlphaAsymmetricSemanticEmbedding     embeddings = AlephAlphaSymmetricSemanticEmbedding()     document = \"This is a content of the document\"    query = \"What is the content of the document?\"     doc_result = embeddings.embed_documents([document])    query_result = embeddings.embed_query(query)  ",
"class BedrockEmbeddings(BaseModel, Embeddings) is Embeddings provider to invoke Bedrock embedding models.   To authenticate, the AWS client uses the following methods to  automatically load credentials:  https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html   If a specific credential profile should be used, you must pass  the name of the profile from the ~/.aws/credentials file that is to be used.   Make sure the credentials / roles used have the required policies to  access the Bedrock service. ",
"class CohereEmbeddings(BaseModel, Embeddings) is Wrapper around Cohere embedding models.   To use, you should have the ``cohere`` python package installed, and the  environment variable ``COHERE_API_KEY`` set with your API key or pass it  as a named parameter to the constructor.   Example:   .. code-block:: python     from langchain.embeddings import CohereEmbeddings    cohere = CohereEmbeddings(     model=\"embed-english-light-v2.0\", cohere_api_key=\"my-api-key\"    ) ",
"class DashScopeEmbeddings(BaseModel, Embeddings) is Wrapper around DashScope embedding models.   To use, you should have the ``dashscope`` python package installed, and the  environment variable ``DASHSCOPE_API_KEY`` set with your API key or pass it  as a named parameter to the constructor.   Example:   .. code-block:: python     from langchain.embeddings import DashScopeEmbeddings    embeddings = DashScopeEmbeddings(dashscope_api_key=\"my-api-key\")   Example:   .. code-block:: python     import os    os.environ[\"DASHSCOPE_API_KEY\"] = \"your DashScope API KEY\"     from langchain.embeddings.dashscope import DashScopeEmbeddings    embeddings = DashScopeEmbeddings(     model=\"text-embedding-v1\",    )    text = \"This is a test query.\"    query_result = embeddings.embed_query(text)  ",
"class DeepInfraEmbeddings(BaseModel, Embeddings) is Wrapper around Deep Infra's embedding inference service.   To use, you should have the  environment variable ``DEEPINFRA_API_TOKEN`` set with your API token, or pass  it as a named parameter to the constructor.  There are multiple embeddings models available,  see https://deepinfra.com/models?type=embeddings.   Example:   .. code-block:: python     from langchain.embeddings import DeepInfraEmbeddings    deepinfra_emb = DeepInfraEmbeddings(     model_id=\"sentence-transformers/clip-ViT-B-32\",     deepinfra_api_token=\"my-api-key\"    )    r1 = deepinfra_emb.embed_documents(     [      \"Alpha is the first letter of Greek alphabet\",      \"Beta is the second letter of Greek alphabet\",     ]    )    r2 = deepinfra_emb.embed_query(     \"What is the second letter of Greek alphabet\"    )  ",
"class EmbaasEmbeddings(BaseModel, Embeddings) is Wrapper around embaas's embedding service.   To use, you should have the  environment variable ``EMBAAS_API_KEY`` set with your API key, or pass  it as a named parameter to the constructor.   Example:   .. code-block:: python     # Initialise with default model and instruction    from langchain.embeddings import EmbaasEmbeddings    emb = EmbaasEmbeddings()     # Initialise with custom model and instruction    from langchain.embeddings import EmbaasEmbeddings    emb_model = \"instructor-large\"    emb_inst = \"Represent the Wikipedia document for retrieval\"    emb = EmbaasEmbeddings(     model=emb_model,     instruction=emb_inst    ) ",
"class GooglePalmEmbeddings(BaseModel, Embeddings) is Model name to use.",
"class HuggingFaceEmbeddings(BaseModel, Embeddings) is Wrapper around sentence_transformers embedding models.   To use, you should have the ``sentence_transformers`` python package installed.   Example:   .. code-block:: python     from langchain.embeddings import HuggingFaceEmbeddings     model_name = \"sentence-transformers/all-mpnet-base-v2\"    model_kwargs = {'device': 'cpu'}    encode_kwargs = {'normalize_embeddings': False}    hf = HuggingFaceEmbeddings(     model_name=model_name,     model_kwargs=model_kwargs,     encode_kwargs=encode_kwargs    ) ",
"class HuggingFaceInstructEmbeddings(BaseModel, Embeddings) is Wrapper around sentence_transformers embedding models.   To use, you should have the ``sentence_transformers``  and ``InstructorEmbedding`` python packages installed.   Example:   .. code-block:: python     from langchain.embeddings import HuggingFaceInstructEmbeddings     model_name = \"hkunlp/instructor-large\"    model_kwargs = {'device': 'cpu'}    encode_kwargs = {'normalize_embeddings': True}    hf = HuggingFaceInstructEmbeddings(     model_name=model_name,     model_kwargs=model_kwargs,     encode_kwargs=encode_kwargs    ) ",
"class HuggingFaceHubEmbeddings(BaseModel, Embeddings) is Wrapper around HuggingFaceHub embedding models.   To use, you should have the ``huggingface_hub`` python package installed, and the  environment variable ``HUGGINGFACEHUB_API_TOKEN`` set with your API token, or pass  it as a named parameter to the constructor.   Example:   .. code-block:: python     from langchain.embeddings import HuggingFaceHubEmbeddings    repo_id = \"sentence-transformers/all-mpnet-base-v2\"    hf = HuggingFaceHubEmbeddings(     repo_id=repo_id,     task=\"feature-extraction\",     huggingfacehub_api_token=\"my-api-key\",    ) ",
"class JinaEmbeddings(BaseModel, Embeddings) is Model name to use.",
"class LlamaCppEmbeddings(BaseModel, Embeddings) is Wrapper around llama.cpp embedding models.   To use, you should have the llama-cpp-python library installed, and provide the  path to the Llama model as a named parameter to the constructor.  Check out: https://github.com/abetlen/llama-cpp-python   Example:   .. code-block:: python     from langchain.embeddings import LlamaCppEmbeddings    llama = LlamaCppEmbeddings(model_path=\"/path/to/model.bin\") ",
"class MiniMaxEmbeddings(BaseModel, Embeddings) is Wrapper around MiniMax's embedding inference service.   To use, you should have the environment variable ``MINIMAX_GROUP_ID`` and  ``MINIMAX_API_KEY`` set with your API token, or pass it as a named parameter to  the constructor.   Example:   .. code-block:: python     from langchain.embeddings import MiniMaxEmbeddings    embeddings = MiniMaxEmbeddings()     query_text = \"This is a test query.\"    query_result = embeddings.embed_query(query_text)     document_text = \"This is a test document.\"    document_result = embeddings.embed_documents([document_text])  ",
"class ModelScopeEmbeddings(BaseModel, Embeddings) is Wrapper around modelscope_hub embedding models.   To use, you should have the ``modelscope`` python package installed.   Example:   .. code-block:: python     from langchain.embeddings import ModelScopeEmbeddings    model_id = \"damo/nlp_corom_sentence-embedding_english-base\"    embed = ModelScopeEmbeddings(model_id=model_id) ",
"class MosaicMLInstructorEmbeddings(BaseModel, Embeddings) is Wrapper around MosaicML's embedding inference service.   To use, you should have the  environment variable ``MOSAICML_API_TOKEN`` set with your API token, or pass  it as a named parameter to the constructor.   Example:   .. code-block:: python     from langchain.llms import MosaicMLInstructorEmbeddings    endpoint_url = (     \"https://models.hosted-on.mosaicml.hosting/instructor-large/v1/predict\"    )    mosaic_llm = MosaicMLInstructorEmbeddings(     endpoint_url=endpoint_url,     mosaicml_api_token=\"my-api-key\"    ) ",
"class OpenAIEmbeddings(BaseModel, Embeddings) is Wrapper around OpenAI embedding models.   To use, you should have the ``openai`` python package installed, and the  environment variable ``OPENAI_API_KEY`` set with your API key or pass it  as a named parameter to the constructor.   Example:   .. code-block:: python     from langchain.embeddings import OpenAIEmbeddings    openai = OpenAIEmbeddings(openai_api_key=\"my-api-key\")   In order to use the library with Microsoft Azure endpoints, you need to set  the OPENAI_API_TYPE, OPENAI_API_BASE, OPENAI_API_KEY and OPENAI_API_VERSION.  The OPENAI_API_TYPE must be set to 'azure' and the others correspond to  the properties of your endpoint.  In addition, the deployment name must be passed as the model parameter.   Example:   .. code-block:: python     import os    os.environ[\"OPENAI_API_TYPE\"] = \"azure\"    os.environ[\"OPENAI_API_BASE\"] = \"https://<your-endpoint.openai.azure.com/\"    os.environ[\"OPENAI_API_KEY\"] = \"your AzureOpenAI key\"    os.environ[\"OPENAI_API_VERSION\"] = \"2023-03-15-preview\"    os.environ[\"OPENAI_PROXY\"] = \"http://your-corporate-proxy:8080\"     from langchain.embeddings.openai import OpenAIEmbeddings    embeddings = OpenAIEmbeddings(     deployment=\"your-embeddings-deployment-name\",     model=\"your-embeddings-model-name\",     openai_api_base=\"https://your-endpoint.openai.azure.com/\",     openai_api_type=\"azure\",    )    text = \"This is a test query.\"    query_result = embeddings.embed_query(text)  ",
"class SagemakerEndpointEmbeddings(BaseModel, Embeddings) is Wrapper around custom Sagemaker Inference Endpoints.   To use, you must supply the endpoint name from your deployed  Sagemaker model & the region where it is deployed.   To authenticate, the AWS client uses the following methods to  automatically load credentials:  https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html   If a specific credential profile should be used, you must pass  the name of the profile from the ~/.aws/credentials file that is to be used.   Make sure the credentials / roles used have the required policies to  access the Sagemaker endpoint.  See: https://docs.aws.amazon.com/IAM/latest/UserGuide/access_policies.html ",
"class TensorflowHubEmbeddings(BaseModel, Embeddings) is Wrapper around tensorflow_hub embedding models.   To use, you should have the ``tensorflow_text`` python package installed.   Example:   .. code-block:: python     from langchain.embeddings import TensorflowHubEmbeddings    url = \"https://tfhub.dev/google/universal-sentence-encoder-multilingual/3\"    tf = TensorflowHubEmbeddings(model_url=url) ",
"class RunEvaluatorOutputParser(BaseOutputParser[EvaluationResult]) is Parse the output of a run.",
"class AutoGPTMemory(BaseChatMemory) is VectorStoreRetriever object to connect to.",
"class GenerativeAgent(BaseModel) is A character with memory and innate characteristics.",
"class GenerativeAgentMemory(BaseMemory) is The core language model.",
"class Step(BaseModel) is Add step and step response to the container.",
"class ListStepContainer(BaseModel) is Parse into a plan.",
"class BaseExecutor(BaseModel) is Take step.",
"class ChainExecutor(BaseExecutor) is Take step.",
"class BasePlanner(BaseModel) is Given input, decided what to do.",
"class LLMPlanner(BasePlanner) is Given input, decided what to do.",
"class GraphIndexCreator(BaseModel) is Functionality to create graph index.",
"class VectorStoreIndexWrapper(BaseModel) is Wrapper around a vectorstore for easy access.",
"class VectorstoreIndexCreator(BaseModel) is Logic for creating indexes.",
"class AI21PenaltyData(BaseModel) is Parameters for AI21 penalty data.",
"class _AnthropicCommon(BaseModel) is Model name to use.",
"class BaseLLM(BaseLanguageModel, ABC) is LLM wrapper should take in a prompt and return a string.",
"class LLM(BaseLLM) is LLM class that expect subclasses to implement a simpler call method.   The purpose of this class is to expose a simpler interface for working  with LLMs, rather than expect the user to implement the full _generate method. ",
"class _DatabricksClientBase(BaseModel, ABC) is A base JSON API client that talks to Databricks.",
"class GooglePalm(BaseLLM, BaseModel) is Model name to use.",
"class BaseOpenAI(BaseLLM) is Wrapper around OpenAI large language models.",
"class OpenAI(BaseOpenAI) is Wrapper around OpenAI large language models.   To use, you should have the ``openai`` python package installed, and the  environment variable ``OPENAI_API_KEY`` set with your API key.   Any parameters that are valid to be passed to the openai.create call can be passed  in, even if not explicitly saved on this class.   Example:   .. code-block:: python     from langchain.llms import OpenAI    openai = OpenAI(model_name=\"text-davinci-003\") ",
"class AzureOpenAI(BaseOpenAI) is Wrapper around Azure-specific OpenAI large language models.   To use, you should have the ``openai`` python package installed, and the  environment variable ``OPENAI_API_KEY`` set with your API key.   Any parameters that are valid to be passed to the openai.create call can be passed  in, even if not explicitly saved on this class.   Example:   .. code-block:: python     from langchain.llms import AzureOpenAI    openai = AzureOpenAI(model_name=\"text-davinci-003\") ",
"class OpenAIChat(BaseLLM) is Wrapper around OpenAI Chat large language models.   To use, you should have the ``openai`` python package installed, and the  environment variable ``OPENAI_API_KEY`` set with your API key.   Any parameters that are valid to be passed to the openai.create call can be passed  in, even if not explicitly saved on this class.   Example:   .. code-block:: python     from langchain.llms import OpenAIChat    openaichat = OpenAIChat(model_name=\"gpt-3.5-turbo\") ",
"class _VertexAICommon(BaseModel) is Wrapper around Google Vertex AI large language models.",
"class ConversationBufferMemory(BaseChatMemory) is Buffer for storing conversation memory.",
"class ConversationStringBufferMemory(BaseMemory) is Buffer for storing conversation memory.",
"class ConversationBufferWindowMemory(BaseChatMemory) is Buffer for storing conversation memory.",
"class BaseChatMemory(BaseMemory, ABC) is Save context from this conversation to buffer.",
"class CombinedMemory(BaseMemory) is Class for combining multiple memories' data together.",
"class BaseEntityStore(BaseModel, ABC) is Get entity value from store.",
"class InMemoryEntityStore(BaseEntityStore) is Basic in-memory entity store.",
"class RedisEntityStore(BaseEntityStore) is Redis-backed Entity store. Entities get a TTL of 1 day by default, and  that TTL is extended by 3 days every time the entity is read back. ",
"class SQLiteEntityStore(BaseEntityStore) is SQLite-backed Entity store",
"class ConversationEntityMemory(BaseChatMemory) is Entity extractor & summarizer memory.   Extracts named entities from the recent chat history and generates summaries.  With a swapable entity store, persisting entities across conversations.  Defaults to an in-memory entity store, and can be swapped out for a Redis,  SQLite, or other entity store. ",
"class ConversationKGMemory(BaseChatMemory) is Knowledge graph memory for storing conversation memory.   Integrates with external knowledge graph to store and retrieve  information about knowledge triples in the conversation. ",
"class MotorheadMemory(BaseChatMemory) is      You must provide an API key or a client ID to use the managed     version of Motorhead. Visit https://getmetal.io for more information.    ",
"class ReadOnlySharedMemory(BaseMemory) is A memory wrapper that is read-only and cannot be changed.",
"class SimpleMemory(BaseMemory) is Simple memory for storing context or other bits of information that shouldn't  ever change between prompts. ",
"class SummarizerMixin(BaseModel) is Conversation summarizer to memory.",
"class ConversationSummaryBufferMemory(BaseChatMemory, SummarizerMixin) is Buffer with summarizer for storing conversation memory.",
"class ConversationTokenBufferMemory(BaseChatMemory) is Buffer for storing conversation memory.",
"class VectorStoreRetrieverMemory(BaseMemory) is Class for a VectorStore-backed memory object.",
"class CassandraChatMessageHistory(BaseChatMessageHistory) is Chat message history that stores history in Cassandra.   Args:   contact_points: list of ips to connect to Cassandra cluster   session_id: arbitrary key that is used to store the messages    of a single chat session.   port: port to connect to Cassandra cluster   username: username to connect to Cassandra cluster   password: password to connect to Cassandra cluster   keyspace_name: name of the keyspace to use   table_name: name of the table to use ",
"class CosmosDBChatMessageHistory(BaseChatMessageHistory) is Chat history backed by Azure CosmosDB.",
"class DynamoDBChatMessageHistory(BaseChatMessageHistory) is Chat message history that stores history in AWS DynamoDB.  This class expects that a DynamoDB table with name `table_name`  and a partition Key of `SessionId` is present.   Args:   table_name: name of the DynamoDB table   session_id: arbitrary key that is used to store the messages    of a single chat session.   endpoint_url: URL of the AWS endpoint to connect to. This argument    is optional and useful for test purposes, like using Localstack.    If you plan to use AWS cloud service, you normally don't have to    worry about setting the endpoint_url. ",
"class FirestoreChatMessageHistory(BaseChatMessageHistory) is Chat history backed by Google Firestore.",
"class ChatMessageHistory(BaseChatMessageHistory, BaseModel) is Add a self-created message to the store",
"class MomentoChatMessageHistory(BaseChatMessageHistory) is Chat message history cache that uses Momento as a backend.  See https://gomomento.com/",
"class MongoDBChatMessageHistory(BaseChatMessageHistory) is Chat message history that stores history in MongoDB.   Args:   connection_string: connection string to connect to MongoDB   session_id: arbitrary key that is used to store the messages    of a single chat session.   database_name: name of the database to use   collection_name: name of the collection to use ",
"class PostgresChatMessageHistory(BaseChatMessageHistory) is CREATE TABLE IF NOT EXISTS {self.table_name} (    id SERIAL PRIMARY KEY,    session_id TEXT NOT NULL,    message JSONB NOT NULL   );",
"class RedisChatMessageHistory(BaseChatMessageHistory) is Construct the record key to use",
"class ZepChatMessageHistory(BaseChatMessageHistory) is A ChatMessageHistory implementation that uses Zep as a backend.   Recommended usage::    # Set up Zep Chat History   zep_chat_history = ZepChatMessageHistory(    session_id=session_id,    url=ZEP_API_URL,   )    # Use a standard ConversationBufferMemory to encapsulate the Zep chat history   memory = ConversationBufferMemory(    memory_key=\"chat_history\", chat_memory=zep_chat_history   )    Zep provides long-term conversation storage for LLM apps. The server stores,  summarizes, embeds, indexes, and enriches conversational AI chat  histories, and exposes them via simple, low-latency APIs.   For server installation instructions and more, see: https://getzep.github.io/   This class is a thin wrapper around the zep-python package. Additional  Zep functionality is exposed via the `zep_summary` and `zep_messages`  properties.   For more information on the zep-python package, see:  https://github.com/getzep/zep-python ",
"class BooleanOutputParser(BaseOutputParser[bool]) is Parse the output of an LLM call to a boolean.    Args:    text: output of language model    Returns:    boolean   ",
"class CombiningOutputParser(BaseOutputParser) is Class to combine multiple output parsers into one.",
"class DatetimeOutputParser(BaseOutputParser[datetime]) is Write a datetime string that matches the     following pattern: \"{self.format}\". Examples: {examples}",
"class OutputFixingParser(BaseOutputParser[T]) is Wraps a parser and tries to fix parsing errors.",
"class ListOutputParser(BaseOutputParser) is Class to parse the output of an LLM call to a list.",
"class RegexParser(BaseOutputParser) is Class to parse the output into a dictionary.",
"class RegexDictParser(BaseOutputParser) is Class to parse the output into a dictionary.",
"class RetryOutputParser(BaseOutputParser[T]) is Wraps a parser and tries to fix parsing errors.   Does this by passing the original prompt and the completion to another  LLM, and telling it the completion did not satisfy criteria in the prompt. ",
"class RetryWithErrorOutputParser(BaseOutputParser[T]) is Wraps a parser and tries to fix parsing errors.   Does this by passing the original prompt, the completion, AND the error  that was raised to another language model and telling it that the completion  did not work, and raised the given error. Differs from RetryOutputParser  in that this implementation provides the error that was raised back to the  LLM, which in theory should give it more information on how to fix it. ",
"class StringPromptTemplate(BasePromptTemplate, ABC) is String prompt should expose the format method, returning a prompt.",
"class MessagesPlaceholder(BaseMessagePromptTemplate) is Prompt template that assumes variable is already list of messages.",
"class BaseStringMessagePromptTemplate(BaseMessagePromptTemplate, ABC) is To a BaseMessage.",
"class ChatMessagePromptTemplate(BaseStringMessagePromptTemplate) is Return prompt as string.",
"class BaseChatPromptTemplate(BasePromptTemplate, ABC) is Format kwargs into a list of messages.",
"class PipelinePromptTemplate(BasePromptTemplate) is A prompt template for composing multiple prompts together.   This can be useful when you want to reuse parts of prompts.  A PipelinePrompt consists of two main parts:   - final_prompt: This is the final prompt that is returned   - pipeline_prompts: This is a list of tuples, consisting    of a string (`name`) and a Prompt Template.    Each PromptTemplate will be formatted and then passed    to future prompt templates as a variable with    the same name as `name` ",
"class LengthBasedExampleSelector(BaseExampleSelector, BaseModel) is Select examples based on length.",
"class NGramOverlapExampleSelector(BaseExampleSelector, BaseModel) is Select and order examples based on ngram overlap score (sentence_bleu score).   https://www.nltk.org/_modules/nltk/translate/bleu_score.html  https://aclanthology.org/P02-1040.pdf ",
"class SemanticSimilarityExampleSelector(BaseExampleSelector, BaseModel) is Example selector that selects examples based on SemanticSimilarity.",
"class ArxivRetriever(BaseRetriever, ArxivAPIWrapper) is   It is effectively a wrapper for ArxivAPIWrapper.  It wraps load() to get_relevant_documents().  It uses all ArxivAPIWrapper arguments without any change. ",
"class AwsKendraIndexRetriever(BaseRetriever) is Wrapper around AWS Kendra.",
"class AzureCognitiveSearchRetriever(BaseRetriever, BaseModel) is Wrapper around Azure Cognitive Search.",
"class ChatGPTPluginRetriever(BaseRetriever, BaseModel) is Configuration for this pydantic object.",
"class ContextualCompressionRetriever(BaseRetriever, BaseModel) is Retriever that wraps a base retriever and compresses the results.",
"class ElasticSearchBM25Retriever(BaseRetriever) is Wrapper around Elasticsearch using BM25 as a retrieval method.    To connect to an Elasticsearch instance that requires login credentials,  including Elastic Cloud, use the Elasticsearch URL format  https://username:password@es_host:9243. For example, to connect to Elastic  Cloud, create the Elasticsearch URL with the required authentication details and  pass it to the ElasticVectorSearch constructor as the named parameter  elasticsearch_url.   You can obtain your Elastic Cloud URL and login credentials by logging in to the  Elastic Cloud console at https://cloud.elastic.co, selecting your deployment, and  navigating to the \"Deployments\" page.   To obtain your Elastic Cloud password for the default \"elastic\" user:   1. Log in to the Elastic Cloud console at https://cloud.elastic.co  2. Go to \"Security\" > \"Users\"  3. Locate the \"elastic\" user and click \"Edit\"  4. Click \"Reset password\"  5. Follow the prompts to reset the password   The format for Elastic Cloud URLs is  https://username:password@cluster_id.region_id.gcp.cloud.es.io:9243. ",
"class KNNRetriever(BaseRetriever, BaseModel) is Configuration for this pydantic object.",
"class LlamaIndexRetriever(BaseRetriever, BaseModel) is Question-answering with sources over an LlamaIndex data structure.",
"class LlamaIndexGraphRetriever(BaseRetriever, BaseModel) is Question-answering with sources over an LlamaIndex graph data structure.",
"class MergerRetriever(BaseRetriever) is   This class merges the results of multiple retrievers.   Args:   retrievers: A list of retrievers to merge. ",
"class MilvusRetriever(BaseRetriever) is Add text to the Milvus store    Args:    texts (List[str])**: The text    metadatas (List[dict])**: Metadata dicts, must line up with existing store  ",
"class PineconeHybridSearchRetriever(BaseRetriever, BaseModel) is Configuration for this pydantic object.",
"class PubMedRetriever(BaseRetriever, PubMedAPIWrapper) is   It is effectively a wrapper for PubMedAPIWrapper.  It wraps load() to get_relevant_documents().  It uses all PubMedAPIWrapper arguments without any change. ",
"class SVMRetriever(BaseRetriever, BaseModel) is Configuration for this pydantic object.",
"class TFIDFRetriever(BaseRetriever, BaseModel) is Configuration for this pydantic object.",
"class TimeWeightedVectorStoreRetriever(BaseRetriever, BaseModel) is Retriever combining embedding similarity with recency.",
"class VespaRetriever(BaseRetriever) is Instantiate retriever from params.    Args:    url (str)**: Vespa app URL.    content_field (str)**: Field in results to return as Document page_content.    k (Optional[int])**: Number of Documents to return. Defaults to None.    metadata_fields(Sequence[str] or \"*\")**: Fields in results to include in     document metadata. Defaults to empty tuple ().    sources (Sequence[str] or \"*\" or None)**: Sources to retrieve     from. Defaults to None.    _filter (Optional[str])**: Document filter condition expressed in YQL.     Defaults to None.    yql (Optional[str])**: Full YQL query to be used. Should not be specified     if _filter or sources are specified. Defaults to None.    kwargs (Any)**: Keyword arguments added to query body.  ",
"class WeaviateHybridSearchRetriever(BaseRetriever) is Configuration for this pydantic object.",
"class WikipediaRetriever(BaseRetriever, WikipediaAPIWrapper) is   It is effectively a wrapper for WikipediaAPIWrapper.  It wraps load() to get_relevant_documents().  It uses all WikipediaAPIWrapper arguments without any change. ",
"class ZepRetriever(BaseRetriever) is A Retriever implementation for the Zep long-term memory store. Search your  user's long-term chat history with Zep.   Note: You will need to provide the user's `session_id` to use this retriever.   More on Zep:  Zep provides long-term conversation storage for LLM apps. The server stores,  summarizes, embeds, indexes, and enriches conversational AI chat  histories, and exposes them via simple, low-latency APIs.   For server installation instructions, see:  https://getzep.github.io/deployment/quickstart/ ",
"class ZillizRetriever(BaseRetriever) is Add text to the Zilliz store    Args:    texts (List[str])**: The text    metadatas (List[dict])**: Metadata dicts, must line up with existing store  ",
"class BaseDocumentCompressor(BaseModel, ABC) is Base abstraction interface for document compression.",
"class DocumentCompressorPipeline(BaseDocumentCompressor) is Document compressor that uses a pipeline of transformers.",
"class NoOutputParser(BaseOutputParser[str]) is Parse outputs that could return a null string of some sort.",
"class LLMChainExtractor(BaseDocumentCompressor) is LLM wrapper to use for compressing documents.",
"class LLMChainFilter(BaseDocumentCompressor) is Filter that drops documents that aren't relevant to the query.",
"class CohereRerank(BaseDocumentCompressor) is Configuration for this pydantic object.",
"class EmbeddingsFilter(BaseDocumentCompressor) is Embeddings to use for embedding document contents and queries.",
"class SelfQueryRetriever(BaseRetriever, BaseModel) is Retriever that wraps around a vector store and uses an LLM to generate  the vector store queries.",
"class ChildTool(BaseTool) is      raise SchemaAnnotationError(      f\"Tool definition for {name} must include valid type annotations\"      f\" for argument 'args_schema' to behave as expected.\ \"      f\"Expected annotation of 'Type[BaseModel]'\"      f\" but got '{args_schema_type}'.\ \"      f\"Expected class looks like:\ \"      f\"{typehint_mandate}\"     )   # Pass through to Pydantic's metaclass   return super().__new__(cls, name, bases, dct)   def _create_subset_model(  name: str, model: BaseModel, field_names: list ) -> Type[BaseModel]: ",
"class Tool(BaseTool) is Tool that takes in function or coroutine directly.",
"class StructuredTool(BaseTool) is Tool that can operate on any number of inputs.",
"class IFTTTWebhook(BaseTool) is IFTTT Webhook.   Args:   name: name of the tool   description: description of the tool   url: url to hit with the json event. ",
"class ApiConfig(BaseModel) is AI Plugin Definition.",
"class AIPluginToolSchema(BaseModel) is AIPLuginToolSchema.",
"class AIPluginTool(BaseTool) is Use the tool.",
"class ArxivQueryRun(BaseTool) is Tool that adds the capability to search using the Arxiv API.",
"class AzureCogsFormRecognizerTool(BaseTool) is Tool that queries the Azure Cognitive Services Form Recognizer API.   In order to set this up, follow instructions at:  https://learn.microsoft.com/en-us/azure/applied-ai-services/form-recognizer/quickstarts/get-started-sdks-rest-api?view=form-recog-3.0.0&pivots=programming-language-python ",
"class AzureCogsImageAnalysisTool(BaseTool) is Tool that queries the Azure Cognitive Services Image Analysis API.   In order to set this up, follow instructions at:  https://learn.microsoft.com/en-us/azure/cognitive-services/computer-vision/quickstarts-sdk/image-analysis-client-library-40 ",
"class AzureCogsSpeech2TextTool(BaseTool) is Tool that queries the Azure Cognitive Services Speech2Text API.   In order to set this up, follow instructions at:  https://learn.microsoft.com/en-us/azure/cognitive-services/speech-service/get-started-speech-to-text?pivots=programming-language-python ",
"class AzureCogsText2SpeechTool(BaseTool) is Tool that queries the Azure Cognitive Services Text2Speech API.   In order to set this up, follow instructions at:  https://learn.microsoft.com/en-us/azure/cognitive-services/speech-service/get-started-text-to-speech?pivots=programming-language-python ",
"class BingSearchRun(BaseTool) is Tool that adds the capability to query the Bing search API.",
"class BingSearchResults(BaseTool) is Tool that has capability to query the Bing Search API and get back json.",
"class BraveSearch(BaseTool) is Use the tool.",
"class DuckDuckGoSearchRun(BaseTool) is Tool that adds the capability to query the DuckDuckGo search API.",
"class DuckDuckGoSearchResults(BaseTool) is Tool that queries the Duck Duck Go Search API and get back json.",
"class FileCopyInput(BaseModel) is Input for CopyFileTool.",
"class FileDeleteInput(BaseModel) is Input for DeleteFileTool.",
"class FileSearchInput(BaseModel) is Input for FileSearchTool.",
"class DirectoryListingInput(BaseModel) is Input for ListDirectoryTool.",
"class FileMoveInput(BaseModel) is Input for MoveFileTool.",
"class ReadFileInput(BaseModel) is Input for ReadFileTool.",
"class BaseFileToolMixin(BaseModel) is Mixin for file system tools.",
"class WriteFileInput(BaseModel) is Input for WriteFileTool.",
"class SearchArgsSchema(BaseModel) is Run the tool.",
"class GetThreadSchema(BaseModel) is Run the tool.",
"class SendMessageSchema(BaseModel) is Create a message for an email.",
"class GooglePlacesSchema(BaseModel) is Tool that adds the capability to query the Google places API.",
"class GoogleSearchRun(BaseTool) is Tool that adds the capability to query the Google search API.",
"class GoogleSearchResults(BaseTool) is Tool that has capability to query the Google Search API and get back json.",
"class GoogleSerperRun(BaseTool) is Tool that adds the capability to query the Serper.dev Google search API.",
"class GoogleSerperResults(BaseTool) is Tool that has capability to query the Serper.dev Google Search API  and get back json.",
"class BaseGraphQLTool(BaseTool) is Base tool for querying a GraphQL API.",
"class HumanInputRun(BaseTool) is Tool that adds the capability to ask user for input.",
"class JiraAction(BaseTool) is Use the Atlassian Jira API to run an operation.",
"class JsonSpec(BaseModel) is Base class for JSON spec.",
"class JsonListKeysTool(BaseTool) is Tool for listing keys in a JSON spec.",
"class JsonGetValueTool(BaseTool) is Tool for getting a value in a JSON spec.",
"class MetaphorSearchResults(BaseTool) is Tool that has capability to query the Metaphor Search API and get back json.",
"class APIPropertyBase(BaseModel) is Base model for an API property.",
"class APIRequestBody(BaseModel) is A model for a request body.",
"class APIOperation(BaseModel) is A model for a single API operation.",
"class OpenWeatherMapQueryRun(BaseTool) is Tool that adds the capability to query using the OpenWeatherMap API.",
"class BaseBrowserTool(BaseTool) is Base class for browser tools.",
"class ClickToolInput(BaseModel) is Input for ClickTool.",
"class ClickTool(BaseBrowserTool) is Whether to consider only visible elements.",
"class CurrentWebPageTool(BaseBrowserTool) is Use the tool.",
"class ExtractHyperlinksToolInput(BaseModel) is Input for ExtractHyperlinksTool.",
"class ExtractHyperlinksTool(BaseBrowserTool) is Extract all hyperlinks on the page.",
"class ExtractTextTool(BaseBrowserTool) is Check that the arguments are valid.",
"class GetElementsToolInput(BaseModel) is Input for GetElementsTool.",
"class GetElementsTool(BaseBrowserTool) is Use the tool.",
"class NavigateToolInput(BaseModel) is Input for NavigateToolInput.",
"class NavigateTool(BaseBrowserTool) is Use the tool.",
"class NavigateBackTool(BaseBrowserTool) is Navigate back to the previous page in the browser history.",
"class QueryPowerBITool(BaseTool) is Tool for querying a Power BI Dataset.",
"class InfoPowerBITool(BaseTool) is Tool for getting metadata about a PowerBI Dataset.",
"class ListPowerBITool(BaseTool) is Tool for getting tables names.",
"class PubmedQueryRun(BaseTool) is Tool that adds the capability to search using the PubMed API.",
"class PythonREPLTool(BaseTool) is A tool for running python code in a REPL.",
"class PythonAstREPLTool(BaseTool) is A tool for running python code in a REPL.",
"class BaseRequestsTool(BaseModel) is Base class for requests tools.",
"class RequestsGetTool(BaseRequestsTool, BaseTool) is Tool for making a GET request to an API endpoint.",
"class RequestsPostTool(BaseRequestsTool, BaseTool) is Tool for making a POST request to an API endpoint.",
"class RequestsPatchTool(BaseRequestsTool, BaseTool) is Tool for making a PATCH request to an API endpoint.",
"class RequestsPutTool(BaseRequestsTool, BaseTool) is Tool for making a PUT request to an API endpoint.",
"class RequestsDeleteTool(BaseRequestsTool, BaseTool) is Tool for making a DELETE request to an API endpoint.",
"class SceneXplainInput(BaseModel) is Input for SceneXplain.",
"class SceneXplainTool(BaseTool) is Tool that adds the capability to explain images.",
"class SearxSearchRun(BaseTool) is Tool that adds the capability to query a Searx instance.",
"class SearxSearchResults(BaseTool) is Tool that has the capability to query a Searx instance and get back json.",
"class ShellInput(BaseModel) is Commands for the Bash Shell tool.",
"class ShellTool(BaseTool) is Tool to run shell commands.",
"class SleepInput(BaseModel) is Input for CopyFileTool.",
"class SleepTool(BaseTool) is Tool that adds the capability to sleep.",
"class BaseSparkSQLTool(BaseModel) is Base tool for interacting with Spark SQL.",
"class Config(BaseTool.Config) is Configuration for this pydantic object.",
"class QuerySparkSQLTool(BaseSparkSQLTool, BaseTool) is Tool for querying a Spark SQL.",
"class InfoSparkSQLTool(BaseSparkSQLTool, BaseTool) is Tool for getting metadata about a Spark SQL.",
"class ListSparkSQLTool(BaseSparkSQLTool, BaseTool) is Tool for getting tables names.",
"class QueryCheckerTool(BaseSparkSQLTool, BaseTool) is Use an LLM to check if a query is correct.  Adapted from https://www.patterns.app/blog/2023/01/18/crunchbot-sql-analyst-gpt/",
"class BaseSQLDatabaseTool(BaseModel) is Base tool for interacting with a SQL database.",
"class QuerySQLDataBaseTool(BaseSQLDatabaseTool, BaseTool) is Tool for querying a SQL database.",
"class InfoSQLDatabaseTool(BaseSQLDatabaseTool, BaseTool) is Tool for getting metadata about a SQL database.",
"class ListSQLDatabaseTool(BaseSQLDatabaseTool, BaseTool) is Tool for getting tables names.",
"class QuerySQLCheckerTool(BaseSQLDatabaseTool, BaseTool) is Use an LLM to check if a query is correct.  Adapted from https://www.patterns.app/blog/2023/01/18/crunchbot-sql-analyst-gpt/",
"class SteamshipImageGenerationTool(BaseTool) is Tool used to generate images from a text-prompt.",
"class BaseVectorStoreTool(BaseModel) is Base class for tools that use a VectorStore.",
"class VectorStoreQATool(BaseVectorStoreTool, BaseTool) is Tool for the VectorDBQA chain. To be initialized with name and chain.",
"class VectorStoreQAWithSourcesTool(BaseVectorStoreTool, BaseTool) is Tool for the VectorDBQAWithSources chain.",
"class WikipediaQueryRun(BaseTool) is Tool that adds the capability to search using the Wikipedia API.",
"class WolframAlphaQueryRun(BaseTool) is Tool that adds the capability to query using the Wolfram Alpha SDK.",
"class YouTubeSearchTool(BaseTool) is Use the tool.",
"class ZapierNLARunAction(BaseTool) is   Args:   action_id: a specific action ID (from list actions) of the action to execute    (the set api_key must be associated with the action owner)   instructions: a natural language instruction string for using the action    (eg. \"get the latest email from Mike Knoop\" for \"Gmail: find email\" action)   params: a dict, optional. Any params provided will *override* AI guesses    from `instructions` (see \"understanding the AI guessing flow\" here:    https://nla.zapier.com/api/v1/docs)  ",
"class ZapierNLAListActions(BaseTool) is   Args:   None  ",
"class ApifyWrapper(BaseModel) is Wrapper around Apify.   To use, you should have the ``apify-client`` python package installed,  and the environment variable ``APIFY_API_TOKEN`` set with your API key, or pass  `apify_api_token` as a named parameter to the constructor. ",
"class ArxivAPIWrapper(BaseModel) is Wrapper around ArxivAPI.   To use, you should have the ``arxiv`` python package installed.  https://lukasschwab.me/arxiv.py/index.html  This wrapper will use the Arxiv API to conduct searches and  fetch document summaries. By default, it will return the document summaries  of the top-k results.  It limits the Document content by doc_content_chars_max.  Set doc_content_chars_max=None if you don't want to limit the content size.   Parameters:   top_k_results: number of the top-scored document used for the arxiv tool   ARXIV_MAX_QUERY_LENGTH: the cut limit on the query used for the arxiv tool.   load_max_docs: a limit to the number of loaded documents   load_all_available_meta:     if True: the `metadata` of the loaded Documents gets all available meta info    (see https://lukasschwab.me/arxiv.py/index.html#Result),     if False: the `metadata` gets only the most informative fields.  ",
"class LambdaWrapper(BaseModel) is Wrapper for AWS Lambda SDK.   Docs for using:   1. pip install boto3  2. Create a lambda function using the AWS Console or CLI  3. Run `aws configure` and enter your AWS credentials  ",
"class BibtexparserWrapper(BaseModel) is Wrapper around bibtexparser.   To use, you should have the ``bibtexparser`` python package installed.  https://bibtexparser.readthedocs.io/en/master/   This wrapper will use bibtexparser to load a collection of references from  a bibtex file and fetch document summaries. ",
"class BingSearchAPIWrapper(BaseModel) is Wrapper for Bing Search API.   In order to set this up, follow instructions at:  https://levelup.gitconnected.com/api-tutorial-how-to-use-bing-web-search-api-in-python-4165d5592a7e ",
"class DuckDuckGoSearchAPIWrapper(BaseModel) is Wrapper for DuckDuckGo Search API.   Free and does not require any setup ",
"class GooglePlacesAPIWrapper(BaseModel) is Wrapper around Google Places API.   To use, you should have the ``googlemaps`` python package installed,   **an API key for the google maps platform**,   and the enviroment variable ''GPLACES_API_KEY''   set with your API key , or pass 'gplaces_api_key'   as a named parameter to the constructor.   By default, this will return the all the results on the input query.   You can use the top_k_results argument to limit the number of results.   Example:   .. code-block:: python      from langchain import GooglePlacesAPIWrapper    gplaceapi = GooglePlacesAPIWrapper() ",
"class GoogleSearchAPIWrapper(BaseModel) is Wrapper for Google Search API.   Adapted from: Instructions adapted from https://stackoverflow.com/questions/  37083058/  programmatically-searching-google-in-python-using-custom-search   TODO: DOCS for using it  1. Install google-api-python-client  - If you don't already have a Google account, sign up.  - If you have never created a Google APIs Console project,  read the Managing Projects page and create a project in the Google API Console.  - Install the library using pip install google-api-python-client  The current version of the library is 2.70.0 at this time   2. To create an API key:  - Navigate to the APIs & Services→Credentials panel in Cloud Console.  - Select Create credentials, then select API key from the drop-down menu.  - The API key created dialog box displays your newly created key.  - You now have an API_KEY   3. Setup Custom Search Engine so you can search the entire web  - Create a custom search engine in this link.  - In Sites to search, add any valid URL (i.e. www.stackoverflow.com).  - That’s all you have to fill up, the rest doesn’t matter.  In the left-side menu, click Edit search engine → {your search engine name}  → Setup Set Search the entire web to ON. Remove the URL you added from  the list of Sites to search.  - Under Search engine ID you’ll find the search-engine-ID.   4. Enable the Custom Search API  - Navigate to the APIs & Services→Dashboard panel in Cloud Console.  - Click Enable APIs and Services.  - Search for Custom Search API and click on it.  - Click Enable.  URL for it: https://console.cloud.google.com/apis/library/customsearch.googleapis  .com ",
"class GoogleSerperAPIWrapper(BaseModel) is Wrapper around the Serper.dev Google Search API.   You can create a free API key at https://serper.dev.   To use, you should have the environment variable ``SERPER_API_KEY``  set with your API key, or pass `serper_api_key` as a named parameter  to the constructor.   Example:   .. code-block:: python     from langchain import GoogleSerperAPIWrapper    google_serper = GoogleSerperAPIWrapper() ",
"class GraphQLAPIWrapper(BaseModel) is Wrapper around GraphQL API.   To use, you should have the ``gql`` python package installed.  This wrapper will use the GraphQL API to conduct queries. ",
"class JiraAPIWrapper(BaseModel) is Wrapper for Jira API.",
"class MetaphorSearchAPIWrapper(BaseModel) is Wrapper for Metaphor Search API.",
"class OpenWeatherMapAPIWrapper(BaseModel) is Wrapper for OpenWeatherMap API using PyOWM.   Docs for using:   1. Go to OpenWeatherMap and sign up for an API key  2. Save your API KEY into OPENWEATHERMAP_API_KEY env variable  3. pip install pyowm ",
"class PowerBIDataset(BaseModel) is Create PowerBI engine from dataset ID and credential or token.   Use either the credential or a supplied token to authenticate.  If both are supplied the credential is used to generate a token.  The impersonated_user_name is the UPN of a user to be impersonated.  If the model is not RLS enabled, this will be ignored. ",
"class PubMedAPIWrapper(BaseModel) is   Wrapper around PubMed API.   This wrapper will use the PubMed API to conduct searches and fetch  document summaries. By default, it will return the document summaries  of the top-k results of an input search.   Parameters:   top_k_results: number of the top-scored document used for the PubMed tool   load_max_docs: a limit to the number of loaded documents   load_all_available_meta:     if True: the `metadata` of the loaded Documents gets all available meta info    (see https://www.ncbi.nlm.nih.gov/books/NBK25499/#chapter4.ESearch)     if False: the `metadata` gets only the most informative fields. ",
"class PythonREPL(BaseModel) is Simulates a standalone Python REPL.",
"class SceneXplainAPIWrapper(BaseSettings, BaseModel) is Wrapper for SceneXplain API.   In order to set this up, you need API key for the SceneXplain API.  You can obtain a key by following the steps below.  - Sign up for a free account at https://scenex.jina.ai/.  - Navigate to the API Access page (https://scenex.jina.ai/api)    and create a new API key. ",
"class SearxSearchWrapper(BaseModel) is Wrapper for Searx API.   To use you need to provide the searx host by passing the named parameter  ``searx_host`` or exporting the environment variable ``SEARX_HOST``.   In some situations you might want to disable SSL verification, for example  if you are running searx locally. You can do this by passing the named parameter  ``unsecure``. You can also pass the host url scheme as ``http`` to disable SSL.   Example:   .. code-block:: python     from langchain.utilities import SearxSearchWrapper    searx = SearxSearchWrapper(searx_host=\"http://localhost:8888\")   Example with SSL disabled:   .. code-block:: python     from langchain.utilities import SearxSearchWrapper    # note the unsecure parameter is not needed if you pass the url scheme as    # http    searx = SearxSearchWrapper(searx_host=\"http://localhost:8888\",              unsecure=True)   ",
"class SerpAPIWrapper(BaseModel) is Wrapper around SerpAPI.   To use, you should have the ``google-search-results`` python package installed,  and the environment variable ``SERPAPI_API_KEY`` set with your API key, or pass  `serpapi_api_key` as a named parameter to the constructor.   Example:   .. code-block:: python     from langchain import SerpAPIWrapper    serpapi = SerpAPIWrapper() ",
"class TwilioAPIWrapper(BaseModel) is Sms Client using Twilio.   To use, you should have the ``twilio`` python package installed,  and the environment variables ``TWILIO_ACCOUNT_SID``, ``TWILIO_AUTH_TOKEN``, and  ``TWILIO_FROM_NUMBER``, or pass `account_sid`, `auth_token`, and `from_number` as  named parameters to the constructor.   Example:   .. code-block:: python     from langchain.utilities.twilio import TwilioAPIWrapper    twilio = TwilioAPIWrapper(     account_sid=\"ACxxx\",     auth_token=\"xxx\",     from_number=\"+10123456789\"    )    twilio.run('test', '+12484345508') ",
"class WikipediaAPIWrapper(BaseModel) is Wrapper around WikipediaAPI.   To use, you should have the ``wikipedia`` python package installed.  This wrapper will use the Wikipedia API to conduct searches and  fetch page summaries. By default, it will return the page summaries  of the top-k results.  It limits the Document content by doc_content_chars_max. ",
"class WolframAlphaAPIWrapper(BaseModel) is Wrapper for Wolfram Alpha.   Docs for using:   1. Go to wolfram alpha and sign up for a developer account  2. Create an app and get your APP ID  3. Save your APP ID into WOLFRAM_ALPHA_APPID env variable  4. pip install wolframalpha  ",
"class ZapierNLAWrapper(BaseModel) is Wrapper for Zapier NLA.   Full docs here: https://nla.zapier.com/api/v1/docs   Note: this wrapper currently only implemented the `api_key` auth method for  testingand server-side production use cases (using the developer's connected  accounts on Zapier.com)   For use-cases where LangChain + Zapier NLA is powering a user-facing application,  and LangChain needs access to the end-user's connected accounts on Zapier.com,  you'll need to use oauth. Review the full docs above and reach out to  nla@zapier.com for developer support. ",
"class AzureSearchVectorStoreRetriever(BaseRetriever, BaseModel) is Configuration for this pydantic object.",
"class VectorStoreRetriever(BaseRetriever, BaseModel) is Configuration for this pydantic object.",
"class ClickhouseSettings(BaseSettings) is ClickHouse Client Configuration   Attribute:   clickhouse_host (str) : An URL to connect to MyScale backend.         Defaults to 'localhost'.   clickhouse_port (int) : URL port to connect with HTTP. Defaults to 8443.   username (str) : Username to login. Defaults to None.   password (str) : Password to login. Defaults to None.   index_type (str)**: index type string.   index_param (list)**: index build parameter.   index_query_params(dict)**: index query parameters.   database (str) : Database name to find the table. Defaults to 'default'.   table (str) : Table name to operate on.        Defaults to 'vector_table'.   metric (str) : Metric to compute distance,      supported are ('angular', 'euclidean', 'manhattan', 'hamming',      'dot'). Defaults to 'angular'.      https://github.com/spotify/annoy/blob/main/src/annoymodule.cc#L149-L169    column_map (Dict) : Column type map to project column name onto langchain        semantics. Must have keys: `text`, `id`, `vector`,        must be same size to number of columns. For example:        .. code-block:: python          {          'id': 'text_id',          'uuid': 'global_unique_id'          'embedding': 'text_embedding',          'document': 'text_plain',          'metadata': 'metadata_dictionary_in_json',         }         Defaults to identity map. ",
"class MyScaleSettings(BaseSettings) is MyScale Client Configuration   Attribute:   myscale_host (str) : An URL to connect to MyScale backend.         Defaults to 'localhost'.   myscale_port (int) : URL port to connect with HTTP. Defaults to 8443.   username (str) : Username to login. Defaults to None.   password (str) : Password to login. Defaults to None.   index_type (str)**: index type string.   index_param (dict)**: index build parameter.   database (str) : Database name to find the table. Defaults to 'default'.   table (str) : Table name to operate on.        Defaults to 'vector_table'.   metric (str) : Metric to compute distance,      supported are ('l2', 'cosine', 'ip'). Defaults to 'cosine'.   column_map (Dict) : Column type map to project column name onto langchain        semantics. Must have keys: `text`, `id`, `vector`,        must be same size to number of columns. For example:        .. code-block:: python          {          'id': 'text_id',          'vector': 'text_embedding',          'text': 'text_plain',          'metadata': 'metadata_dictionary_in_json',         }         Defaults to identity map.  ",
"class BaseModel(Base) is    Get or create a collection.   Returns [Collection, bool] where the bool is True if the collection was created.  ",
"class EmbeddingStore(BaseModel) is   VectorStore implementation using Postgres and pgvector.  - `connection_string` is a postgres connection string.  - `embedding_function` any embedding function implementing   `langchain.embeddings.base.Embeddings` interface.  - `collection_name` is the name of the collection to use. (default: langchain)   - NOTE: This is not the name of the table, but the name of the collection.    The tables will be created when initializing the store (if not exists)    So, make sure the user has the right permissions to create tables.  - `distance_strategy` is the distance strategy to use. (default: EUCLIDEAN)   - `EUCLIDEAN` is the euclidean distance.   - `COSINE` is the cosine distance.  - `pre_delete_collection` if True, will delete the collection if it exists.   (default: False)   - Useful for testing. ",
"class JsonSerializer(BaseSerializer) is Serializes data in json using the json package from python standard library.",
"class BsonSerializer(BaseSerializer) is Serializes data in binary json using the bson python package.",
"class ParquetSerializer(BaseSerializer) is Serializes data in Apache Parquet format using the pyarrow package.",
"class DocArrayDoc(BaseDoc) is Run more texts through the embeddings and add to the vectorstore.    Args:    texts: Iterable of strings to add to the vectorstore.    metadatas: Optional list of metadatas associated with the texts.    Returns:    List of ids from adding the texts into the vectorstore.  ",
