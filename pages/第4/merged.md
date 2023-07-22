# 数据概览

在当代，大量的大语言模型（LLM）应用需要用户特定的数据，而这部分数据并不包含在模型的训练集中。因此，如何加载、转化、存储和查询这些数据，便成为我们需要解决的核心问题。针对这一需求，LangChain 提供了一整套解决方案。它通过一系列的构建模块，包括文档加载器（Document loaders）、文档转换器（Document transformers）、文本嵌入模型（Text embedding models）、向量存储（Vector stores）以及检索器（Retrievers），让我们能够自由地与自己的数据和大语言模型进行交互。

在以下的篇章中，我们将详细地探讨每一个构建模块的实现原理。

## 2. 文档加载器：灵活加载文档的利器

文档加载器能够从众多不同的来源加载文档。一个文档可以简单地理解为一段文字和其相关的元数据。例如，文档加载器可以从一个简单的.txt 文件中，从任意网页的文本内容，甚至从 YouTube 视频的字幕中加载数据。

文档加载器提供一个 "load" 方法，用于从配置的数据源加载数据作为文档。它们还可选择性地实现 "lazy load"，以便于懒加载数据到内存中。

## 3. 文档转换器：为你的应用精确定制文档

一旦文档被加载，我们往往需要对它们进行一定的转换，以便更好地适应我们的应用。最简单的例子是，我们可能需要将一个长文档分割成更小的片段，以便它们能够适应模型的上下文窗口。LangChain 内置了多种文档转换器，使得分割、组合、过滤和其他形式的文档操作变得轻而易举。

## 4. 文本嵌入模型：将无结构文本转化为浮点数列表

Embeddings 类是为接口化文本嵌入模型而设计的一个类。目前有许多嵌入模型提供商（如 OpenAI, Cohere, Hugging Face 等），而这个类就是为了提供一个所有这些供应商的标准接口。

Embeddings 类将一段文本创建为一个向量表示。这是有用的，因为我们可以将文本在向量空间中进行思考，并做诸如语义搜索这样的操作，我们可以寻找在向量空间中最相似的文本片段。

LangChain 中的基础 Embeddings 类公开了两种方法：一种用于嵌入文档，另一种用于嵌入查询。前者接受多个文本作为输入，而后者接受单个文本。之所以将这两种方法分开，是因为一些嵌入提供商对待待查询的文档与查询本身有着不同的嵌入方法。

## 5. 向量存储：为你的嵌入数据提供存储和搜索功能

最常见的存储和搜索无结构数据的方式之一是将其嵌入并存储生成的嵌入向量，然后在查询时嵌入无结构查询并检索那些与嵌入查询“最相似”的嵌入向量。向量存储负责存储嵌入数据并执行向量搜索。

## 6. 检索器：返回文档的接口

检索器是一个能够根据无结构查询返回文档的接口。它比向量存储的应用更为广泛。检索器不需要能够存储文档，只需返回（或检索）文档即可。向量存储可以作为检索器的支撑结构，但也存在其他类型的检索器。

以上就是 LangChain 通过这些技术准备数据，让我们使用自己的数据与大语言模型交互的全过程。LangChain 用这种简单而高效的方式，让我们可以轻松地处理各种数据，并使之与我们的大语言模型交互。
# 3.1.1 加载器概览

在 LangChain 的数据处理流程中，Document Loaders 起着至关重要的作用。它们被用来从各种来源加载数据，并将其转换为“文档”（Document）的形式。

一个“文档”可以理解为一段文本及其相关元数据。例如，我们可以有专门用于加载简单.txt 文件的文档加载器，也可以有加载任何网页的文本内容的文档加载器，甚至还可以有加载 YouTube 视频转录文本的文档加载器。不同类型的文档加载器，使得 LangChain 可以从各种各样的数据源中抽取并处理数据。

这些文档加载器都会暴露出一个名为 "load" 的方法，用于从配置的数据源加载数据作为文档。这个 "load" 方法可以从指定的数据源中读取数据，并将其转换成一份或多份文档。这使得 LangChain 能够处理各种形式的输入数据，不仅仅限于文本文件，还可以是网页、视频字幕等等。

值得注意的是，文档加载器还可以选择性地实现一个名为 "lazy load" 的方法，这个方法的作用是实现数据的懒加载，即在需要时才将数据加载到内存中。这种方式可以有效减少内存占用，并提高数据处理的效率。

总的来说，通过 Document Loaders，LangChain 可以将各种各样的数据源无缝地转换为标准的文档形式，为后续的数据处理和分析提供了坚实的基础。

下面是最简单的文档加载器的代码示例：

加载简单.txt 文件的文档加载器。

```
from langchain.document_loaders import TextLoader

loader = TextLoader("./index.md")
loader.load()

```
打印导入的结果。

```
[
    Document(page_content='---\nsidebar_position: 0\n---\n# Document loaders\n\nUse document loaders to load data from a source as `Document`\'s. A `Document` is a piece of text\nand associated metadata. For example, there are document loaders for loading a simple `.txt` file, for loading the text\ncontents of any web page, or even for loading a transcript of a YouTube video.\n\nEvery document loader exposes two methods:\n1. "Load": load documents from the configured source\n2. "Load and split": load documents from the configured source and split them using the passed in text splitter\n\nThey optionally implement:\n\n3. "Lazy load": load documents into memory lazily\n', metadata={'source': '../docs/docs_skeleton/docs/modules/data_connection/document_loaders/index.md'})
]
```
# 多元加载器

LangChain 通过实现各种服务提供商的数据加载器（Loader），实现了对多元化数据源的处理。这些加载器的设计使得 LangChain 可以无缝地从各种服务提供商加载数据，进一步扩大了其在不同业务场景的应用范围。下面，我们将对这些加载器进行分类并简要描述。

## 1. 文件和目录加载器

这类加载器主要处理存储在本地或云端的文件和目录，例如 `CSVLoader`，`DirectoryLoader`，`JSONLoader`，`S3DirectoryLoader`，`S3FileLoader` 等。

## 2. 文本和标记语言文件加载器

这类加载器负责处理各种文本和标记语言格式的文件，如 `BSHTMLLoader`，`MarkdownLoader`，`TextLoader`，`UnstructuredHTMLLoader`，`UnstructuredMarkdownLoader` 等。

## 3. 文档和电子书加载器

处理电子书和文档的加载器包括 `Docx2txtLoader`，`PDFMinerLoader`，`UnstructuredEPubLoader`，`UnstructuredPDFLoader`，`UnstructuredWordDocumentLoader` 等。

## 4. 社交媒体和在线论坛加载器

这类加载器主要用于处理来自社交媒体和在线论坛的数据，如 `FacebookChatLoader`，`GitHubIssuesLoader`，`RedditPostsLoader`，`TwitterTweetLoader` 等。

## 5. 云服务提供商加载器

对于云服务提供商的数据源，LangChain 提供了相应的加载器，如 `AirbyteJSONLoader`，`AirtableLoader`，`AzureBlobStorageContainerLoader`，`AzureBlobStorageFileLoader`，`GoogleDriveLoader`，`OneDriveLoader` 等。

## 6. 文本聊天和消息服务加载器

这类加载器主要处理各种聊天和消息服务的数据，如 `DiscordChatLoader`，`SlackDirectoryLoader`，`TelegramChatApiLoader`，`TelegramChatFileLoader`，`WhatsAppChatLoader` 等。

## 7. 网页和网站数据加载器

对于网页和网站的数据，如 `SitemapLoader`，`UnstructuredURLLoader`，`WebBaseLoader` 等加载器提供了处理方法。

## 8. 其他特殊类型的加载器

这类加载器主要处理一些特定的或者特殊的数据源，如 `HuggingFaceDatasetLoader`，`UnstructuredImageLoader`，`WeatherDataLoader`，`YoutubeAudioLoader` 等。

总的来说，通过实现这些加载器，LangChain 可以从各种各样的服务提供商处加载数据，进一步提升了它的多样化数据处理能力。这对于需要处理大量、多样性数据的机器学习、自然语言处理等领域具有非常重要的意义。
# LangChain 如何加载不同格式的数据

LangChain 的数据加载能力并不限于单一的数据源或格式，它可以处理各种常见的数据格式，例如 CSV、文件目录、HTML、JSON、Markdown 以及 PDF 等。下面，我们将分别解析一下这些不同格式数据的加载方法。

## 1. CSV 文件的加载

逗号分隔值（Comma-Separated Values，简称 CSV）文件是一种使用逗号来分隔值的文本文件。每一行都是一条数据记录，每条记录包含一个或多个用逗号分隔的字段。LangChain 可以加载 CSV 数据，其中每一行都被视为一个独立的文档。

## 2. 文件目录的加载

对于文件目录，LangChain 提供了一种方法来加载目录中的所有文档。在底层，它默认使用 UnstructuredLoader 来实现这个功能。这意味着，只要将文档存放在同一目录下，无论数量多少，LangChain 都能够将它们全部加载进来。

## 3. HTML 文件的加载

HTML 是用于设计在 Web 浏览器中显示的文档的标准标记语言。LangChain 可以将 HTML 文档加载为我们后续使用的文档格式。这就意味着，我们可以直接从网页上提取并处理数据。

## 4. JSON 文件的加载

JSON 是一种使用人类可读文本来存储和传输数据对象的开放标准文件格式和数据交换格式，这些对象由属性-值对和数组（或其他可序列化值）组成。LangChain 的 JSONLoader 使用指定的 jq 模式来解析 JSON 文件。jq 是一种适用于 Python 的软件包。JSON 文件的每一行都被视为一个独立的文档。

## 5. Markdown 文件的加载

Markdown 是一种使用纯文本编辑器创建格式化文本的轻量级标记语言。LangChain 可以将 Markdown 文档加载为我们后续使用的文档格式。

## 6. PDF 文件的加载

PDF（Portable Document Format）是 Adobe 在 1992 年开发的一种文件格式，用于以独立于应用软件、硬件和操作系统的方式呈现文档，包括文本格式化和图像。LangChain 可以将 PDF 文档加载为我们后续使用的文档格式。

总的来说，通过对各种不同数据格式的加载能力，LangChain 为大规模、多样性的数据处理提供了强大的支持。
# LangChain 的文档转换器和文本分割器：工作原理与应用

LangChain 为处理语言数据提供了一系列内置工具，包括文档加载器、文档转换器和文本分割器等。在你加载了文档之后，通常需要对其进行转换以更好地适应你的应用。这就需要用到 LangChain 的文档转换器和文本分割器。

## 1. 文档转换器

文档转换器可以轻松地将文档进行分割、合并、过滤和其他操作，以满足你的实际需求。例如，你可能希望将长文档分割成小块，以便适应你模型的上下文窗口。

## 2. 文本分割器

当处理长文本时，往往需要将文本分割成块。尽管这看起来简单，但实际上可能涉及很多复杂性。理想情况下，你会希望将语义相关的文本部分保持在一起。而 "语义相关" 的含义可能取决于文本的类型。下面将介绍几种实现这一目标的方法。

在高层次上，文本分割器的工作原理如下：

- 将文本分割成小的、语义上有意义的块（通常是句子）。
- 将这些小块开始组合成一个大的块，直到达到某个大小（通过某种函数进行测量）。
- 一旦达到该大小，将该块作为自己的文本片段，然后开始创建新的文本块，新的文本块和前一个文本块会有一些重叠（以保持块与块之间的上下文）。

这意味着，你可以沿着两个不同的轴来定制你的文本分割器：

- 文本如何被分割
- 块的大小如何被测量

## 3. 使用文本分割器

默认推荐的文本分割器是 `RecursiveCharacterTextSplitter`。这个文本分割器接受一个字符列表，它尝试基于第一个字符进行分割，但如果任何块太大，它就会移动到下一个字符，依此类推。默认情况下，它尝试分割的字符是 ["\n\n", "\n", " ", ""]。

除了可以控制分割的字符，你还可以控制以下几点：

- `length_function`：如何计算块的长度。默认只计算字符数量，但是在此通常会传入一个标记计数器。
- `chunk_size`：你的块的最大大小（由长度函数测量）。
- `chunk_overlap`：块之间的最大重叠。有一些重叠可以在块之间保持连续性（例如采用滑动窗口的方式）。
- `add_start_index`：是否在元数据中包含每个块在原始文档中的起始位置。

通过以上内容，我们可以看到 LangChain 的文档转换器和文本分割器为处理和转换大规模文本提供了有效的工具，无论是文本的分割、组合、过滤还是其他操作，都能够得心应手。
# 文本分割方法

在处理大规模文本数据时，LangChain 提供了多种文本分割方法，以满足各种类型的应用需求。本文将详细介绍这些方法的特性和工作原理。

## 1. 按字符分割

这是最简单的方法。它基于字符（默认为 "\n\n"）进行分割，并通过字符数量来衡量块的大小。

- 文本如何分割：按单个字符
- 块大小如何测量：按字符数量

## 2. 代码分割

`CodeTextSplitter` 允许你对多种语言的代码进行分割。导入枚举 `Language` 并指定语言即可。

## 3. Markdown 标题文本分割器

这种分割器的动机源于许多聊天或问答应用需要在嵌入和向量存储之前将输入文档进行分块。Pinecone 的这些注释提供了一些有用的提示。

## 4. 递归按字符分割

这是通用文本推荐的文本分割器。它由一系列字符参数化。直到块足够小，它会尝试按顺序分割它们。默认列表是 ["\n\n", "\n", " ", ""]。这样做的效果是尽可能地将所有段落（然后是句子，然后是单词）保持在一起，因为这些通常看起来是最强的语义相关的文本部分。

## 5. 按标记分割

语言模型有一个标记限制。你不应该超过这个标记限制。因此，当你将文本分割成块时，计算标记的数量是个好主意。有许多标记化器。当你在文本中计数标记时，应该使用与语言模型中使用的相同的标记化器。

通过以上内容，我们可以看到 LangChain 提供的各种文本分割方法能够满足不同类型的文本处理需求，无论是基于字符的分割，还是基于特定语言代码的分割，甚至是针对 Markdown 格式文本的分割，都具有各自的优点和特性，为处理和转换大规模文本提供了有效的工具。
# 优化文本处理的方法

在处理大规模文本数据时，我们可以利用 Doctran 库，基于 OpenAI 的函数调用特性，从文档中抽取具体的元数据。这种方法能够从多个方面助力我们处理文本信息。

## 1. Doctran 抽取文档特性

我们可以从文档中抽取有用的特性，这对多种任务都很有帮助，包括：

- 分类：将文档分类到不同的类别
- 数据挖掘：提取可以用于数据分析的结构化数据
- 风格转换：改变文本的写作方式，使其更接近用户预期的输入，从而提高向量搜索的结果

## 2. Doctran 询问文档

通常存储在向量库知识库中的文档以叙述或会话格式存储。然而，大多数用户查询都是以问题格式提出的。如果我们在将文档向量化之前将其转换为问答格式，我们可以增加检索到相关文档的可能性，减少检索到无关文档的可能性。

我们可以利用 Doctran 库，使用 OpenAI 的函数调用特性来“询问”文档。

## 3. Doctran 翻译文档

通过嵌入比较文档具有跨多种语言工作的优点。"Harrison says hello" 和 "Harrison dice hola" 将在向量空间中占据相似的位置，因为它们在语义上有相同的含义。

然而，在将文档向量化之前，使用 LLM 将文档翻译成其他语言仍然可能是有用的。这在用户预期以不同的语言查询知识库，或者对于给定的语言没有可用的最先进的嵌入模型时，特别有帮助。

我们可以利用 Doctran 库，使用 OpenAI 的函数调用特性在语言之间翻译文档。

## 4. OpenAI 函数元数据标签器

标记文档常常会很有用，这可以为文档添加如标题、调性或长度等结构化的元数据，以便后续进行更有针对性的相似性搜索。然而，对于大量文档，手动执行这个标记过程可能会很繁琐。

OpenAIMetadataTagger 文档转换器通过根据提供的模式从每个提供的文档中提取元数据，自动化了这个过程。它在底层使用一个可配置的 OpenAI Functions-powered 链，所以如果你传递一个自定义的 LLM 实例，它必须是一个支持函数的 OpenAI 模型。

注意：这个文档转换器最适合处理完整的文档，所以在进行任何其他分割或处理之前，最好先使用整个文档运行它。

我们可以看到，通过使用 Doctran 库，我们不仅可以提取文档的重要特性，还可以对文档进行提问、翻译等操作，大大提升了文本处理的效率与准确性，无论是对于分类、数据挖掘，还是风格转换等任务，都具有重要的实际意义。同时，OpenAI 的元数据标签器也为我们提供了一种自动化处理文档的有效方法，极大地简化了文本处理的过程。
# 文本嵌入模型概览

文本嵌入模型，如其名称所示，是用于处理文本信息的一种重要工具。具体而言，`Embeddings` 类是专门设计用来与文本嵌入模型交互的类。目前，有许多嵌入模型提供商（如 OpenAI、Cohere、Hugging Face 等），而此类旨在为所有这些供应商提供一个标准的接口。

文本嵌入模型的核心工作就是为一段文本创建一个向量表示。这非常有用，因为我们可以在向量空间中思考文本，以及执行如语义搜索这样的操作，即在向量空间中寻找最相似的文本片段。

在 LangChain 中，基础的 `Embeddings` 类暴露了两种方法：一种用于嵌入文档，另一种用于嵌入查询。前者接收多个文本作为输入，而后者接收单个文本。之所以要将这两种方法区分开来，是因为一些嵌入供应商对于文档（要搜索的对象）与查询（搜索查询本身）有不同的嵌入方法。


# 文本嵌入类型

LangChain 的文本嵌入类型丰富多样，无论面对何种文本处理需求或特定挑战，用户都有可能在其提供的嵌入类型列表中找到合适的解决方案。下面我们将对其支持的文本嵌入类型进行分类，并详细阐述各类的特点。


### 1. 自然语言模型（Natural Language Model）嵌入

此类嵌入主要利用诸如 OpenAI、Hugging Face 等自然语言处理（NLP）模型进行文本嵌入，特点是能够充分利用大规模预训练模型的语义理解能力。包括以下几种类型：

- "OpenAIEmbeddings"
- "HuggingFaceEmbeddings"
- "HuggingFaceHubEmbeddings"
- "HuggingFaceInstructEmbeddings"
- "SelfHostedHuggingFaceEmbeddings"
- "SelfHostedHuggingFaceInstructEmbeddings"

### 2. AI 平台或云服务嵌入

此类嵌入主要依托 AI 平台或云服务的能力进行文本嵌入，典型的包括 Elasticsearch、SagemakerEndpoint、DeepInfra 等。这些嵌入方式主要特点是能够利用云计算的优势，处理大规模的文本数据。

- "ElasticsearchEmbeddings"
- "SagemakerEndpointEmbeddings"
- "DeepInfraEmbeddings"
- "VertexAIEmbeddings"

### 3. 专门的嵌入模型

此类嵌入主要是专门用于处理特定结构文本的嵌入模型，例如 AlephAlpha 的 AsymmetricSemanticEmbedding 和 SymmetricSemanticEmbedding，适用于结构不同或相似的文本。

- "AlephAlphaAsymmetricSemanticEmbedding"
- "AlephAlphaSymmetricSemanticEmbedding"
- "SentenceTransformerEmbeddings"
- "GooglePalmEmbeddings"

### 4. 自托管嵌入

这类嵌入一般适用于用户自行部署和管理的场景，如 SelfHostedEmbeddings，给予用户更大的灵活性和控制权。

- "SelfHostedEmbeddings"

### 5. 仿真或测试用嵌入

FakeEmbeddings 一般用于测试或模拟场景，不涉及实际的嵌入计算。

- "FakeEmbeddings"

### 6. 其他类型

此外，LangChain 还支持一些其他类型的嵌入方式，如 Cohere、LlamaCpp、ModelScope、TensorflowHub、MosaicMLInstructor、MiniMax、Bedrock、DashScope 和 Embaas 等。这些嵌入方式各有特点，能够满足不同的文本处理需求。

在以上这些类型中，用户可以根据自己的具体需求，选择最合适的文本嵌入类型。同时，LangChain 也将持续引入更多的嵌入类型，以进一步提升其处理文本的能力。
# 文本嵌入模型的应用

我们将通过以下步骤介绍如何使用这种文本嵌入模型。

### 1.环境设置

首先，我们需要安装 OpenAI 的 Python 包。

```
pip install openai
```

接着，我们需要获取一个 API 密钥来访问 API，这可以通过创建一个账户并访问指定页面来获取。一旦我们拿到密钥，我们需要通过运行指定的代码将其设置为环境变量。

```
export OPENAI_API_KEY="..."
```

如果你不想设置环境变量，你也可以在初始化 OpenAI LLM 类时，通过名为 openai_api_key 的参数直接传入密钥。

```
from langchain.embeddings import OpenAIEmbeddings

embeddings_model = OpenAIEmbeddings(openai_api_key="...")
```
如果你设置环境变量的话，就不用传递参数。

```
from langchain.embeddings import OpenAIEmbeddings

embeddings_model = OpenAIEmbeddings()
```

### 2. 嵌入文本列表

这一步是将一组文本进行嵌入。你只需调用 Embeddings 类的相关方法，传入你要嵌入的文本列表即可。

```
embeddings = embeddings_model.embed_documents(
    [
        "Hi there!",
        "Oh, hello!",
        "What's your name?",
        "My friends call me World",
        "Hello World!"
    ]
)
len(embeddings), len(embeddings[0])
```

```
(5, 1536)
```

### 3. 嵌入单一查询

这一步是为了比较与其他嵌入文本的相似度，将单一的文本进行嵌入。同样的，你只需调用 Embeddings 类的相关方法，传入你要嵌入的文本即可。

```
embedded_query = embeddings_model.embed_query("What was the name mentioned in the conversation?")
embedded_query[:5]
```

```
[0.0053587136790156364,
 -0.0004999046213924885,
 0.038883671164512634,
 -0.003001077566295862,
 -0.00900818221271038]
```

以上便是使用 LangChain 进行文本嵌入的基本步骤和方法，相信通过以上的介绍，你已经对如何使用 LangChain 的文本嵌入模型有了基本的理解和掌握。
# 向量存储的原理和使用

## 向量存储的原理

在处理非结构化数据的存储和检索过程中，最常见的方式之一是将其嵌入，并存储生成的嵌入向量。然后在查询时，将非结构化查询嵌入，并检索与嵌入查询 "最相似" 的嵌入向量。向量存储就是用来存储嵌入数据和执行向量搜索的。

## 如何使用向量存储

下面我们将通过以下步骤介绍如何使用向量存储。

### 1.环境设置

本文展示了与向量存储相关的基本功能。使用向量存储的关键部分是创建要放入其中的向量，这通常是通过嵌入来创建的。因此，在深入了解这个内容之前，建议你先熟悉文本嵌入模型的接口。

本文使用的是 FAISS 向量数据库，该数据库使用了 Facebook AI Similarity Search (FAISS)库。

```
pip install faiss-cpu
```
我们需要使用 OpenAIEmbeddings，所以我们需要获取 OpenAI API Key。

```
import os
import getpass

os.environ['OPENAI_API_KEY'] = getpass.getpass('OpenAI API Key:')
```

```
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS


raw_documents = TextLoader('../../../state_of_the_union.txt').load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
documents = text_splitter.split_documents(raw_documents)

embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(documents, embeddings)
```

### 2.相似度搜索

```
query = "What did the president say about Ketanji Brown Jackson"
docs = db.similarity_search(query)
print(docs[0].page_content)
```
```
    Tonight. I call on the Senate to: Pass the Freedom to Vote Act. Pass the John Lewis Voting Rights Act. And while you’re at it, pass the Disclose Act so Americans can know who is funding our elections.

    Tonight, I’d like to honor someone who has dedicated his life to serve this country: Justice Stephen Breyer—an Army veteran, Constitutional scholar, and retiring Justice of the United States Supreme Court. Justice Breyer, thank you for your service.

    One of the most serious constitutional responsibilities a President has is nominating someone to serve on the United States Supreme Court.

    And I did that 4 days ago, when I nominated Circuit Court of Appeals Judge Ketanji Brown Jackson. One of our nation’s top legal minds, who will continue Justice Breyer’s legacy of excellence.
```
通过向量进行相似度搜索：你也可以使用 similarity_search_by_vector 方法来搜索与给定的嵌入向量相似的文档，这个方法接受一个嵌入向量作为参数，而不是一个字符串。

```
embedding_vector = embeddings.embed_query(query)
docs = db.similarity_search_by_vector(embedding_vector)
```


### 3.异步操作

向量存储通常作为一个需要一些 IO 操作的单独服务运行，因此可能会异步调用。这有助于提升性能，因为你不需要浪费时间等待外部服务的响应。如果你使用的是异步框架，例如 FastAPI，这可能非常重要。

Langchain 支持在向量存储上进行异步操作。所有的方法都可以使用其异步对应方法调用，这些方法前缀为 "a"，表示异步。

Qdrant 是一个向量存储，支持所有的异步操作，因此将在本文中使用。

```
pip install qdrant-client
```
```
from langchain.vectorstores import Qdrant
```

### 4.异步创建向量存储

```
db = await Qdrant.afrom_documents(documents, embeddings, "http://localhost:6333")
```

### 5.相似度搜索

通过向量进行相似度搜索。

```
query = "What did the president say about Ketanji Brown Jackson"
docs = await db.asimilarity_search(query)
print(docs[0].page_content)
```

```
    Tonight. I call on the Senate to: Pass the Freedom to Vote Act. Pass the John Lewis Voting Rights Act. And while you’re at it, pass the Disclose Act so Americans can know who is funding our elections.

    Tonight, I’d like to honor someone who has dedicated his life to serve this country: Justice Stephen Breyer—an Army veteran, Constitutional scholar, and retiring Justice of the United States Supreme Court. Justice Breyer, thank you for your service.

    One of the most serious constitutional responsibilities a President has is nominating someone to serve on the United States Supreme Court.

    And I did that 4 days ago, when I nominated Circuit Court of Appeals Judge Ketanji Brown Jackson. One of our nation’s top legal minds, who will continue Justice Breyer’s legacy of excellence.
```    

### 6.最大边际相关性搜索（MMR）

最大边际相关性是为查询的相似性和所选文档的多样性进行优化。在异步 API 中也支持这个功能。

```
query = "What did the president say about Ketanji Brown Jackson"
found_docs = await qdrant.amax_marginal_relevance_search(query, k=2, fetch_k=10)
for i, doc in enumerate(found_docs):
    print(f"{i + 1}.", doc.page_content, "\n")
```

```
1. Tonight. I call on the Senate to: Pass the Freedom to Vote Act. Pass the John Lewis Voting Rights Act. And while you’re at it, pass the Disclose Act so Americans can know who is funding our elections.

Tonight, I’d like to honor someone who has dedicated his life to serve this country: Justice Stephen Breyer—an Army veteran, Constitutional scholar, and retiring Justice of the United States Supreme Court. Justice Breyer, thank you for your service.

One of the most serious constitutional responsibilities a President has is nominating someone to serve on the United States Supreme Court.

And I did that 4 days ago, when I nominated Circuit Court of Appeals Judge Ketanji Brown Jackson. One of our nation’s top legal minds, who will continue Justice Breyer’s legacy of excellence.

2. We can’t change how divided we’ve been. But we can change how we move forward—on COVID-19 and other issues we must face together.

I recently visited the New York City Police Department days after the funerals of Officer Wilbert Mora and his partner, Officer Jason Rivera.

They were responding to a 9-1-1 call when a man shot and killed them with a stolen gun.

Officer Mora was 27 years old.

Officer Rivera was 22.

Both Dominican Americans who’d grown up on the same streets they later chose to patrol as police officers.

I spoke with their families and told them that we are forever in debt for their sacrifice, and we will carry on their mission to restore the trust and safety every community deserves.

I’ve worked on these issues a long time.

I know what works: Investing in crime preventionand community police officers who’ll walk the beat, who’ll know the neighborhood, and who can restore trust and safety.
```

以上便是使用 LangChain 进行向量存储的基本步骤和方法，相信通过以上的介绍，你已经对如何使用 LangChain 的向量存储有了基本的理解和掌握。
# 检索器的定义和使用

检索器是什么？它与向量存储有什么区别？本文尝试解释检索器的定义以及如何使用。

## 检索器的定义

检索器是一种可以通过非结构化查询返回文档的接口。它比向量存储的应用范围更广。一个检索器并不需要能够存储文档，只需要能够返回（或检索）文档即可。向量存储可以作为检索器的支撑，但是也存在其他类型的检索器。

## 检索器的使用

LangChain 中的 BaseRetriever 类的公开 API 非常简单。

```
from abc import ABC, abstractmethod
from typing import Any, List
from langchain.schema import Document
from langchain.callbacks.manager import Callbacks

class BaseRetriever(ABC):
    ...
    def get_relevant_documents(
        self, query: str, *, callbacks: Callbacks = None, **kwargs: Any
    ) -> List[Document]:
        """Retrieve documents relevant to a query.
        Args:
            query: string to find relevant documents for
            callbacks: Callback manager or list of callbacks
        Returns:
            List of relevant documents
        """
        ...

    async def aget_relevant_documents(
        self, query: str, *, callbacks: Callbacks = None, **kwargs: Any
    ) -> List[Document]:
        """Asynchronously get documents relevant to a query.
        Args:
            query: string to find relevant documents for
            callbacks: Callback manager or list of callbacks
        Returns:
            List of relevant documents
        """
        ...
```        

你可以调用 get_relevant_documents 或者异步的 get_relevant_documents 方法来检索与查询相关的文档，其中 "relevance" 是由你正在调用的特定检索器对象定义的。

我们主要关注的检索器类型是向量存储检索器，我们将重点介绍这个类型。

为了理解什么是向量存储检索器，理解向量存储的概念是非常重要的。所以我们首先来看看向量存储是什么。

默认情况下，LangChain 使用 Chroma 作为向量存储来索引和搜索嵌入。在本文中，我们首先需要安装 chromadb。

```
pip install chromadb
```
我们选择了在文档上进行问题回答的示例来进行演示，因为它很好地整合了许多不同的元素（文本分割器，嵌入，向量存储），并展示了如何将它们串联使用。

文档上的问题回答包括四个步骤：

1. 创建索引
2. 从索引创建检索器
3. 创建问题回答链
4. 提出问题！

每一个步骤都有多个子步骤和潜在的配置。在本指南中，我们将主要关注步骤（1）。我们将首先展示一个一行代码的示例，然后详细解析其中的内容。

首先，让我们导入公共类。

```
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
```

在通用设置中，接下来我们来指定我们想要使用的文档加载器。

```
from langchain.document_loaders import TextLoader
loader = TextLoader('../state_of_the_union.txt', encoding='utf8')
```

为了尽快开始，我们可以使用 VectorstoreIndexCreator。

```
from langchain.indexes import VectorstoreIndexCreator
```

```
index = VectorstoreIndexCreator().from_loaders([loader])
```

```
    Running Chroma using direct local API.
    Using DuckDB in-memory for database. Data will be transient.
```    

创建索引后，我们就可以使用它来提出数据问题了！注意，这里面实际上也做了几个步骤，我们将在后面的指南中介绍。

```
query = "What did the president say about Ketanji Brown Jackson"
index.query(query)
```

```
    " The president said that Ketanji Brown Jackson is one of the nation's top legal minds, a former top litigator in private practice, a former federal public defender, and from a family of public school educators and police officers. He also said that she is a consensus builder and has received a broad range of support from the Fraternal Order of Police to former judges appointed by Democrats and Republicans."
```

```
query = "What did the president say about Ketanji Brown Jackson"
index.query_with_sources(query)
```

```
    {'question': 'What did the president say about Ketanji Brown Jackson',
     'answer': " The president said that he nominated Circuit Court of Appeals Judge Ketanji Brown Jackson, one of the nation's top legal minds, to continue Justice Breyer's legacy of excellence, and that she has received a broad range of support from the Fraternal Order of Police to former judges appointed by Democrats and Republicans.\n",
     'sources': '../state_of_the_union.txt'}
```


VectorstoreIndexCreator 返回的是 VectorStoreIndexWrapper，它提供了查询和带来源的查询功能。
如果我们只想直接访问向量存储，我们也可以这样做。

```
index.vectorstore
```

```
    <langchain.vectorstores.chroma.Chroma at 0x119aa5940>
```

如果我们只想直接访问 VectorstoreRetriever，我们也可以这样做。

```
index.vectorstore.as_retriever()
```

```
    VectorStoreRetriever(vectorstore=<langchain.vectorstores.chroma.Chroma object at 0x119aa5940>, search_kwargs={})
```   

接下来，我们要深入了解一下索引是如何创建的。VectorstoreIndexCreator 中隐藏了很多魔法。它做了什么呢？

在文档加载后，主要进行了三个步骤：

1. 将文档分割成块
2. 为每个文档创建嵌入
3. 在向量存储中存储文档和嵌入

让我们在代码中一步步解析这个过程。

加载文档。

```
documents = loader.load()
```
分割文档。

```
from langchain.text_splitter import CharacterTextSplitter
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)
```
创建嵌入。

```
from langchain.embeddings import OpenAIEmbeddings
embeddings = OpenAIEmbeddings()
```
创建向量存储
```
from langchain.vectorstores import Chroma
db = Chroma.from_documents(texts, embeddings)
```

```
    Running Chroma using direct local API.
    Using DuckDB in-memory for database. Data will be transient.
```
创建索引。

```
retriever = db.as_retriever()
```

我们像以前一样，创建问答链。

```
qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=retriever)
```

```
query = "What did the president say about Ketanji Brown Jackson"
qa.run(query)
```

```
    " The President said that Judge Ketanji Brown Jackson is one of the nation's top legal minds, a former top litigator in private practice, a former federal public defender, and from a family of public school educators and police officers. He said she is a consensus builder and has received a broad range of support from organizations such as the Fraternal Order of Police and former judges appointed by Democrats and Republicans."
```   

VectorstoreIndexCreator 不过是一种封装了各种逻辑的工具。它的配置是灵活的，可以设定使用的文本分割器、嵌入方式以及使用的向量存储。例如，你可以按照以下方式对其进行配置：

```
index_creator = VectorstoreIndexCreator(
    vectorstore_cls=Chroma,
    embedding=OpenAIEmbeddings(),
    text_splitter=CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
)
```

我们希望能揭示出 VectorstoreIndexCreator 内部的运行机制。我们认为提供一种简单的索引创建方式是重要的，但同时，理解这个过程中底层的操作也同样关键。

# 检索器的应用

在高维空间中，距离型向量数据库检索工具通过将查询嵌入来找出相似的文档。然而，微妙的查询词汇改变或者嵌入不能很好地捕捉数据语义时，检索结果可能会有所不同。人们有时会进行提示工程或者调优来手动解决这些问题，但这样做可能很繁琐。

多查询检索器（MultiQueryRetriever）自动化了提示调优的过程，它使用大语言模型（LLM）从不同角度为给定的用户输入查询生成多个查询。对于每个查询，它都会检索一组相关文档，并取所有查询的唯一并集，以得到一组可能相关的更大的文档集。通过从多个角度生成对同一个问题的视角，多查询检索器可能能够克服基于距离的检索的一些局限性，得到更丰富的结果集。

当然，检索过程也面临一些挑战。通常在将数据录入系统时，你无法知道文档存储系统将面临的具体查询。这意味着与查询最相关的信息可能被埋在大量无关文本的文档中。将完整的文档传递给应用程序可能会导致更昂贵的 LLM 调用和较差的响应。

上下文压缩（Contextual compression）就是为了解决这个问题。其思想很简单：不是立即原样返回检索到的文档，而是可以使用给定查询的上下文来压缩它们，这样只有相关的信息会被返回。这里的“压缩”既指压缩单个文档的内容，也指整体过滤掉文档。

此外，我们还有自我查询检索器（Self-querying retriever），它能够自我查询。特别地，给定任何自然语言查询，检索器使用查询构造的 LLM 链来编写结构化查询，然后将该结构化查询应用到其底层的 VectorStore。这使得检索器不仅可以使用用户输入的查询来与存储文档的内容进行语义相似性比较，还可以从用户查询中提取有关存储文档元数据的过滤器并执行这些过滤器。

然后是时间加权向量存储检索器（Time-weighted vector store retriever），它使用语义相似性和时间衰减的组合。

最后，我们还有向量存储支持的检索器（Vector store-backed retriever），它是一种使用向量存储来检索文档的检索器。它是围绕 Vector Store 类的轻量级封装，使其符合检索器接口。它使用由向量存储实现的搜索方法，如相似性搜索和 MMR，来查询向量存储中的文本。

以上这些检索器都有各自独特的使用场景和优势，能够有效应对各种复杂的信息检索任务。
# 自查询检索器原理与分类

##  自查询检索器的概念

自查询检索器，顾名思义，是具有自我查询能力的检索器。具体来说，给定任何自然语言查询，检索器使用一个查询构造的大语言模型（LLM）链来编写结构化查询，然后将该结构化查询应用于其底层的向量存储。这使得检索器不仅可以使用用户输入的查询与存储文档内容进行语义相似性比较，而且还可以从用户查询中提取关于存储文档元数据的过滤器，并执行这些过滤器。

## 自查询检索器的分类

### 1. Chroma 自查询检索器

Chroma 是一个用于构建嵌入式 AI 应用程序的数据库。在我们的示例中，我们将展示一个围绕 Chroma 向量存储的自查询检索器。

### 2. MyScale 自查询检索器

MyScale 是一个集成向量数据库。你可以通过 SQL 和 LangChain 访问你的数据库。MyScale 可以利用各种数据类型和函数进行过滤。无论你是扩大数据规模还是将系统扩展到更广泛的应用，MyScale 都可以提升你的 LLM 应用的性能。

### 3. Pinecone 自查询检索器

在我们的演示中，我们将展示一个与 Pinecone 向量存储一起使用的自查询检索器。

### 4. Qdrant 自查询检索器

Qdrant 是一个向量相似性搜索引擎。它提供了一个生产就绪的服务，具有便捷的 API 来存储、搜索和管理点 - 带有额外负载的向量。Qdrant 针对扩展过滤支持进行了优化，使其更加实用。在我们的示例中，我们将展示一个围绕 Qdrant 向量存储的自查询检索器。

### 5. Weaviate 自查询检索器

创建 Weaviate 向量存储是首要步骤，我们希望为其添加一些数据。我们创建了一个包含电影摘要的小型演示文档集。

注意：自查询检索器需要你安装 lark（使用 pip install lark 命令进行安装）。我们还需要 weaviate-client 包。

以上就是自查询检索器的基本概念和分类，希望能对你的学习和理解有所帮助。
# 应用检索器

LangChain 提供了广泛的检索器支持，以满足各种不同的需求。以下是 LangChain 目前支持的检索器的分类：

1. ArxivRetriever：专门用于检索 Arxiv 这类科学预印本数据库中的文档。

2. AwsKendraIndexRetriever：基于亚马逊的 Kendra 索引服务的检索器。

3. AzureCognitiveSearchRetriever：基于微软 Azure Cognitive Search 的检索器。

4. ChatGPTPluginRetriever：与 ChatGPT 插件相关的检索器。

5. ContextualCompressionRetriever：实现上下文压缩功能的检索器。

6. DataberryRetriever：可能是基于 Databerry 数据管理平台的检索器。

7. ElasticSearchBM25Retriever：基于 Elasticsearch 和 BM25 算法的检索器。

8. KNNRetriever：基于最近邻算法的检索器。

9. LlamaIndexGraphRetriever 和 LlamaIndexRetriever：基于 Llama Index 图数据库的检索器。

10. MergerRetriever：整合多个检索器结果的检索器。

11. MetalRetriever：可能是针对特定数据集（如金属相关数据）的检索器。

12. MilvusRetriever：基于 Milvus 向量搜索引擎的检索器。

13. PineconeHybridSearchRetriever：结合 Pinecone 混合搜索的检索器。

14. PubMedRetriever：专门用于检索 PubMed（生物医学文献数据库）的检索器。

15. RemoteLangChainRetriever：远程调用 LangChain 的检索器。

16. SVMRetriever：基于支持向量机（SVM）算法的检索器。

17. SelfQueryRetriever：能够自我查询的检索器。

18. TFIDFRetriever：基于 TF-IDF 算法的检索器。

19. TimeWeightedVectorStoreRetriever：引入时间权重的向量存储检索器。

20. VespaRetriever：基于 Vespa 搜索引擎的检索器。

21. WeaviateHybridSearchRetriever：结合 Weaviate 混合搜索的检索器。

22. WikipediaRetriever：专门用于检索 Wikipedia 的检索器。

23. ZepRetriever：可能是与 Zep 团队合作开发的检索器。

24. ZillizRetriever：基于 Zilliz 数据分析平台的检索器。

25. DocArrayRetriever：处理文档数组的检索器。

以上检索器的名称和功能可能根据具体实现有所不同，但它们共同的目标都是提供高效准确的信息检索服务。
