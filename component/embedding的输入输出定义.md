最后一种类型是 Embeddings文本嵌入组件。这类组件的主要功能是将文本信息转化为机器能够理解的数字化表示形式，即向量，可以用于各种自然语言处理的任务，如文本分类、语义理解等。
创建 Embeddings这种类型组件的主要目的是为了适配多种嵌入模型提供者，如 OpenAI、Cohere、Hugging Face 等。文本嵌入模型的功能是将文本转换成向量，这是非常有用的，因为这意味着我们可以在向量空间中考虑文本，并做一些诸如语义搜索的事情，寻找在向量空间中最相似的文本片段。
至2023年7月，LangChain 已经实现了22种针对不同文本嵌入模型的 "Embeddings" 类型的包装器，主要包括：
- OpenAIEmbeddings：用于包装 OpenAI 的文本嵌入模型
- HuggingFaceEmbeddings：用于包装 Hugging Face 的文本嵌入模型
- CohereEmbeddings：用于包装 Cohere 的文本嵌入模型
- LlamaCppEmbeddings：用于包装 LlamaCpp 的文本嵌入模型
- HuggingFaceHubEmbeddings：用于包装 HuggingFaceHub 的文本嵌入模型
- ModelScopeEmbeddings：用于包装 ModelScope 的文本嵌入模型
- TensorflowHubEmbeddings：用于包装 TensorflowHub 的文本嵌入模型
- GooglePalmEmbeddings：用于包装 GooglePalm 的文本嵌入模型
- MiniMaxEmbeddings：用于包装 MiniMax 的文本嵌入模型
这些包装器都是 BaseEmbeddings 的子类，继承了 BaseEmbeddings 的所有属性和方法，并根据需要添加或覆盖了一些自己的方法。例如，如果你想要使用 OpenAI 的文本嵌入模型，你可以选择使用 OpenAIEmbeddings 包装器，如下所示：

from langchain.embeddings import OpenAIEmbeddings
emb = OpenAIEmbeddings(openai_api_key="my-api-key")

所有通过 langchain.embeddings获取的所有对象都是Embeddings类型，这些对象我们称为Embeddings 类包装器。在这里， emb 是 OpenAIEmbeddings 类的一个实例，它继承了 OpenAIEmbeddings 类的所有属性和方法，你可以使用这个 emb 对象来调用 OpenAI 的文本嵌入模型的功能。
