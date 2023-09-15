


Langchain的核心模块——模型I/O模块、Chain链模块、记忆模块、数据连接模块以及Agent模块。
模型I/O模块由


Langchain提供了一个{模块名称}模块，可以无缝地执行聊天机器人的大部分繁重工作：(i) 请求模型I/O模块的LLM类包装器解释用户的输入问题并生成一个辅助SQL查询，(ii) 在数据库上执行所述SQL查询，和(iii) 请求模型I/O模块的LLM类包装器生成一个自然语言的答案；开发者只需用输入问题调用此模块API并将答案回传给用户。

### 转提示词模板：

请将原始的 {"Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.\n\nIn addition to giving an answer, also return a score of how fully it answered the user's question. This should be in the following format:\n\nQuestion: [question here]\nHelpful Answer: [answer here]\nScore: [score between 0 and 100]\n\nHow to determine the score:\n- Higher is a better answer\n- Better responds fully to the asked question, with sufficient level of detail\n- If you do not know the answer based on the context, that should be a score of 0\n- Don't be overconfident!\n\nExample #1\n\nContext:\n---------\nApples are red\n---------\nQuestion: what color are apples?\nHelpful Answer: red\nScore: 100\n\nExample #2\n\nContext:\n---------\nit was night and the witness forgot his glasses. he was not sure if it was a sports car or an suv\n---------\nQuestion: what type was the car?\nHelpful Answer: a sports car or an suv\nScore: 60\n\nExample #3\n\nContext:\n---------\nPears are either red or orange\n---------\nQuestion: what color are apples?\nHelpful Answer: This document does not answer the question\nScore: 0\n\nBegin!\n\nContext:\n---------\n{context}\n---------\nQuestion: {question}\nHelpful Answer:"} 转换为 格式化这样的：{1 你是一个PostgreSQL专家。给定一个输入问题，首先创建一个
↩→ 语法正确的PostgreSQL查询来运行，然后查看查询的
↩→ 结果，并返回对输入问题的答案。
2 除非用户在问题中明确指定要获得的特定数量的示例，
↩→ 否则使用LIMIT子句按照PostgreSQL查询最多{top_k}的结果。
↩→ 你可以对结果进行排序，以返回数据库中的最有信息的数据。
3 绝不要查询表中的所有列。你只能查询回答问题所需的
↩→ 列。用双引号(")将每个列名包起来，表示它们是界定的标识符。
4 注意只使用你在下面的表中可以看到的列名。小心不要查询
↩→ 不存在的列。此外，注意哪一列在哪个表中。
5 如果问题涉及“今天”，请注意使用CURRENT_DATE函数获取当前日期。
6
7 使用以下格式：
8
9 问题：这里的问题
10 SQL查询：要运行的SQL查询
11 SQL结果：SQL查询的结果
12 答案：这里的最终答案
13
14 只使用以下表格：
15
16 {table_info}
17
18 问题：{input}
}

编号指的是每一行的行号。请重新编辑给我。每一行都要标注行号，空行也需要标注。你的回答是：

请将原始的提示词转为我给的模板的格式和分析，将原始的转为模板式样格式，以及写解释。 原始的提示词是："Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.\n\nIn addition to giving an answer, also return a score of how fully it answered the user's question. This should be in the following format:\n\nQuestion: [question here]\nHelpful Answer: [answer here]\nScore: [score between 0 and 100]\n\nHow to determine the score:\n- Higher is a better answer\n- Better responds fully to the asked question, with sufficient level of detail\n- If you do not know the answer based on the context, that should be a score of 0\n- Don't be overconfident!\n\nExample #1\n\nContext:\n---------\nApples are red\n---------\nQuestion: what color are apples?\nHelpful Answer: red\nScore: 100\n\nExample #2\n\nContext:\n---------\nit was night and the witness forgot his glasses. he was not sure if it was a sports car or an suv\n---------\nQuestion: what type was the car?\nHelpful Answer: a sports car or an suv\nScore: 60\n\nExample #3\n\nContext:\n---------\nPears are either red or orange\n---------\nQuestion: what color are apples?\nHelpful Answer: This document does not answer the question\nScore: 0\n\nBegin!\n\nContext:\n---------\n{context}\n---------\nQuestion: {question}\nHelpful Answer:"





请按照以下格式，编写这个清单列表的解释。

比如我给出清单列表：{
    1. 当你面对以下的背景信息时，如何回答最后的问题是关键。如果不知道答案，直接说你不知道，不要试图编造答案。
2. 
3. 除了提供答案外，还需要给出一个分数，表示它如何完全回答了用户的问题。请按照以下格式：
4. 
5. 问题：[这里的问题]
6. 
7. 有帮助的答案：[这里的答案]
8. 
9. 分数：[分数范围在0到100之间]
10. 
11. 如何确定分数：
12.    - 更高的分数代表更好的答案
13.    - 更好的答案能够充分地回应所提出的问题，并提供足够的细节
14.    - 如果根据上下文不知道答案，那么分数应该是0
15.    - 不要过于自信！
16. 
17. 示例 #1
18. 
19. 背景：
20.    - 苹果是红色的
21. 
22. 问题：苹果是什么颜色？
23. 
24. 有帮助的答案：红色
25. 
26. 分数：100
27. 
28. 示例 #2
29. 
30. 背景：
31.    - 那是夜晚，证人忘了带他的眼镜。他不确定那是一辆跑车还是SUV
32. 
33. 问题：那辆车是什么类型的？
34. 
35. 有帮助的答案：跑车或SUV
36. 
37. 分数：60
38. 
39. 示例 #3
40. 
41. 背景：
42.    - 梨要么是红色的，要么是橙色的
43. 
44. 问题：苹果是什么颜色？
45. 
46. 有帮助的答案：这个文档没有回答这个问题
47. 
48. 分数：0
49. 
50. 开始！
51. 
52. 背景：
53.    - {context}
54. 
55. 问题：{question}
56. 
57. 有帮助的答案：

}

编写的解释是这样：

{为了确保语言模型能够在接到问题后提供准确和有用的答案，我们为模型设计了一套详细的指南。该指南描述了如何根据给定的背景信息回答问题，并如何为答案打分。这部分强调了整体目标：使模型能够根据给定的背景信息提供准确答案，并为其答案打分。（行1-3）。

首先，模型需要明白其核心任务：根据给定的背景信息回答问题。如果模型不知道答案，它应直接表示不知道，而不是试图编造答案。这部分提醒模型，如果不知道答案，应该直接表示不知道，而不是编造答案。（行4-6）。

接下来，为模型提供了答案和评分的标准格式。答案部分要求模型简洁、明确地回答问题，而评分部分则要求模型为其答案给出一个0到100的分数，用以表示答案的完整性和准确性。这部分明确了答案和评分的格式，并强调了答案的完整性和准确性。（行7-10）。

此处，强调了答案的完整性和准确性是评分的核心标准，并为模型提供了三个示例来进一步说明如何评分。通过三个示例，模型可以更好地理解如何根据答案的相关性和准确性为其打分。（行11-15）。

最后，为了使模型能够在具体的实践中应用上述指南，为模型提供了一个背景和问题的模板。当模型接到一个问题时，它应使用此模板为问题提供答案和评分。这部分提供了实际应用的模板，确保模型在实际操作中能够遵循上述指南。（行16-20）。}

现在清单是：
{
    ↩→ 语法正确的PostgreSQL查询来运行，然后查看查询的
↩→ 结果，并返回对输入问题的答案。
2 除非用户在问题中明确指定要获得的特定数量的示例，
↩→ 否则使用LIMIT子句按照PostgreSQL查询最多{top_k}的结果。
↩→ 你可以对结果进行排序，以返回数据库中的最有信息的数据。
3 绝不要查询表中的所有列。你只能查询回答问题所需的
↩→ 列。用双引号(")将每个列名包起来，表示它们是界定的标识符。
4 注意只使用你在下面的表中可以看到的列名。小心不要查询
↩→ 不存在的列。此外，注意哪一列在哪个表中。
5 如果问题涉及“今天”，请注意使用CURRENT_DATE函数获取当前日期。
6
7 使用以下格式：
8
9 问题：这里的问题
10 SQL查询：要运行的SQL查询
11 SQL结果：SQL查询的结果
12 答案：这里的最终答案
13
14 只使用以下表格：
15
16 {table_info}
17
18 问题：{input}

}




请参考我给的案例，将以下内容，更加清晰的解释代码发生了什么，获得了什么结果，是什么原因。
案例是：{这里我们引入 OpenAI 模型包装器：

from langchain.llms import OpenAI
model = OpenAI(temperature=0)

我们将 subject 的值设为 "ice cream flavors"，然后调用 prompt.format(subject="ice cream flavors") 方法，这将返回一个完整的提示字符串，包含指导模型产生五种冰淇淋口味的指令。

接下来，我们将这个提示传给 OpenAI 模型，让模型根据这个提示生成一个响应。

output = model(_input)

最后，我们调用输出解析器的 parse(output) 方法，将模型输出的字符串解析为一个列表。由于我们的输出解析器是 CommaSeparatedListOutputParser，所以它会将模型输出的逗号分隔的文本解析为列表。

output_parser.parse(output)

所以，最后得到的结果是一个包含五种冰淇淋口味的列表：

['Vanilla', 'Chocolate', 'Strawberry', 'Mint Chocolate Chip', 'Cookies and Cream']}
请参照案例，不要使用有序列表无序列表，只需要使用段落，注意不要ＭＤ格式语法，只要word格式。需要改造的内容是：{}

