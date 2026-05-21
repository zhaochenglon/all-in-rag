import os
from pathlib import Path

# Hugging Face 镜像（国内下载嵌入模型失败时取消下一行注释）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

_SCRIPT_DIR = Path(__file__).resolve().parent
load_dotenv(_SCRIPT_DIR / ".env")
load_dotenv()  # 也支持项目根目录或系统环境变量

api_key = os.getenv("DEEPSEEK_API_KEY")
if not api_key:
    raise ValueError(
        "未找到 DEEPSEEK_API_KEY。请在 code/C1/.env 中设置，"
        "或参考 .env.example 配置后重试。"
    )

markdown_path = _SCRIPT_DIR / "../../data/C1/markdown/easy-rl-chapter1.md"

# 加载本地 markdown 文件
loader = TextLoader(str(markdown_path.resolve()), encoding="utf-8")
docs = loader.load()

# 文本分块
text_splitter = RecursiveCharacterTextSplitter()
chunks = text_splitter.split_documents(docs)

# 输出每个分块的内容（同时写入文件，便于查看）
chunks_output_path = _SCRIPT_DIR / "chunks_output.txt"
with open(chunks_output_path, "w", encoding="utf-8") as f:
    header = (
        f"共 {len(chunks)} 个块 | "
        f"chunk_size={text_splitter._chunk_size}, "
        f"chunk_overlap={text_splitter._chunk_overlap}\n"
    )
    print(header)
    f.write(header + "\n")
    for i, chunk in enumerate(chunks, start=1):
        content = chunk.page_content
        block = (
            f"{'=' * 60}\n"
            f"Chunk {i}/{len(chunks)} | 长度: {len(content)} 字符\n"
            f"{'=' * 60}\n"
            f"{content}\n"
        )
        print(block)
        f.write(block + "\n")
print(f"完整分块已保存至: {chunks_output_path.resolve()}\n")

# 中文嵌入模型
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)
  
# 构建向量存储
vectorstore = InMemoryVectorStore(embeddings)
vectorstore.add_documents(chunks)

# 提示词模板
prompt = ChatPromptTemplate.from_template("""请根据下面提供的上下文信息来回答问题。
请确保你的回答完全基于这些上下文。
如果上下文中没有足够的信息来回答问题，请直接告知：“抱歉，我无法根据提供的上下文找到相关信息来回答此问题。”

上下文:
{context}

问题: {question}

回答:"""
                                          )

# 配置大语言模型

llm = ChatOpenAI(
    model="deepseek-chat",
    temperature=0.7,
    max_tokens=4096,
    api_key=api_key,
    base_url="https://api.deepseek.com/v1",
)

# 用户查询
question = "文中举了哪些例子？"

# 在向量存储中查询相关文档
retrieved_docs = vectorstore.similarity_search(question, k=3)
docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)

chain = prompt | llm
answer = chain.invoke({"question": question, "context": docs_content})
print(answer.content)
