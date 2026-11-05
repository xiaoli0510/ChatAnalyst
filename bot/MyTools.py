from langchain_community.utilities import SerpAPIWrapper
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_community.embeddings import DashScopeEmbeddings
from dotenv import load_dotenv
import os
import requests


load_dotenv()

@tool
def search(query: str) -> str:
    """只有需要了解实时的信息或者不知道的事情的时候才会使用这个工具"""
    serp = SerpAPIWrapper()
    result = serp.run(query)
    print("这是搜索的结果",result)
    return result

@tool
def get_info_from_local_db(query: str) -> str:
    """当用户询问关于2024年运势、龙年运势、生肖运势、财运、事业、感情等算命相关的问题时，必须优先使用此工具查询本地知识库。"""
    client = QdrantClient(path="./local_qdrand1")

    if not client.collection_exists("local_documents1"):
        client.create_collection(
            collection_name="local_documents1",
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
        )

    vector_store = QdrantVectorStore(
        client=client,
        collection_name="local_documents1",
        embedding=DashScopeEmbeddings(model="text-embedding-v2"),
    )
    # 强行塞入数据，保证库不是空的
    if client.count(collection_name="local_documents1").count == 0:
        from langchain_core.documents import Document
        vector_store.add_documents([
            Document(page_content="2024年龙年，属龙的人财运非常旺盛，正财偏财皆有收获，但需注意理财，避免冲动消费。")
        ])

    retriever = vector_store.as_retriever(search_type="mmr")
    result = retriever.invoke(query)
    print("这是检索数据库的result",result)
    if not result:
        return "本地知识库中没有找到相关信息。"
    return "\n\n".join([doc.page_content for doc in result])


@tool
def bazi_cesuan(query: str):
    """只有做八字排盘的时候才会使用这个工具，需要输入用户姓名和出生年月日时，如果缺少姓名和出生年月日时则不能使用    """
    url = "https://api.yuanfenju.com/index.php/v1/Bazi/pan"
    BAZI_API_KEY = os.getenv("BAZI_API_KEY")
    prompt = ChatPromptTemplate.from_template(
        f"""你是一个参数查询助手，根据用户输入内容找出相关的参数并按json格式返回.
        JSON字段如下：
        'api_key' => '{BAZI_API_KEY}',
        'name' => '张三',
        'sex' => '1',
        'type' => '1',
        'year' => '1988',
        'month' => '11',
        'day' => '8',
        'hours' => '12',
        'minute' => '20',
       
        【重要规则】：
        1. 如果用户性别是【男】，sex 字段返回 '1'；如果性别是【女】，sex 字段返回 '0'。
        2. type 字段代表历法，公历（阳历）返回 '0'，农历（阴历）返回 '1'。如果不确定默认给 '0'。
        3. 必须包含上面的 api_key。
        4. 如果没有找到相关参数（如缺少姓名或出生时间），请直接告诉用户缺少什么。
        5. 只返回标准的 JSON 数据结构，不要有其他评论。
        
        如果没有找到相关参数，则需要提醒用户告诉你这些内容，只返回数据结构，不要有其他的评论和内容，用户输入{query}
        """)
    parser = JsonOutputParser()
    prompt = prompt.partial(format_instructions=parser.get_format_instructions())
    llm = ChatOpenAI(
        model="qwen-max",
        temperature=0,
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    chain = prompt | llm | parser
    data = chain.invoke({"query": query})

    print("八字查询:",data)
    try:
        result = requests.post(url, data=data, headers={"Content-Type": "application/x-www-form-urlencoded"})
        print(result)
        if result.status_code == 200:
            json = result.json()
            print("获取到的八字json是：",json)
            return str(json["data"]["base_info"])
        else:
            return f"获取八字异常，状态码:{result.status_code}"
    except Exception as e:
        print(f"八字请求报错了，{e}")
        return "八字请求异常"

@tool
def yaoyiguan():
    """只有用户想要占卜抽签的时候才会使用这个工具。"""
    url="https://api.yuanfenju.com/index.php/v1/Zhanbu/meiri"
    result = requests.post(url, data={"api_key": os.getenv("BAZI_API_KEY")},  headers={"Content-Type": "application/x-www-form-urlencoded"})
    if result.status_code == 200:
        json = result.json()
        print("摇一卦的json结果",json)
        return str(json["data"])
    else:
        return f"请求yaoyiguan异常，状态码:{result.status_code}"

@tool
def jiemeng(query: str):
    """只有解梦的时候才会使用这个工具，通过输入梦境关键字来查询梦境释意"""
    url = "https://api.yuanfenju.com/index.php/v1/Gongju/zhougong"
    prompt = ChatPromptTemplate.from_template("根据内容提取一个关键词，只返回一个关键词(必须使用简体中文)，，内容为{topic}")
    prompt_value =prompt.invoke({"topic": query})
    llm = ChatOpenAI(
        model="qwen-max",
        temperature=0,
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    keyword = llm.invoke(prompt_value)
    print("提取的关键词是：",keyword.content)

    result = requests.post(url, data={"api_key": os.getenv("BAZI_API_KEY"),"title_zhougong": keyword.content},
                           headers={"Content-Type": "application/x-www-form-urlencoded"})
    if result.status_code == 200:
        json = result.json()
        print("解梦的结果", json)
        return str(json["data"][:2])
    else:
        return f"请求解梦异常，状态码:{result.status_code}"



