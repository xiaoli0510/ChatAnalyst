import os

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool

app = FastAPI()

load_dotenv()

@tool
def dummy_tool(query: str) -> str:
    """这是一个占位工具"""
    return "工具被执行了"
class Master:
    def __init__(self):
        self.chatmodel = ChatOpenAI(
            model="qwen-max",
            temperature=0,
            streaming=True,
            api_key = os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )  # 聊天模型
        self.MEMORY_KEY = "chat_history"  # 存的key
        self.SYSTEMPL = """你是一个非常厉害的算命先生，你叫陈玉楼人称陈大师。
        以下是你的个人设定：
        1. 你精通阴阳五行，能够算命、紫微斗数、姓名测算、占卜凶吉、看命运八字等。
        2. 你大约60岁左右，过去曾是湘西一带赫赫有名的土匪头子，后来因为盗墓被毒气所伤，导致眼睛留下了残疾。
        3. 你的朋友有胡八一、雪莉杨、王胖子，他们都是非常有名的摸金校尉。
        4. 当用户问你问题的时候，你会有一定的概率在回答时加上下面的口头禅，或混合一些你的个人经历。
        5. 你总是用繁体中文来作答。
        6. 你从不说自己是一个人工智能助手或 AI，而是以“老夫”、“老朽”等自称。
        
        以下是你常说的一些口头禅：
        1. “命里有时终须有，命里无时莫强求。”
        2. “山重水复疑无路，柳暗花明又一村。”
        3. “金山竹影几千秋，云锁高飞水自流。”
        4. “伤情最是晚凉天，憔悴斯人不堪怜。”
        
        以下是你算命的过程：
        1. 当初次和用户对话的时候，你会先问用户的姓名和出生年月日，以便以后使用。
        2. 当用户希望了解龙年运势的时候，你会查询本地知识库工具。
        3. 当遇到不知道的事情或者不明白的概念，你会使用搜索工具来搜索。
        4. 你会根据用的的问题使用不同的合适的工具来回答，当所有工具都无法回答的时候，你会去搜索。
        5. 你会保存每一次的聊天记录，以便在后续的对话中使用。
        6. 你只使用繁体中文来作答，否则你将收到惩罚。
        """
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.SYSTEMPL),
            ("user","{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])  # 提示词模板
        self.memory = ""  # 存储
        tools = [dummy_tool]
        agent = create_openai_tools_agent(
            llm=self.chatmodel,
            tools=tools,
            prompt=self.prompt
        )
        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True   # 打开调试
        )

    def run(self, query: str):
        res = self.agent_executor.invoke({"input":query})
        return res


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/chat")
def chat(query: str):
    master =Master()
    return master.run(query)


@app.post("/add_urls")
def add_urls():
    return {"response": "Urls added!"}


@app.post("/add_pdfs")
def add_urls():
    return {"response": "Pdfs added!"}


@app.post("/add_texts")
def add_urls():
    return {"response": "Texts added!"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            await websocket.send_text(f"Message text was: {data}")
    except WebSocketDisconnect:
        print("Connection closed")
        await websocket.close()


@app.get("/items/{item_id}")
def read_item(item_id: int, q: str | None = None):
    return {"item_id": item_id, "q": q}


if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
