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
        self.SYSTEMPL = ""  # 系统模板
        self.prompt = ChatPromptTemplate.from_messages([
            ("system","你是一个助理"),
            ("user","{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),  # 👈 加这一行
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
