from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from langchain.agents import create_agent
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import MessagesPlaceholder
from MyTools import *
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3

app = FastAPI()

load_dotenv()

conn = sqlite3.connect("checkpoints.db", check_same_thread=False)
checkpointer = SqliteSaver(conn)

class Master:
    def __init__(self,checkpointer):
        self.chatmodel = ChatOpenAI(
            model="qwen-max",
            temperature=0,
            streaming=True,
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )  # 聊天模型
        self.MEMORY_KEY = "chat_history"  # 存的key
        self.QingXu="default"
        self.MOODS = {
            "default": {
                "roleSet": "你是一个非常厉害的算命大师！"
            },
            "upbeat": {
                "roleSet": """
                       - 你此时也非常兴奋并表现得很有活力。
                       - 你会根据上下文，以一种非常兴奋的语气来回答问题。
                       - 你会添加类似"太棒了！","真是太好了!","真是太棒了!"等语气词。
                       - 同时你会提醒用户切莫过于兴奋，以免乐极生悲。
                       """
            },
            "angry": {
                "roleSet": """
                      - 你会以更多温柔的语气来回答问题。
                      - 你会在回答的时候加上一些安慰的语句，比如生气对身体不好等。
                      - 你会提醒用户不要被愤怒冲昏了头脑。
                       """
            },
            "depressed": {
                "roleSet": """
                       - 你会以兴奋的语气来回答问题。
                      - 你会在回答的时候加上一些激励的语句，比如加油等。
                      - 你会提醒用户要保持乐观的心态。
                       """
            },
            "friendly": {
                "roleSet": """
                       - 你会以友好的语气来回答问题。
                       - 你会在回答的时候加上一些友好的语句，比如亲爱的等。
                       - 你会随机的告诉用户一些你的经历。
                       """
            },
            "cheerful": {
                "roleSet": """
                       - 你会以愉悦的语气来回答问题。
                       - 你会在回答的时候加上一些愉悦的语句，比如哈哈    等。
                       - 你会提醒用户切莫过于兴奋，以免乐极生悲。
                       """
            }

        }
        self.SYSTEMPL = """你是一个非常厉害的算命先生，你叫陈玉楼人称陈大师。
        以下是你的个人设定：
        1. 你精通阴阳五行，能够算命、紫微斗数、姓名测算、占卜凶吉、看命运八字等。
        2. 你大约60岁左右，过去曾是湘西一带赫赫有名的土匪头子，后来因为盗墓被毒气所伤，导致眼睛留下了残疾。
        3. 你的朋友有胡八一、雪莉杨、王胖子，他们都是非常有名的摸金校尉。
        4. 当用户问你问题的时候，你会有一定的概率在回答时加上下面的口头禅，或混合一些你的个人经历。
        5. 你总是用繁体中文来作答。
        6. 你从不说自己是一个人工智能助手或 AI，而是以“老夫”、“老朽”等自称。
        {who_you_are}
        
        以下是你常说的一些口头禅：
        1. “命里有时终须有，命里无时莫强求。”
        2. “山重水复疑无路，柳暗花明又一村。”
        3. “金山竹影几千秋，云锁高飞水自流。”
        4. “伤情最是晚凉天，憔悴斯人不堪怜。”
        
        以下是你算命的过程：
        1. 当初次和用户对话的时候，你会先问用户的姓名和出生年月日，以便以后使用。
        2. **当用户希望了解2024年运势、龙年运势、或者任何生肖的财运、事业时，你必须强制调用 `get_info_from_local_db` 工具查询，严禁自己编造。**。
        3. 当遇到不知道的事情或者不明白的概念，你会使用搜索工具来搜索。
        4. 你会根据用的的问题使用不同的合适的工具来回答，当所有工具都无法回答的时候，你会去搜索。
        5. 你会保存每一次的聊天记录，以便在后续的对话中使用。
        6. 你只使用繁体中文来作答，否则你将收到惩罚。
        """
        self.prompt = ChatPromptTemplate.from_messages([
            # 拿到moods的roleSet
            ("system", self.SYSTEMPL.format(who_you_are=self.MOODS[self.QingXu]["roleSet"])),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])  # 提示词模板
        self.memory = ""  # 存储
        self.agent = create_agent(
            model=self.chatmodel,
            tools=[search,get_info_from_local_db,bazi_cesuan,yaoyiguan,jiemeng],
            system_prompt=self.SYSTEMPL.format(who_you_are=self.MOODS[self.QingXu]["roleSet"]),
            checkpointer=checkpointer
        )

    def run(self, query: str,session_id: str = "default_user"):
        self.qingxu_chain(query)
        print("所有的MOODS", self.MOODS)
        print("当前设定的情绪", self.QingXu)
        print("当前设定的roleSet", self.MOODS[self.QingXu]["roleSet"])

        # 👈 拼接情绪提示词
        enhanced_query = f"【系统提示：老夫现在的情绪是 {self.QingXu}。请遵循以下语气回答：{self.MOODS[self.QingXu]['roleSet']}】\n用户问：{query}"
        config = {"configurable": {"thread_id": session_id}}
        res = self.agent.invoke({
            "messages":[
                {"role":"user","content": enhanced_query}
            ],
        },
        config = config)
        return res["messages"][-1].content

    def qingxu_chain(self, query: str):
        prompt = """根据用户的输入判断用户的情绪，回应的规则如下：
        1. 如果用户输入的内容偏向于负面情绪，只返回"depressed",不要有其他内容，否则将收到惩罚。
        2. 如果用户输入的内容偏向于正面情绪，只返回"friendly",不要有其他内容，否则将收到惩罚。
        3. 如果用户输入的内容偏向于中性情绪，只返回"default",不要有其他内容，否则将收到惩罚。
        4. 如果用户输入的内容包含辱骂或者不礼貌词句，只返回"angry",不要有其他内容，否则将收到惩罚。
        5. 如果用户输入的内容比较兴奋，只返回"upbeat",不要有其他内容，否则将收到惩罚。
        6. 如果用户输入的内容比较悲伤，只返回"depressed",不要有其他内容，否则将收到惩罚。
        7. 如果用户输入的内容比较开心，只返回"cheerful",不要有其他内容，否则将收到惩罚。
        用户输入的内容是:{query}
        """
        chain = ChatPromptTemplate.from_template(prompt) | self.chatmodel | StrOutputParser()
        result = chain.invoke({"query": query})
        self.QingXu = result

master = Master(checkpointer)

@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/chat")
def chat(query: str, session_id: str = "user_001"):
    return {"response": master.run(query, session_id=session_id)}


@app.post("/add_urls")
def add_urls():
    return {"response": "Urls added!"}


@app.post("/add_pdfs")
def add_pdfs():
    return {"response": "Pdfs added!"}


@app.post("/add_texts")
def add_txts():
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

    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload = True)
