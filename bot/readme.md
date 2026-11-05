# 服务器
1. 接口访问，python选型fastapi
2. /chat接口,post
3. /add_urls,从url中学习知识
4. /add_pdfs，从pdf中学习知识
5. /add_texts，从txt中学习知识


# 人性化
1. 用户输入->ai判断当前问题的情绪倾向->判断>反馈>agent判断
  * 添加MOODS,根据情绪来设置userSet
2. 工具调用，agent判断使用的工具

# 项目介绍
1. 整体框架设计：serer端
   * fastapi+langchain
2. API层实现
   * fastapi
3. 主Class的实现
4. Agent的实现
5. Prompt设计
6. Chain在情绪判断的使用