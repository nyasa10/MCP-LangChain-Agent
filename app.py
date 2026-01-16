import io
import os
import pandas as pd
import chainlit as cl

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent

# 1. Setup Logic for PDF/TXT (RAG)
async def process_file(file: cl.File):
    if file.name.endswith(".pdf"):
        loader = PyPDFLoader(file.path)
    else:
        loader = TextLoader(file.path)
    
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = text_splitter.split_documents(docs)
    
    vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
    return vectorstore.as_retriever()

@cl.on_chat_start
async def start():
    # Wait for user to upload files
    files = await cl.AskFileMessage(
        content="Please upload PDF, TXT, or CSV files to begin!",
        accept=["text/plain", "application/pdf", "text/csv"],
        max_size_mb=20,
        max_files=5
    ).send()

    all_tools = []
    llm = ChatOpenAI(model="gpt-4o", temperature=0, streaming=True)

    for file in files:
        msg = cl.Message(content=f"Processing `{file.name}`...")
        await msg.send()

        if file.name.endswith(".csv"):
            # Setup Pandas Agent as a Tool
            df = pd.read_csv(file.path)
            pandas_agent = create_pandas_dataframe_agent(
                llm, df, agent_type="openai-tools", allow_dangerous_code=True
            )
            # Wrap as a LangChain tool
            from langchain.tools import Tool
            csv_tool = Tool(
                name=f"query_csv_{file.name.replace('.','_')}",
                func=lambda q, agent=pandas_agent: agent.run(q),
                description=f"Use this to answer questions about the data in {file.name}"
            )
            all_tools.append(csv_tool)
        else:
            # Setup RAG Retriever as a Tool
            retriever = await process_file(file)
            retriever_tool = create_retriever_tool(
                retriever, 
                f"search_{file.name.replace('.','_')}", 
                f"Search for specific details inside {file.name}"
            )
            all_tools.append(retriever_tool)

    # 2. Add MCP Tools (External Data)
    try:
        mcp_client = MultiServerMCPClient({
            "sql_db": {
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-postgres", os.getenv("DB_URL")],
                "transport": "stdio",
            }
        })
        mcp_tools = await mcp_client.get_tools()
        all_tools.extend(mcp_tools)
        cl.user_session.set("mcp_client", mcp_client)
    except Exception as e:
        print(f"MCP failed to load: {e}")

    # 3. Initialize Unified Agent
    agent_executor = create_react_agent(llm, all_tools)
    cl.user_session.set("agent", agent_executor)
    cl.user_session.set("history", [])

    await cl.Message(content="All files processed! I can now query your docs, CSVs, and SQL database.").send()

@cl.on_message
async def main(message: cl.Message):
    agent = cl.user_session.get("agent")
    history = cl.user_session.get("history")
    
    history.append(("user", message.content))
    
    # Run agent
    res = await agent.ainvoke({"messages": history})
    answer = res["messages"][-1].content
    
    history.append(("assistant", answer))
    cl.user_session.set("history", history)
    
    await cl.Message(content=answer).send()

@cl.on_chat_end
async def end():
    mcp_client = cl.user_session.get("mcp_client")
    if mcp_client:
        await mcp_client.close()
