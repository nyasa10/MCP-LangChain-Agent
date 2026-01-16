import chainlit as cl
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain.tools.retriever import create_retriever_tool

# --- ASSUMPTION: You have your vector store logic here ---
# from your_module import get_vector_retriever 

@cl.on_chat_start
async def start():
    # 1. Initialize Retriever (Replace with your actual PDF loading logic)
    # For now, we assume retriever is initialized elsewhere or here
    # retriever = get_vector_retriever() 
    
    # 2. Convert PDF Retriever into a Tool
    # Note: Ensure 'retriever' variable is actually defined!
    pdf_tool = create_retriever_tool(
        retriever, 
        "search_pdf", 
        "Search for information within the uploaded PDF documents."
    )

    # 3. Connect to MCP Servers
    # TIP: Using 'npx' requires Node.js installed on your machine
    mcp_client = MultiServerMCPClient({
        "sql_db": {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-postgres", "postgresql://user@localhost/db"],
            "transport": "stdio",
        }
    })

    # 4. Combine all tools
    try:
        mcp_tools = await mcp_client.get_tools()
        all_tools = [pdf_tool] + mcp_tools
    except Exception as e:
        await cl.Message(content=f"Error connecting to MCP servers: {e}").send()
        all_tools = [pdf_tool]

    # 5. Create the Agent
    llm = ChatOpenAI(model="gpt-4o", streaming=True)
    agent_executor = create_react_agent(llm, all_tools)

    # 6. Initialize History
    cl.user_session.set("message_history", [])
    cl.user_session.set("agent", agent_executor)
    cl.user_session.set("mcp_client", mcp_client)

    await cl.Message(content="MCP Agent initialized. How can I help?").send()

@cl.on_message
async def main(message: cl.Message):
    agent = cl.user_session.get("agent")
    history = cl.user_session.get("message_history")
    
    # Add new user message to history
    history.append(("user", message.content))
    
    # Execute agent with full history for context-awareness
    res = await agent.ainvoke({"messages": history})
    
    # Get the response text
    answer = res["messages"][-1].content
    
    # Add assistant response to history
    history.append(("assistant", answer))
    cl.user_session.set("message_history", history)
    
    await cl.Message(content=answer).send()

@cl.on_chat_end
async def end():
    client = cl.user_session.get("mcp_client")
    if client:
        await client.close()
