# MCP-LangChain-Agent

Powerful AI assistant that bridges the gap between static documents and live data. Built with LangChain, OpenAI, and the Model Context Protocol (MCP), it allows you to chat with PDFs, TXT files, and CSV data, while also querying live SQL databases—all through a single, unified interface.

## Key Features

Multi-File RAG: Upload and query PDF and TXT files using ChromaDB vector search.

Data Analysis: Native CSV support via a specialized Pandas execution agent.

Live SQL Integration: Connects to external databases via the Model Context Protocol (MCP).

Agentic Reasoning: Uses a ReAct agent to intelligently decide which tool (PDF, CSV, or SQL) to use based on your question.

Interactive UI: Built with Chainlit for a modern, real-time chat experience.

## Tech Stack

LLM: OpenAI GPT-4o

Orchestration: LangChain & LangGraph

Interface: Chainlit

Vector Database: ChromaDB

Protocol: Model Context Protocol (MCP)
