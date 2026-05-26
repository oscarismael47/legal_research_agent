# legal_research_agent

A Python-based multi-agent legal research workflow built on LangGraph and LangChain.

## Overview

This repository implements a structured research agent system for legal and complex topic research. It includes:

- User scoping and clarification via structured output
- Research brief generation from user messages
- A supervisor agent that coordinates parallel research tasks
- Individual researcher agents that perform iterative tool-calling research
- Final report synthesis from aggregated findings

## Architecture

### `agents/common_utils`

Contains shared utilities and model configuration:

- `base_models.py`: loads `OPENAI_API_KEY` from `.env` and initializes `ChatOpenAI` with `gpt-4.1`
- `utils.py`: shared helper functions used across agents

### `agents/deep_research`

Core multi-agent research system:

- `scope_agent.py`: decides whether clarification is needed and transforms conversation history into a research brief
- `research_agent.py`: implements a researcher loop that uses tool calls and compresses findings
- `research_supervisor_agent.py`: supervises research, dispatches parallel tasks, and decides when research is complete
- `reasearch_multi_agent_full.py`: integrates the full workflow from user clarification to final report generation
- `prompts.py`: prompt templates for scoping, research, summarization, and supervision
- `schemas.py`: Pydantic schemas for structured output validation
- `states.py`: state definitions for graph execution

### `agents/main_agent`

Contains workflow orchestration and final report generation logic:

- `graph.py`: builds the complete StateGraph workflow including scoping, research supervision, and report generation
- `prompts.py`: prompt templates used by the main workflow
- `states.py`: graph state classes
- `schemas/amparo_analizador.py`: domain-specific schema definitions

### `agents/deep_research/tools`

Custom tools used by the researcher and supervisor:

- `reasoning_tools.py`: reasoning and internal planning tools
- `tavily_tools.py`: search tool integration
- `supervisor_tools.py`: tools for supervisor action delegation and task management

## Dependency and Environment

The project uses Python `>=3.13` and dependencies declared in `pyproject.toml`.

Key dependencies include:

- `langgraph`
- `langchain`
- `langchain-openai`
- `langchain-anthropic`
- `langchain_tavily`
- `langchain-qdrant`
- `langchain_mcp_adapters`
- `pydantic`
- `rich`
- `jupyter`
- `ipykernel`
- `tavily-python`

### Environment setup

1. Create a virtual environment:

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```bash
pip install -e .
```

3. Add a `.env` file with your OpenAI key:

```text
OPENAI_API_KEY=your_openai_api_key_here
```

## Usage

This repository is primarily a library of agent workflows rather than a finished CLI application.

### Example

```python
from agents.deep_research.reasearch_multi_agent_full import build_graph

workflow = build_graph()
# Example input will depend on the graph state schema and invocation API
# result = workflow.invoke({...})
```

### Note

`main.py` is currently a placeholder entry point and prints a simple greeting:

```python
Hello from legal-research-agent!
```

## Tests

Tests are available under the `tests/` directory:

- `tests/test_reasearch_multi_agent_full.py`
- `tests/test_research_agent.py`
- `tests/test_research_supervisor_agent.py`
- `tests/test_scope_agent.py`

Run tests with your preferred test runner after installing dependencies.

## License

Add your license information here if applicable.
