
from typing_extensions import Literal
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage, get_buffer_string, filter_messages
from langgraph.graph import StateGraph, START, END

from agents.common_utils.base_models import MODEL
from agents.common_utils import utils

from agents.deep_research import schemas, prompts
from agents.deep_research.states import ResearcherState, ResearcherOutputState
from agents.deep_research.tools.reasoning_tools import think_tool
from agents.deep_research.tools.tavily_tools import tavily_search

# Set up tools and model binding
tools = [tavily_search, think_tool]
tools_by_name = {tool.name: tool for tool in tools}

model_with_tools = MODEL.bind_tools(tools)

# ===== AGENT NODES =====

def llm_call(state: ResearcherState):
    """Analyze current state and decide on next actions.
    
    The model analyzes the current conversation state and decides whether to:
    1. Call search tools to gather more information
    2. Provide a final answer based on gathered information
    
    Returns updated state with the model's response.
    """
    return {
        "researcher_messages": [
            model_with_tools.invoke(
                [SystemMessage(content=prompts.research_agent_prompt.format(date=utils.get_today_str()))] + state["researcher_messages"]
            )
        ]
    }

def tool_node(state: ResearcherState):
    """Execute all tool calls from the previous LLM response.
    
    Executes all tool calls from the previous LLM responses.
    Returns updated state with tool execution results.
    """
    tool_calls = state["researcher_messages"][-1].tool_calls
 
    # Execute all tool calls
    observations = []
    for tool_call in tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observations.append(tool.invoke(tool_call["args"]))
            
    # Create tool message outputs
    tool_outputs = [
        ToolMessage(
            content=observation,
            name=tool_call["name"],
            tool_call_id=tool_call["id"]
        ) for observation, tool_call in zip(observations, tool_calls)
    ]
    
    return {"researcher_messages": tool_outputs}

def compress_research(state: ResearcherState) -> dict:
    """Compress research findings into a concise summary.
    
    Takes all the research messages and tool outputs and creates
    a compressed summary suitable for the supervisor's decision-making.
    """
    research_topic = state.get("research_topic","")
    system_message = prompts.compress_research_system_prompt.format(date=utils.get_today_str())
    messages = (
    [SystemMessage(content=system_message)]
    + state.get("researcher_messages", [])
    + [HumanMessage(content=prompts.compress_research_human_message.format(
        research_topic=research_topic
    ))]
)
    response = MODEL.invoke(messages)
    
    # Extract raw notes from tool and AI messages
    raw_notes = [
        str(m.content) for m in filter_messages(
            state["researcher_messages"], 
            include_types=["tool", "ai"]
        )
    ]
    
    return {
        "compressed_research": str(response.content),
        "raw_notes": ["\n".join(raw_notes)]
    }

# ===== ROUTING LOGIC =====

def should_continue(state: ResearcherState) -> Literal["tool_node", "compress_research"]:
    """Determine whether to continue research or provide final answer.
    
    Determines whether the agent should continue the research loop or provide
    a final answer based on whether the LLM made tool calls.
    
    Returns:
        "tool_node": Continue to tool execution
        "compress_research": Stop and compress research
    """
    messages = state["researcher_messages"]
    last_message = messages[-1]
    
    # If the LLM makes a tool call, continue to tool execution
    if last_message.tool_calls:
        return "tool_node"
    # Otherwise, we have a final answer
    return "compress_research"

# ===== GRAPH CONSTRUCTION =====

def build_graph(checkpointer = None):
    """Constructs the research agent's workflow graph."""
    # Build the agent workflow
    graph = StateGraph(ResearcherState, output_schema=ResearcherOutputState)

    # Add nodes to the graph
    graph.add_node("llm_call", llm_call)
    graph.add_node("tool_node", tool_node)
    graph.add_node("compress_research", compress_research)

    # Add edges to connect nodes
    graph.add_edge(START, "llm_call")
    graph.add_conditional_edges(
        "llm_call",
        should_continue,
        {
            "tool_node": "tool_node", # Continue research loop
            "compress_research": "compress_research", # Provide final answer
        },
    )
    graph.add_edge("tool_node", "llm_call") # Loop back for more research
    graph.add_edge("compress_research", END)

    # Compile the workflow
    if checkpointer:
        return graph.compile(checkpointer=checkpointer)
    else:
        return graph.compile()