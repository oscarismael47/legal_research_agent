"""
Full Multi-Agent Research System

This module integrates all components of the research system:
- User clarification and scoping
- Research brief generation  
- Multi-agent research coordination
- Final report generation

The system orchestrates the complete research workflow from initial user
input through final report delivery.
"""

from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END

from agents.common_utils import utils
from agents.deep_research import prompts
from agents.deep_research.states import AgentInputState, AgentState
from agents.deep_research.scope_agent import clarify_with_user, write_research_brief
from agents.deep_research.research_supervisor_agent import build_graph as build_research_supervisor_graph  

from agents.common_utils.base_models import MODEL

def final_report_generation(state: AgentState):
    """
    Final report generation node.
    
    Synthesizes all research findings into a comprehensive final report
    """
    
    notes = state.get("notes", [])
    
    findings = "\n".join(notes)

    final_report_prompt = prompts.final_report_generation_prompt.format(
        research_brief=state.get("research_brief", ""),
        findings=findings,
        date=utils.get_today_str()
    )
    
    final_report = MODEL.invoke([HumanMessage(content=final_report_prompt)])
    
    return {
        "final_report": final_report.content, 
        "messages": [  AIMessage(content= "Here is the final report: " + final_report.content) ],
    }


# ===== GRAPH CONSTRUCTION =====
def build_graph(checkpointer = None):
    """
    """
    supervisor_subgraph = build_research_supervisor_graph()

    # Build the overall workflow
    graph = StateGraph(AgentState, input_schema=AgentInputState)

    # Add workflow nodes
    graph.add_node("clarify_with_user", clarify_with_user)
    graph.add_node("write_research_brief", write_research_brief)
    graph.add_node("supervisor_subgraph", supervisor_subgraph)
    graph.add_node("final_report_generation", final_report_generation)

    # Add workflow edges
    graph.add_edge(START, "clarify_with_user")
    graph.add_edge("write_research_brief", "supervisor_subgraph")
    graph.add_edge("supervisor_subgraph", "final_report_generation")
    graph.add_edge("final_report_generation", END)

    # Compile the workflow
    if checkpointer:
        return graph.compile(checkpointer=checkpointer)
    else:
        return graph.compile()