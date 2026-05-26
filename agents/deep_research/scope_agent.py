from typing_extensions import Literal
from langchain_core.messages import HumanMessage, AIMessage, get_buffer_string
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command

from agents.common_utils.base_models import MODEL
from agents.common_utils import utils

from agents.deep_research import schemas, prompts
from agents.deep_research.states import AgentState, AgentInputState

# ===== WORKFLOW NODES =====

def clarify_with_user(state: AgentState) -> Command[Literal["write_research_brief", "__end__"]]:
    """
    Determine if the user's request contains sufficient information to proceed with research.
    
    Uses structured output to make deterministic decisions and avoid hallucination.
    Routes to either research brief generation or ends with a clarification question.
    """
    # Set up structured output model
    structured_output_model = MODEL.with_structured_output(schemas.ClarifyWithUser)

    # Invoke the model with clarification instructions
    response = structured_output_model.invoke([
        HumanMessage(content=prompts.clarify_with_user_instructions.format(
            messages=get_buffer_string(messages=state["messages"]), 
            date=utils.get_today_str()
        ))
    ])
    
    # Route based on clarification need
    if response.need_clarification:
        return Command(
            goto=END, 
            update={"messages": [AIMessage(content=response.question)]}
        )
    else:
        return Command(
            goto="write_research_brief", 
            update={"messages": [AIMessage(content=response.verification)]}
        )

def write_research_brief(state: AgentState):
    """
    Transform the conversation history into a comprehensive research brief.
    
    Uses structured output to ensure the brief follows the required format
    and contains all necessary details for effective research.
    """
    # Set up structured output model
    structured_output_model = MODEL.with_structured_output(schemas.ResearchQuestion)
    
    # Generate research brief from conversation history
    response = structured_output_model.invoke([
        HumanMessage(content=prompts.transform_messages_into_research_topic_prompt.format(
            messages=get_buffer_string(state.get("messages", [])),
            date=utils.get_today_str()
        ))
    ])
    
    # Update state with generated research brief and pass it to the supervisor
    return {
        "research_brief": response.research_brief,
        "supervisor_messages": [HumanMessage(content=f"{response.research_brief}.")]
    }


# ===== GRAPH CONSTRUCTION =====

def build_graph(checkpointer = None):
    """Constructs the agent's workflow graph for scoping research questions."""
    # Build the scoping workflow
    graph = StateGraph(AgentState, input_schema=AgentInputState)

    # Add workflow nodes
    graph.add_node("clarify_with_user", clarify_with_user)
    graph.add_node("write_research_brief", write_research_brief)

    # Add workflow edges
    graph.add_edge(START, "clarify_with_user")
    graph.add_edge("write_research_brief", END)

    # Compile the workflow
    if checkpointer:
        return graph.compile(checkpointer=checkpointer)
    else:
        return graph.compile()

