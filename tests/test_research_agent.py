import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from agents.deep_research.research_agent import build_graph
from agents.common_utils.utils import format_messages

if __name__ == "__main__":

    checkpointer = InMemorySaver()
    research_agent = build_graph(checkpointer=checkpointer)

    graph_image = research_agent.get_graph(xray=True).draw_mermaid_png()

    ## save graph image for visualization
    with open("./static_images/research_agent_graph.png", "wb") as f:
        f.write(graph_image)

    # Run the workflow
    thread = {"configurable": {"thread_id": "1"}}

    # Example brief
    research_brief = """I want to identify and evaluate the coffee shops in San Francisco that are considered the best based specifically  
    on coffee quality. My research should focus on analyzing and comparing coffee shops within the San Francisco area, 
    using coffee quality as the primary criterion. I am open regarding methods of assessing coffee quality (e.g.,      
    expert reviews, customer ratings, specialty coffee certifications), and there are no constraints on ambiance,      
    location, wifi, or food options unless they directly impact perceived coffee quality. Please prioritize primary    
    sources such as the official websites of coffee shops, reputable third-party coffee review organizations (like     
    Coffee Review or Specialty Coffee Association), and prominent review aggregators like Google or Yelp where direct  
    customer feedback about coffee quality can be found. The study should result in a well-supported list or ranking of
    the top coffee shops in San Francisco, emphasizing their coffee quality according to the latest available data as  
    of July 2025."""

    result = research_agent.invoke({"researcher_messages": [HumanMessage(content=f"{research_brief}.")]}, config=thread)
    format_messages(result['researcher_messages'])