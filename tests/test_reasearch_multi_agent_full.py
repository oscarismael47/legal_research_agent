import uuid
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from agents.deep_research.reasearch_multi_agent_full import build_graph
from agents.common_utils.utils import format_messages, format_messages_as_str

if __name__ == "__main__":

    output_dir = Path(__file__).resolve().parent.parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpointer = InMemorySaver()

    reasearch_multi_agent_full = build_graph(checkpointer=checkpointer)

    graph_image = reasearch_multi_agent_full.get_graph(xray=True).draw_mermaid_png()

    ## save graph image for visualization
    with open(f"{output_dir}/reasearch_multi_agent_full_graph.png", "wb") as f:
        f.write(graph_image)

    # Run the workflow
    thread = {"configurable": {"thread_id": str(uuid.uuid4())}}

    # # Example brief
    # research_brief = """I want to identify and evaluate the coffee shops in San Francisco that are considered the best based specifically  
    # on coffee quality. My research should focus on analyzing and comparing coffee shops within the San Francisco area, 
    # using coffee quality as the primary criterion. I am open regarding methods of assessing coffee quality (e.g.,      
    # expert reviews, customer ratings, specialty coffee certifications), and there are no constraints on ambiance,      
    # location, wifi, or food options unless they directly impact perceived coffee quality. Please prioritize primary    
    # sources such as the official websites of coffee shops, reputable third-party coffee review organizations (like     
    # Coffee Review or Specialty Coffee Association), and prominent review aggregators like Google or Yelp where direct  
    # customer feedback about coffee quality can be found. The study should result in a well-supported list or ranking of
    # the top coffee shops in San Francisco, emphasizing their coffee quality according to the latest available data as  
    # of July 2025."""

    research_brief = """Compare Gemini and OpenAI Deep Research agents across capability, performance, pricing, and ideal research use cases."""

    result = reasearch_multi_agent_full.invoke({"messages": [HumanMessage(content=f"{research_brief}.")]}, config=thread)
    final_report = result["final_report"]
    supervisor_messages_str = format_messages_as_str(result["supervisor_messages"])

    format_messages(result["supervisor_messages"])

    # save final_report as markdown file
    with open(f"{output_dir}/final_report.md", "w", encoding="utf-8") as f:
        f.write(final_report.strip() + "\n")

    with open(f"{output_dir}/supervisor_messages.txt", "w", encoding="utf-8") as file:
        file.write(supervisor_messages_str)