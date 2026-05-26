import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from agents.deep_research.scope_agent import build_graph
from agents.common_utils.utils import format_messages

if __name__ == "__main__":

    checkpointer = InMemorySaver()
    scope_agent = build_graph(checkpointer=checkpointer)

    graph_image = scope_agent.get_graph(xray=True).draw_mermaid_png()

    ## save graph image for visualization
    with open("./static_images/scope_agent_graph.png", "wb") as f:
        f.write(graph_image)

    # Run the workflow
    thread = {"configurable": {"thread_id": "1"}}

    while True:
        user_message = input("Enter your research question (or 'exit' to quit): ")
        if user_message.lower() == "exit":
            break
        result = scope_agent.invoke({"messages": [HumanMessage(content=user_message)]}, config=thread)
        # messages = result.get("messages", [])
        # answer = messages[-1].content if messages else "No response generated."
        # print(f"\nAgent Response:\n{answer}\n") 
        format_messages(result['messages'])