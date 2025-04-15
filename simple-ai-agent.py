# Demo to introduce LangGraph and it's concept of state, node and edges
# State is simply the session state of an agent session. It can be a TypedDict , Pydantic model, or dataclass
# Nodes are functions to carry out logics within your AI agent session.
# Edges are the logic paths existing within your AI agent session (graph). There's always a START and an END

# Import the needed libnraries
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict

load_dotenv()

# llm = ChatOllama(model="deepseek-r1:latest")
llm = ChatOpenAI(model="gpt-4o")

class AgentState(TypedDict):
    user: str
    task: str
    response: str
    
def ask_question(state: AgentState):
    """ We would ask a Chat LLM a question and save some of the session details in the state"""
    user_name = state['user']
    question = state['task']
    response = llm.invoke(question) 
    
    print(response)
    
    # return a response
    return {"user": user_name,
    "task": question,
    "response": response }
    
def main():
    # We build the Graph, specifying the nodes and edges
    # Initialise the graph
    graph_builder = StateGraph(AgentState)
   
   # Add the nodes
    graph_builder.add_node("ask-question",ask_question)
    
    # Set the edges, there's always a START and END node in addition to our explicitly defined node
    graph_builder.add_edge(START,"ask-question")
    graph_builder.add_edge("ask-question",END)
    
    # Compile the graph
    graph = graph_builder.compile()

    # Generate an image file of the graph
    graph.get_graph().draw_mermaid_png(output_file_path="simple-ai-graph.png")
    
    # Trigger the AI agent session
    graph.invoke({
        "user" : "Michael Olafusi",
        "task" : "What country in the world has the largest number of lakes?"
    })
    
    dummy = 123
    
    
    
    
if __name__ == "__main__":
    main()
        