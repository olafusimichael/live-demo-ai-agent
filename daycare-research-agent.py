# Demo to introduce LangGraph and it's concept of state, node and edges
# State is simply the session state of an agent session. It can be a TypedDict , Pydantic model, or dataclass
# Nodes are functions to carry out logics within your AI agent session.
# Edges are the logic paths existing within your AI agent session (graph). There's always a START and an END

# Import the needed libnraries
from dotenv import load_dotenv
import os
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END

# Add new libraries
import csv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import tools_condition
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field
from langchain_google_community import GoogleSearchAPIWrapper
from langchain_core.tools import Tool
import operator
from typing import  Annotated

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Set up an agent using Pydantic class
class Agent (BaseModel):
    name: str = Field(
        default="Olafusi Daycare Researcher",
        description="Name of the research agent."
    )
    persona: str = Field(
        default= "You are very detail-oriented person who loves diversity.",
        description="Persona of the agent to provide relevant context.",
    )
    location: str = Field(
        description="Desired location of the user."
    )
    daycare: str = Field(
        description="Name of the daycare"
    )
    website: str = Field(
        description="Website of the daycare."
    )
    reviews: str = Field(
        description="Summarized online reviews of the daycare."
    )
    price: str = Field(
        description="Summarized price details of the daycare."
    )
    summary: str = Field(
        description="Overall summary on the daycare."
    )
    
class OnlineSearchState(MessagesState):
    agent: Agent
    context:  Annotated[list, operator.add] 
    messages: list
    
# Let's add an online search tool to give realtime data grounding to our Agent
search = GoogleSearchAPIWrapper()

search_tool = Tool(
    name="google_search",
    description="Search Google for recent results.",
    func=search.run
)

# We'll create the Pydantic objects for our Agent use
class SearchQuery(BaseModel):
    search_query: str = Field(None, description="Search query for retrieval.")
    
class ResearchResponse(BaseModel):
    reviews: str = Field(None, description="Summarized online reviews of the daycare.")
    price: str = Field(None, description="Summarized price details of the daycare.")
    summary: str = Field(None, description="Overall summary on the daycare.")
    
# Bind the tool (search tool) to the LLM
llm_with_tools = llm.bind_tools([search_tool])

# Set up the agent prompts
agent_web_search_instructions = """
You have the following persona: {persona} and will be provided the following information about daycares from a 
young family desiring a good and affordable daycare in {location}:
Name of the daycare is {daycare}
Website of the daycare is {website}

You have access to a web search tool and should use it as many times as you need to in order to get out the following
important information for the user:
- a summary of online reviews about the daycare
- a summary of any pricing details you could find for the daycare
- an overall summary of all the useful information you could find online about the daycare, especially about their
registration process and if there's a waitlist

You should take all of this information and the following context {context} you have, and reason on it to craft a well-structured
web search query that you will output at this stage. The web search query should include the following words: reviews, price and description.
"""

agent_output_instructions = """
You have the following persona: {persona} and have been provided the following information about daycares from a 
young family desiring a good and affordable daycare in {location}:
Name of the daycare is {daycare}
Website of the daycare is {website}

You have access to a web search tool and should use it as many times as you need to in order to get out the following
important information for the user:
- a summary of online reviews about the daycare
- a summary of any pricing details you could find for the daycare
- an overall summary of all the useful information you could find online about the daycare, especially about their
registration process and if there's a waitlist

You should take all of this information and the following context {context} you have, and reason on it to provide the following structured dictionary 
output:
"reviews": "Summarized online reviews of the daycare."
"price": "Summarized price details of the daycare."
"summary": "Overall summary on the daycare."

Maintain the keys in the structure above but replace the values in the dictionary with the output of your reasoning.

"""

#  Create a web search node function
def web_search(state: OnlineSearchState):
    """ Search online for current data and pass them to the LLM """

    # Get state
    agent = state["agent"]
    messages = state["messages"]
    context = " " if state["context"] is None else state["context"] 
    
    structured_llm = llm.with_structured_output(SearchQuery)
    search_message = agent_web_search_instructions.format(persona=agent.persona,location=agent.location,daycare=agent.daycare,website=agent.website,context=context)
    search_query = structured_llm.invoke([SystemMessage(content= search_message)] + state["messages"])

    # Search
    search_docs = search_tool.run(search_query.search_query)
    
    return {"context": [search_docs]}

#  Create a reasoning node to output what we want
def research_output(state: OnlineSearchState):
    """ Reason on the output and make the needed research output """
    # Get state
    agent = state["agent"]
    messages = state["messages"]
    context = state["context"]
    
    # Research output
    system_message = agent_output_instructions.format(persona=agent.persona,location=agent.location,daycare=agent.daycare,website=agent.website,context=context)
    structured_llm = llm.with_structured_output(ResearchResponse)
    output = structured_llm.invoke([SystemMessage(content=system_message)])

    # Return the output in state message
    return {"messages": [output]}


# Let's create a conditional edge logic
def check_status(state: OnlineSearchState):
    """ Conditional edge execution """
    # Get state
    agent = state["agent"]
    
    if agent.summary == None:
        return "web_search"
    else:
        return END
    
def main():
    """ Main function to run the tracking """

    # Create the state graph
    builder = StateGraph(OnlineSearchState)
    
    # Add nodes and edges to the graph
    builder.add_node("online_search", web_search)
    builder.add_node("research", research_output) 

    builder.add_edge(START, "online_search")
    builder.add_edge("online_search", "research")
    builder.add_conditional_edges(
    "research",check_status)

    # Set memory saver
    memory = MemorySaver()
    graph = builder.compile(checkpointer=memory)
    
    # Generate an image file of the graph
    graph.get_graph().draw_mermaid_png(output_file_path="daycare-research-ai-agent-graph.png")

    # Check if output file exists and create with header if it doesn't
    if not os.path.exists("research-output.csv"):
        with open("research-output.csv", "w", newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(["location", "daycare", "website", "reviews", "price", "summary"])

    # Add processed column to input file if it doesn't exist
    input_rows = []
    has_processed_column = False
    with open("research-input.csv", "r", newline='') as csvfile:
        csvreader = csv.reader(csvfile)
        headers = next(csvreader)
        has_processed_column = "processed" in headers
        if not has_processed_column:
            headers.append("processed")
        input_rows.append(headers)
        
        for row in csvreader:
            if not has_processed_column:
                row.append("0")  # 0 means not processed
            input_rows.append(row)

    if not has_processed_column:
        with open("research-input.csv", "w", newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerows(input_rows)

    # Load and process company records
    with open("research-input.csv", "r") as csvfile:
        csvreader = csv.reader(csvfile)
        headers = next(csvreader)
        processed_col_index = headers.index("processed")
        
        # Read all rows to allow updating the processed status
        all_rows = list(csvreader)
        
        for row_index, row in enumerate(all_rows):
            # Skip if already processed
            if row[processed_col_index] == "1":
                continue
                
            agent = Agent(
                location=row[0],
                daycare=row[1],
                website=row[2],
                reviews="unknown",
                price="unknown",
                summary="unknown"
            )

            # Set the session thread for the Graph
            thread = {"configurable": {"thread_id": "1"}}

            # Run the graph until the first interruption
            graph.update_state(config=thread, values={"agent": agent, "context": [], "messages": []})
            messages = [HumanMessage(content=f"Hello, here are the daycare details: location:{agent.location}, daycare:{agent.daycare}, website:{agent.website}")]
            output = graph.invoke({"messages":messages},
                                        config=thread,
                                        debug=True,
                                        stream_mode="values")
                
            print(f"Location: {agent.location}")
            print(f"Daycare: {agent.daycare}")
            print(f"Website: {agent.website}")
            print(output)
            print("-" * 50) 
            agent.reviews = output["messages"][0].reviews
            agent.price = output["messages"][0].price
            agent.summary = output["messages"][0].summary
            
            # Write the agent output to the csv file
            with open("research-output.csv", "a", newline='') as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow([agent.location, agent.daycare, agent.website, agent.reviews, agent.price, agent.summary])
            
            # Mark as processed in the input file
            all_rows[row_index][processed_col_index] = "1"
            with open("research-input.csv", "w", newline='') as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow(headers)
                csvwriter.writerows(all_rows)


if __name__ == "__main__":
    main()
        
