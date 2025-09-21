from typing import TypedDict, Annotated
from langchain_google_community import GoogleSearchAPIWrapper
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from transformers import pipeline
from langchain_core.tools import tool
from langchain.agents import AgentExecutor, create_react_agent, create_tool_calling_agent
from langchain_core.prompts import PromptTemplate
from langgraph.prebuilt import ToolNode
from ddgs import DDGS
from langchain import hub
from pydantic import BaseModel, Field
from langchain_community.retrievers import WikipediaRetriever
from langgraph.graph import START, END, StateGraph
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langgraph.prebuilt import tools_condition

HUGGINGFACEHUB_API_TOKEN = ""

def execute_query(query: str) -> str:
    return get_ollama_with_langgraph_state(query)

def get_huggingface_agent(query: str) -> str:
    llm = HuggingFaceEndpoint(
        repo_id="Qwen/Qwen2.5-Coder-32B-Instruct",
        huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
    )
    chat_model = ChatHuggingFace(llm=llm)
    tools = [search_info_tool, search_wikipedia]

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["input", "tools", "tool_names", "agent_scratchpad"],
        tools=tools
    )
    agent = create_tool_calling_agent(chat_model, tools, prompt)

    print(f"Executing query: {query}")
    
    agent_executor = AgentExecutor(agent=agent, tools=tools, handle_parsing_errors=True, max_iterations=5, max_execution_time=60)
    response = agent_executor.invoke({"input": query})
    #chat_with_tools = chat_model.bind_tools(tools)
    #response = chat_with_tools.invoke([{"role": "user", "content": query}])
    print(f"\n\n\nResponse content: {response['output']}\n\n\n")
    return response['output']

def get_ollama_agent(query: str) -> str:
    llm = ChatOllama(model="llama3.2")

    tools = [search_info_tool, search_wikipedia]

    prompt = PromptTemplate(
        template=prompt_template,
        tools=tools,
        tool_names=", ".join([t.name for t in tools]),
    )
    #prompt = hub.pull("hwchase17/react")

    agent = create_tool_calling_agent(llm, tools, prompt)

    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True, max_iterations=5, max_execution_time=60)
    response = agent_executor.invoke({"input": query})
    
    print(f"Ollama response (first 50 chars): {response['output']}")
    return response['output'] 

def get_ollama_with_langgraph_state(query: str) -> str:
    tools = [search_info_tool, search_wikipedia]

    # prompt = PromptTemplate(
    #     template=prompt_template,
    #     tools=tools,
    #     tool_names=", ".join([t.name for t in tools]),
    # )

    builder = StateGraph(AgentState)

    # Define nodes: these do the work
    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(tools))

    # Define edges: these determine how the control flow moves
    builder.add_edge(START, "assistant")
    builder.add_conditional_edges(
        "assistant",
        tools_condition,
    )
    builder.add_edge("tools", "assistant")
    # builder.add_edge("assistant", END)
    chat = builder.compile()

    messages = [HumanMessage(content=query), AIMessage(content="""                                                    
    You are a general AI assistant. I will ask you a question and you must search and finish the answer with YOUR FINAL ANSWER. 
    YOUR FINAL ANSWER should be a number OR as few words as possible OR a comma separated list of numbers and/or strings. 
    If you are asked for a number, don't use comma to write your number neither use units such as $ or percent sign unless specified otherwise. 
    If you are asked for a string, don't use articles, neither abbreviations (e.g. St. Petersburg will be, Saint Petersburg), and write the digits in plain text unless specified otherwise. 
    If you are asked for a comma separated list, apply the above rules depending of whether the element to be put in the list is a number or a string.

    The available tools are: " + ", ".join([tool.name for tool in tools])

    Final Answer: Make sure to follow any formatting instructions given by the user. Do not give too long of an answer.
    """)]

    #messages = [{"role": "user", "content": query}]
    response = chat.invoke({"messages": messages})

    print("Adriano's agent response:")
    print(response['messages'][-1].content)


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

def assistant(state: AgentState):
    llm = ChatOllama(model="PetrosStav/gemma3-tools:4b", verbose=True)
    # llm = HuggingFaceEndpoint(
    #     repo_id="Qwen/Qwen2.5-Coder-32B-Instruct",
    #     huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
    # )
    tools = [search_wikipedia, search_info_tool]
    chat_with_tools = llm.bind_tools(tools)

    return {
        "messages": [chat_with_tools.invoke(state["messages"])],
    }

@tool
def search_google_tool(query: str) -> str:
    """Search information on Google for the given query."""
    search = GoogleSearchAPIWrapper()
    search_results = search.run(query)
    print("\n-----searching on google------\n")
    return search_results

@tool
def search_info_tool(query: str) -> str:
    """Search information on the web when the web"""
    print("\n-----searching on web------\n")
    with DDGS() as ddgs:
        results = ddgs.text(query)
        return "\n".join([result["body"] for result in results])

@tool    
def search_wikipedia(query: str) -> str:
    """Search Wikipedia when question is specific about names, artists, players, albums"""
    print("\n-----searching on wikipedia------\n")
    retriever = WikipediaRetriever()
    results = retriever.invoke(query)
    if results: 
        return "\n\n".join([doc.page_content for doc in results[:5]])
    else:
        return "No matching information found on Wikipedia."

# prompt_template = """You are a general AI assistant. I will ask you a question and you report your thoughts, and finish with YOUR FINAL ANSWER.
# YOUR FINAL ANSWER should be a number OR as few words as possible OR a comma separated list of numbers and/or strings. 
# If you are asked for a number, don't use comma to write your number neither use units such as $ or percent sign unless specified otherwise. 
# If you are asked for a string, don't use articles, neither abbreviations (e.g. for cities), and write the digits in plain text unless specified otherwise. 
# If you are asked for a comma separated list, apply the above rules depending of whether the element to be put in the list is a number or a string.

# YOUR FINAL ANSWER should strict follow any formatting instructions given by the user. 

# You have access only to the following tools:
# {{tools}}

# !IMPORTANT!: You shoud use the tools only when you are sure that you need them to answer the question.

# Begin!

# Question: {input}
# Thought: {agent_scratchpad}
# Action: The action to take, should be one of {{tool_names}}
# Action Input: [The input to the action]

# """""


# IMPORTANT: You MUST strictly follow the ReAct pattern (Reasoning, Action, Observation):
# 1. First reason about the problem in the "Thought" section
# 2. Then decide what action to take in the "Action" section (using the tools)
# 3. Wait for an observation from the tool
# 4. Based on the observation, continue with another thought
# 5. This cycle repeats until you have enough information to provide a final answer
# NEVER fake or simulate tool output yourself. If you are unable to make progreess in a certain way, try a different tool or a different approach.
# ... (this Thought/Action/Observation cycle can repeat as needed) ...
# Thought: I now know the final answer

if __name__ == "__main__":
    #query = "What are the studio albums of Mercedes Sosa from 2000 to 2009? Yiou can search in wikepdia"
    #query = "What is the surname of the equine veterinarian mentioned in 1.E Exercises from the chemistry materials licensed by Marisa Alviar-Agnew & Henry Agnew under the CK-12 license in LibreText's Introductory Chemistry materials as compiled 08/21/2023?"
    #query = "The attached Excel file contains the sales of menu items for a local fast-food chain. What were the total sales that the chain made from food (not including drinks)? Express your answer in USD with two decimal places."
    #query = "Who are the pitchers with the number before and after Taishō Tamai's number as of July 2023? Give them to me in the form Pitcher Before, Pitcher After, use their last names only, in Roman characters."
    #query = "What is the first name of the only Malko Competition recipient from the 20th Century (after 1977) whose nationality on record is a country that no longer exists?"
    # print(get_huggingface_agent(query))
    # print(get_ollama_agent(query))
    print(get_ollama_with_langgraph_state(query))
    #print(execute_query(query))  # Uncomment to use the execute_query function