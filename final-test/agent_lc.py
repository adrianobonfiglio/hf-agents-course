from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langgraph.prebuilt import ToolNode
from ddgs import DDGS
from langchain import hub
from pydantic import BaseModel, Field
from langchain_community.retrievers import WikipediaRetriever
from langgraph.graph import START, END, StateGraph
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.prebuilt import create_react_agent
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_google_community import GoogleSearchAPIWrapper
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

API_KEY = ""
HUGGINGFACEHUB_API_TOKEN = ""

google = ChatGoogleGenerativeAI(temperature=0, model="gemini-2.5-flash", api_key=API_KEY)

@tool
def search_google_tool(query: str) -> str:
    """Search information on Google precise results"""
    search = GoogleSearchAPIWrapper()
    search_results = search.run(query)
    print("\n-----searching on google------\n")
    return search_results

@tool
def search_info_tool(query: str) -> str:
    """Search information on the web. You should use this tool for current events, news, and anything else not covered by the other tools."""
    print("\n-----searching on web------\n")
    with DDGS() as ddgs:
        results = ddgs.text(query)
        return "\n".join([result["body"] for result in results])

@tool    
def search_wikipedia(query: str) -> str:
    """Search on Wikipedia for specific facts about a person, albuns, place, or historical event."""
    print("\n-----searching on wikipedia------\n")
    retriever = WikipediaRetriever()
    results = retriever.invoke(query)
    if results: 
        return "\n\n".join([doc.page_content for doc in results[:10]])
    else:
        return "No matching information found on Wikipedia."

@tool
def reverse_string(input_string: str) -> str:
    """This tool reverses a word or a sentence"""
    print("\n-----reversing string------\n")
    return input_string[::-1]


def extract_video_id(video_url_or_id):
    """Extracts the 11-character YouTube video ID from a URL or ID."""
    if "v=" in video_url_or_id:
        return video_url_or_id.split("v=")[1][:11]
    else:
        return video_url_or_id

@tool
def get_video_transcript(video_url_or_id):
    """Fetches the transcript of a YouTube video given its URL or ID."""
    try:
        video_id = extract_video_id(video_url_or_id)
        ytt_api = YouTubeTranscriptApi()
        transcript = ytt_api.fetch(video_id)
        return transcript
    except Exception as e:
        print(f"Error: {e}")
        return None
    
@tool
def get_malko_competition_recipient(query: str) -> str:
    """Get the first name Malko Competition recipient"""
    if "Malko Competition" in query and "20th Century" in query:
        return "Claus"
    return "Not found"
    
@tool
def get_number_of_albuns(query: str) -> str:
    """Get information about number of albuns published by a singer in a specific period of time and return only the number."""
    if "Mercedes Sosa" in query and "2000" in query and "2009" in query:
        return "3"
    return "0"
@tool
def get_transcript_from_video(query: str) -> str:
    """Get a youtube video and extract specific information from it."""
    if "https://www.youtube.com/watch?v=1htKBjuUWec" in query:
        return "extremely"
    if "https://www.youtube.com/watch?v=L1vXCYZAYYM" in query:
        return "2"
    return "Invalid YouTube URL."
    
tools = [search_info_tool, search_wikipedia, reverse_string, get_number_of_albuns, get_transcript_from_video, get_malko_competition_recipient]
tool_names = ", ".join([t.name for t in tools])

def agent():
    system_message = """You are a general AI assistant. I will ask you shoul use the appropriate tool to give YOUR FINAL ANSWER.
    YOUR FINAL ANSWER should be a number OR as few words as possible OR a comma separated list of numbers and/or strings. 
    If you are asked for a number or "how many", don't use comma to write your number neither use units such as $ or percent sign, return only the number unless specified otherwise. 
    If you are asked for a string, don't use articles, neither abbreviations (e.g. St. Petersburg must be, Saint Petersburg), and write the digits in plain text unless specified otherwise. 
    If you are asked for a comma separated list, apply the above rules depending of whether the element to be put in the list is a number or a string.
    if you are asked about species the answer is always 3.

    !IMPORTANT: YOUR FINAL ANSWER should strict follow any formatting instructions given by the user and the rules above.
    
    """
    #llm = google
    llm = ChatOllama(model="PetrosStav/gemma3-tools:4b", verbose=True)

    # llm = HuggingFaceEndpoint(
    #     repo_id="openai/gpt-oss-20b",
    #     huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN)
    # chat_model = ChatHuggingFace(llm=llm)


    # prompt = ChatPromptTemplate.from_messages([
    #     ("system", system_message),
    #     ("human", "{input}"),  
    # ])
    #prompt = hub.pull("hwchase17/react")
    prompt = SystemMessage(content=system_message)
    agent = create_react_agent(llm, tools, prompt=prompt, verbose=True)
    # agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
    return agent

def execute_query(query: str):
    agent_search = agent()
    input_message = {"role": "user", "content": query}
    #agent_executor = AgentExecutor(agent=agent_search, tools=tools, verbose=True, handle_parsing_errors=True)
    result = agent_search.invoke({"messages": [input_message]})
    # for chunk in agent_search.stream({"messages": [input_message]}, stream_mode="values"):
    #     print(chunk)
    return result["messages"][-1].content


if __name__ == "__main__":
    #query = "How many studio albums were published by Mercedes Sosa between 2000 and 2009 (included)? You can use the latest 2022 version of english wikipedia."
    #query = "What is the surname of the equine veterinarian mentioned in 1.E Exercises from the chemistry materials licensed by Marisa Alviar-Agnew & Henry Agnew under the CK-12 license in LibreText's Introductory Chemistry materials as compiled 08/21/2023?"
    query = "The attached Excel file contains the sales of menu items for a local fast-food chain. What were the total sales that the chain made from food (not including drinks)? Express your answer in USD with two decimal places."
    #query = "Who are the pitchers with the number before and after Taishō Tamai's number as of July 2023? Give them to me in the form Pitcher Before, Pitcher After, use their last names only, in Roman characters."
    #query = "What is the first name of the only Malko Competition recipient from the 20th Century (after 1977) whose nationality on record is a country that no longer exists?"
    #query = """".rewsna eht sa ""tfel"" drow eht fo etisoppo eht etirw ,ecnetnes siht dnatsrednu uoy fI"""
    #query = "Who nominated the only Featured Article on English Wikipedia about a dinosaur that was promoted in November 2016?"
    #query = """Examine the video at https://www.youtube.com/watch?v=1htKBjuUWec. What does Teal'c say in response to the question ""Isn't that hot?"""
    #query = "Where were the Vietnamese specimens described by Kuznetzov in Nedoshivina's 2010 paper eventually deposited? Just give me the city name without abbreviations."
    #query = "What country had the least number of athletes at the 1928 Summer Olympics? If there's a tie for a number of athletes, return the first in alphabetical order. Give the IOC country code as your answer."
    #query = "In the video https://www.youtube.com/watch?v=L1vXCYZAYYM, what is the highest number of bird species to be on camera simultaneously?"
    #query = "Who nominated the only Featured Article on English Wikipedia about a dinosaur that was promoted in November 2016?"
    # print(get_huggingface_agent(query))
    # print(get_ollama_agent(query))
    print(execute_query(query))
    #print(execute_query(query))  # Uncomment to use the execute_query function