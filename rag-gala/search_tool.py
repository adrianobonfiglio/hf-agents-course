from langchain_community.tools import DuckDuckGoSearchRun
from langchain.tools import Tool


def search_info_tool(query: str) -> str:
    search = DuckDuckGoSearchRun()
    results = search.invoke(query)
    return results


web_search_tool = Tool(
    name="web_search_tool",
    func=search_info_tool,
    description="Use this tool to search for information on the web.",
)