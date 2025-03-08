from langchain_core.tools import tool
from langchain_experimental.utilities import PythonREPL

from langchain_core.messages import AIMessage
from typing import Annotated, Tuple
from langgraph.prebuilt import InjectedState
import sys
from io import StringIO
import os
import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px
import pandas as pd
import requests
import wikipediaapi
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
#from langchain_community.document_loaders import PyPDFDirectoryLoader
#from langchain_community.tools import create_retrieval_tool 
#from .KnowledgeBASE import retriever 

# @tool(parse_docstring=True)
# def retriever_tool(query: str) -> Tuple[str, dict]:
#     """
#     Search the Vector Database for a query and generate questions and answers.

#     Args:
#         query (str): The search query string.

#     Returns:
#         Tuple[str, dict]: The formatted Q&A and state updates.
#     """
#     # Execute the retrieval process and get results
#     result = retriever.retrieve(query)
    
#     # Format the response as a Q&A
#     output = {
#         "thought": f"Searching the vector database for the query: {query}",
#         "content": f"""
#             Question: What is the information related to the query '{query}' from the vector database?
#             Answer: {result}
#         """
#     }

#     return output["content"], {
#         "intermediate_outputs": [{
#             "thought": output["thought"],
#             "output": result
#         }]
#     }

# Initialize Wikipedia API
wiki_wiki = wikipediaapi.Wikipedia('en')

@tool(parse_docstring=True)
def duckduckgo_search(query: str) -> Tuple[str, dict]:
    """
    Search DuckDuckGo for a query and generate questions and answers.

    Args:
        query (str): The search query string.

    Returns:
        Tuple[str, dict]: The formatted Q&A and state updates.
    """
    try:
        # Add user-agent header to avoid being blocked
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(
            f"https://api.duckduckgo.com/",
            params={
                'q': query,
                'format': 'json',
                't': 'ML_QA_Bot'
            },
            headers=headers
        )
        
        if response.status_code == 200:
            data = response.json()
            abstract = data.get('AbstractText', '')
            related_topics = data.get('RelatedTopics', [])
            
            # If no abstract, try to get information from related topics
            if not abstract and related_topics:
                abstract = related_topics[0].get('Text', 'No results found.')
            elif not abstract:
                abstract = 'No results found.'
            
            # Format the response as Q&A
            output = {
                "thought": f"Searching DuckDuckGo for information about {query}",
                "content": f"""
                    Question: What are the key concepts of {query} according to the search results?
                    Answer: {abstract}
                    """
            }
            
            return output["content"], {
                "intermediate_outputs": [{
                    "thought": output["thought"],
                    "output": abstract
                }]
            }
        
        error_msg = f"Error: DuckDuckGo API returned status code {response.status_code}"
        return error_msg, {
            "intermediate_outputs": [{
                "thought": "Error in DuckDuckGo search",
                "output": error_msg
            }]
        }
        
    except Exception as e:
        error_msg = f"Error in DuckDuckGo search: {str(e)}"
        return error_msg, {
            "intermediate_outputs": [{
                "thought": "Error in DuckDuckGo search",
                "output": error_msg
            }]
        }

@tool(parse_docstring=True)
def arxiv_search(query: str) -> Tuple[str, dict]:
    """
    Search arXiv for a query and generate questions and answers.

    Args:
        query (str): The search query string.

    Returns:
        Tuple[str, dict]: The formatted Q&A and state updates.
    """
    response = requests.get(f"http://export.arxiv.org/api/query?search_query=all:{query}&start=0&max_results=1")
    if response.status_code == 200:
        result = response.text
        
        # Format the response as Q&A
        output = f"""
        Question: What are the recent research findings about {query} according to arXiv?
        Answer: {result}
        """
        return output, {"intermediate_outputs": [{"thought": f"Searching arXiv for {query}", "output": result}]}
    
    return 'Error fetching results from arXiv.', {"intermediate_outputs": [{"thought": "Error in arXiv search", "output": "Error"}]}

@tool(parse_docstring=True)
def wikipedia_search(query: str) -> Tuple[str, dict]:
    """
    Search Wikipedia for a query and generate questions and answers.

    Args:
        query (str): The search query string.

    Returns:
        Tuple[str, dict]: The formatted Q&A and state updates.
    """
    page = wiki_wiki.page(query)
    if page.exists():
        summary = page.summary
        
        # Format the response as Q&A
        output = f"""
        Question: What is {query} and what are its main concepts according to Wikipedia?
        Answer: {summary}
        """
        return output, {"intermediate_outputs": [{"thought": f"Searching Wikipedia for {query}", "output": summary}]}
    
    return 'No Wikipedia page found for this query.', {"intermediate_outputs": [{"thought": "Error in Wikipedia search", "output": "No page found"}]}



# Python REPL initialization
repl = PythonREPL()

persistent_vars = {}
plotly_saving_code = """import pickle
import uuid
import plotly

for figure in plotly_figures:
    pickle_filename = f"images/plotly_figures/pickle/{uuid.uuid4()}.pickle"
    with open(pickle_filename, 'wb') as f:
        pickle.dump(figure, f)
"""

@tool(parse_docstring=True)
def complete_python_task(
        graph_state: Annotated[dict, InjectedState], thought: str, python_code: str
) -> Tuple[str, dict]:
    """
    Completes a python task and ensures both questions and answers are generated.

    Args:
        graph_state: The current state of the graph.
        thought: Internal thought about the next action to be taken, and the reasoning behind it. This should be formatted in MARKDOWN and be high quality.
        python_code: Python code to be executed to perform analyses, create a new dataset or create a visualization.
    """
    current_variables = graph_state["current_variables"] if "current_variables" in graph_state else {}
    for input_dataset in graph_state["input_data"]:
        if input_dataset.syllabus_name not in current_variables:
            if input_dataset.data_path.endswith('.csv'):
                try:
                    current_variables[input_dataset.syllabus_name] = pd.read_csv(input_dataset.data_path)
                except Exception as e:
                    return str(e), {"intermediate_outputs": [{"thought": thought, "code": python_code, "output": f"Error reading CSV: {str(e)}"}]}
            elif input_dataset.data_path.endswith('.txt'):
                try:
                    with open(input_dataset.data_path, 'r') as f:
                        current_variables[input_dataset.syllabus_name] = f.read()
                except Exception as e:
                    return str(e), {"intermediate_outputs": [{"thought": thought, "code": python_code, "output": f"Error reading text file: {str(e)}"}]}
    if not os.path.exists("images/plotly_figures/pickle"):
        os.makedirs("images/plotly_figures/pickle")

    current_image_pickle_files = os.listdir("images/plotly_figures/pickle")
    try:
        # Capture stdout
        old_stdout = sys.stdout
        sys.stdout = StringIO()

        # Execute the code and capture the result
        exec_globals = globals().copy()
        exec_globals.update(persistent_vars)
        exec_globals.update(current_variables)
        exec_globals.update({"plotly_figures": []})

        exec(python_code, exec_globals)
        persistent_vars.update({k: v for k, v in exec_globals.items() if k not in globals()})

        # Get the captured stdout
        output = sys.stdout.getvalue()

        # Restore stdout 
        sys.stdout = old_stdout

        updated_state = {
            "intermediate_outputs": [{"thought": thought, "code": python_code, "output": output}],
            "current_variables": persistent_vars
        }

        if 'plotly_figures' in exec_globals:
            exec(plotly_saving_code, exec_globals)
            # Check if any images were created
            new_image_folder_contents = os.listdir("images/plotly_figures/pickle")
            new_image_files = [file for file in new_image_folder_contents if file not in current_image_pickle_files]
            if new_image_files:
                updated_state["output_image_paths"] = new_image_files
            
            persistent_vars["plotly_figures"] = []

        # Ensure output includes both questions and answers
        output = f"""
        Question: {thought}
        Answer: {output}
        Code: 
        ```python
        {python_code}
        ```
        """
        return output, updated_state
    except Exception as e:
        return str(e), {"intermediate_outputs": [{"thought": thought, "code": python_code, "output": str(e)}]}
