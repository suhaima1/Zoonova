from langchain_experimental.agents import create_pandas_dataframe_agent

# from langchain.document_loaders import TextLoader
# from langchain.indexes import VectorstoreIndexCreator
from langchain.llms import OpenAI

# from langchain.memory import ConversationBufferMemory,ConversationEntityMemory, ConversationKGMemory,  ConversationSummaryBufferMemory
# from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain.vectorstores import Chroma
# from langchain.text_splitter import CharacterTextSplitter
# from langchain_experimental.agents import create_csv_agent
from langchain.agents import initialize_agent, Tool
from langchain.llms import OpenAI
from langchain.utilities import GoogleSearchAPIWrapper
import openai
import pandas as pd
import os

from dotenv import load_dotenv

load_dotenv()

openai_api_key = os.environ.get("OPENAI_API_KEY")
google_cse_id = os.environ.get("GOOGLE_CSE_ID")
google_api_key = os.environ.get("GOOGLE_API_KEY")

# read csv into a dataframe
df = pd.read_csv("StockDef.csv")

# create a langchain agent
agent = create_pandas_dataframe_agent(
    OpenAI(temperature=0, openai_api_key=openai_api_key), df, verbose=True
)

# wrapper to search the internet when context is not found in knowledge base
search = GoogleSearchAPIWrapper()

# tool for the agent to search both knowledge base and internet
tools = [
    Tool(
        name="Dataframe chatbot",
        func=lambda query: agent.run(query),
        description="Use this always when you get questions about the knowledge base",
    ),
    Tool(
        name="Google search",
        func=search.run,
        description="useful for general question that has no relation with dataframe and we do not have an answer in the dataframe",
    ),
]

# initializing  the agent
self_ask_with_search = initialize_agent(
    tools, llm=OpenAI(temperature=0), agent="zero-shot-react-description"
)

# asking queries from the agent

print(self_ask_with_search.run("how many rows are in the dataframe?"))

self_ask_with_search.run("who is obama")

self_ask_with_search.run("describe each column. what each column represents?")

self_ask_with_search.run(
    "What are the top 10 stocks that show the highest sentiment score? List the stocks and their sentiment score."
)

self_ask_with_search.run(
    "List the stocks that show the positive sentiment score? List all the stocks and their sentiment score."
)

self_ask_with_search.run(
    "List the stocks that show the negative sentiment score? List all the stocks and their sentiment score."
)
