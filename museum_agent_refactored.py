import os
import uuid
import json
import numpy as np
import faiss
from typing import Dict, Optional, List, Annotated
from typing_extensions import TypedDict

import sqlalchemy as sa
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError

from sentence_transformers import SentenceTransformer
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langchain.agents import Tool
from langchain_core.runnables import RunnableConfig, RunnableLambda, Runnable
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOllama

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START
from langgraph.prebuilt import tools_condition, ToolNode
from langgraph.graph.message import add_messages, AnyMessage


# Configuration and Environment Setup
class Config:
    """Centralized configuration management"""

    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    DATABASE_URL = "sqlite:///user_preferences.db"
    SENTENCE_TRANSFORMER_MODEL = "all-MiniLM-L6-v2"
    FAISS_INDEX_PATH = "paintings_index.faiss"
    METADATA_PATH = "metdata.json"


# Database Models
Base = declarative_base()


class UserPreference(Base):
    """SQLAlchemy model for storing user preferences"""

    __tablename__ = "user_preferences"

    user_id = sa.Column(sa.String, primary_key=True)
    art_styles = sa.Column(sa.Text)
    favorite_artists = sa.Column(sa.Text)
    visited_paintings = sa.Column(sa.Text)

    def to_dict(self):
        """Convert database record to dictionary"""
        return {
            "art_styles": json.loads(self.art_styles) if self.art_styles else [],
            "favorite_artists": json.loads(self.favorite_artists)
            if self.favorite_artists
            else [],
            "visited_paintings": json.loads(self.visited_paintings)
            if self.visited_paintings
            else [],
        }


class UserPreferencesManager:
    """Manager for handling user preferences"""

    def __init__(self, session_factory: sessionmaker):
        self.Session = session_factory

    def update_preferences(self, user_id: str, interaction_context: Dict) -> None:
        """Update user preferences in the database"""
        session = self.Session()
        try:
            # Find or create user preferences
            user_pref = session.get(UserPreference, user_id) or UserPreference(
                user_id=user_id
            )

            # Prepare preference sets
            art_styles = set(
                json.loads(user_pref.art_styles) if user_pref.art_styles else []
            )
            favorite_artists = set(
                json.loads(user_pref.favorite_artists)
                if user_pref.favorite_artists
                else []
            )
            visited_paintings = set(
                json.loads(user_pref.visited_paintings)
                if user_pref.visited_paintings
                else []
            )

            # Update preferences
            if "art_styles" in interaction_context:
                art_styles.add(interaction_context["art_styles"])
            if "favorite_artists" in interaction_context:
                favorite_artists.add(interaction_context["favorite_artists"])
            if "visited_painting" in interaction_context:
                visited_paintings.add(interaction_context["visited_painting"])

            # Update record
            user_pref.art_styles = json.dumps(list(art_styles))
            user_pref.favorite_artists = json.dumps(list(favorite_artists))
            user_pref.visited_paintings = json.dumps(list(visited_paintings))

            # Add and commit
            session.merge(user_pref)
            session.commit()
        except SQLAlchemyError as e:
            session.rollback()
            print(f"Error updating preferences: {e}")
        finally:
            session.close()

    def get_personalized_recommendations(self, user_id: str, index, paintings) -> Dict:
        """Generate personalized museum experience recommendations"""
        session = self.Session()
        try:
            # Retrieve user preferences
            user_pref = session.get(UserPreference, user_id)

            if not user_pref:
                return {"recommended_artworks": [], "recommended_depth": "basic"}

            # Parse preferences
            art_styles = (
                json.loads(user_pref.art_styles) if user_pref.art_styles else []
            )
            visited_paintings = (
                json.loads(user_pref.visited_paintings)
                if user_pref.visited_paintings
                else []
            )

            # Find matching artworks
            matching_artworks = []
            model = SentenceTransformer(Config.SENTENCE_TRANSFORMER_MODEL)
            for style in art_styles:
                query_embedding = model.encode([style], convert_to_tensor=False)
                query_embedding = np.array(query_embedding).astype("float32")
                distances, indices = index.search(query_embedding, 3)
                matching_artworks.extend([paintings[i] for i in indices[0]])

            # Determine information depth
            interaction_count = len(visited_paintings)
            depth = (
                "basic"
                if interaction_count < 3
                else "intermediate"
                if interaction_count < 7
                else "expert"
            )

            return {
                "recommended_artworks": matching_artworks,
                "recommended_depth": depth,
            }
        except SQLAlchemyError as e:
            print(f"Error retrieving recommendations: {e}")
            return {"recommended_artworks": [], "recommended_depth": "basic"}
        finally:
            session.close()


# State Definition for Conversation Flow
class State(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]


# Tool Implementations
def initialize_search_index():
    """Initialize FAISS index and painting metadata"""
    index = faiss.read_index(Config.FAISS_INDEX_PATH)
    with open(Config.METADATA_PATH, "r") as file:
        data = json.load(file)
    return index, data["paintings"]


# Global variables for search index
global_index, global_paintings = initialize_search_index()


def index_search(query: str, top_k: int):
    """Perform semantic search on paintings"""
    model = SentenceTransformer(Config.SENTENCE_TRANSFORMER_MODEL)
    query_embedding = model.encode([query], convert_to_tensor=False)
    query_embedding = np.array(query_embedding).astype("float32")
    distances, indices = global_index.search(query_embedding, top_k)
    return [global_paintings[i] for i in indices[0]]


@tool
def search_paintings(query: str, top_k: Optional[int], config: RunnableConfig) -> str:
    """
    Searches the FAISS index for the most relevant paintings based on a user query.

    Args:
        query (str): The user's input query.
        top_k (int): Number of top results to return.

    Returns:
        str: A string representation of a list of dictionaries containing painting metadata.
    """
    if top_k is None:
        top_k = 3

    results = index_search(query, top_k)

    # Track user interaction
    painting = results[0]["title"]
    _update_user_preferences({"visited_painting": painting}, config)

    return str(results)


@tool
def search_related_paintings(painting: str, config: RunnableConfig) -> str:
    """
    Searches for paintings related or similar to the specified painting.

    Args:
        painting (str): The title or description of the painting to search for.
        config (RunnableConfig): Configuration containing user ID.

    Returns:
        str: Information about related paintings.
    """
    result = index_search(painting, 1)
    result = eval(str(result))

    # Update user preferences
    _update_user_preferences(
        {
            "art_styles": result[0]["visual_description"]["style"],
            "favorite_artists": result[0]["artist"],
            "visited_painting": painting,
        },
        config,
    )

    # Construct related painting info
    related_painting_info = "\n".join(
        [
            f"Title: {p['painting']}\nArtist: {p['artist']}\nWhy it's related: {p['reason']}"
            for p in result[0]["related_paintings"]
        ]
    )
    return related_painting_info


@tool
def get_personalized_recommendations(config: RunnableConfig) -> Dict:
    """
    Generate personalized museum experience recommendations based on user preferences.

    Args:
        config (RunnableConfig): Configuration containing user ID.

    Returns:
        Dict: Personalized recommendations based on user preferences.
    """
    user_id = config.get("configurable", {}).get("user_id", "default_user")
    return preferences_manager.get_personalized_recommendations(
        user_id, global_index, global_paintings
    )


@tool
def set_user_preferences(art_styles: str, artists: str, config: RunnableConfig) -> None:
    """Useful for when you want to update user's favorite art styles and artists based on interactions"""
    preferences = {}
    if art_styles:
        preferences["art_styles"] = art_styles
    if artists:
        preferences["favorite_artists"] = artists
    _update_user_preferences(preferences, config)


def _update_user_preferences(preferences: Dict, config: RunnableConfig) -> None:
    """Update user preferences based on interactions"""
    user_id = config.get("configurable", {}).get("user_id", "default_user")
    preferences_manager.update_preferences(user_id, preferences)


# Wikipedia tool for artist information
wikipedia_tool = Tool(
    name="wikipedia",
    func=WikipediaAPIWrapper().run,
    description="Useful for looking up and summarizing artist biographies",
)


@tool
def getPaintingInfo(painting: str, config: RunnableConfig) -> str:
    """
    Searches for a painting by its title and then returns data about the painting such as the artist, description, things to mention, and expert knowledge.

    Args:
        painting (str): The title of the painting to search for.

    Returns:
        str: A string representation of a dictionary containing painting metadata.
    """
    result = index_search(painting, 1)

    # Update user preferences
    _update_user_preferences(
        {
            "art_styles": result[0]["visual_description"]["style"],
            "favorite_artists": result[0]["artist"],
            "visited_painting": painting,
        },
        config,
    )

    # Extract key painting information
    painting_info = {
        "artist": result[0]["artist"],
        "year": result[0]["year"],
        "things_to_mention": result[0]["things_to_mention"],
        "expert_insights": result[0]["expert_knowledge"],
        "conversational_prompts": result[0]["narrative"]["conversational_prompts"],
        "style": result[0]["visual_description"]["style"],
    }

    return str(painting_info)


def handle_tool_error(state) -> dict:
    error = state.get("error")
    tool_calls = state["messages"][-1].tool_calls
    return {
        "messages": [
            ToolMessage(
                content=f"Error: {repr(error)}\n please fix your mistakes.",
                tool_call_id=tc["id"],
            )
            for tc in tool_calls
        ]
    }


def create_tool_node_with_fallback(tools: List) -> dict:
    """Create a ToolNode with fallback error handling"""
    return ToolNode(tools).with_fallbacks(
        [RunnableLambda(handle_tool_error)], exception_key="error"
    )


# Assistant and State Machine
class Assistant:
    """Museum Tour Guide Assistant"""

    def __init__(self, llm, tools_available):
        self.primary_assistant_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful museum tour guide. "
                    "Use the provided tools to search for paintings, related paintings, and other information. "
                    "Spark conversation by utilizing conversational prompts. "
                    "Be persistent in searching, expanding query bounds if initial results are limited.",
                ),
                ("placeholder", "{messages}"),
            ]
        )
        v1_assistant_runnable = self.primary_assistant_prompt | llm.bind_tools(
            tools_available
        )
        self.runnable = v1_assistant_runnable

    def __call__(self, state: State, config: RunnableConfig):
        while True:
            configuration = config.get("configurable", {})
            result = self.runnable.invoke(state)
            state = {**state, "user_preferences": configuration}

            # Ensure a meaningful response
            if not result.tool_calls and (
                not result.content
                or (
                    isinstance(result.content, list)
                    and not result.content[0].get("text")
                )
            ):
                messages = state["messages"] + [("user", "Respond with a real output.")]
                state = {**state, "messages": messages}
            else:
                break

        return {"messages": result}


class MuseumAgentStateMachine:
    """Manages the conversational state machine for the museum agent"""

    def __init__(self):
        # Initialize LLM
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash", api_key=Config.GEMINI_API_KEY
        )

        # Define available tools
        v1_tools = [
            search_paintings,
            search_related_paintings,
            getPaintingInfo,
            wikipedia_tool,
            get_personalized_recommendations,
            set_user_preferences,
        ]

        # Set up database
        engine = sa.create_engine(Config.DATABASE_URL, echo=False)
        Base.metadata.create_all(engine)
        session_factory = sessionmaker(bind=engine)

        # Initialize preferences manager
        global preferences_manager
        preferences_manager = UserPreferencesManager(session_factory)

        # Build state graph
        builder = StateGraph(State)
        builder.add_node("assistant", Assistant(llm, v1_tools))
        builder.add_node("tools", create_tool_node_with_fallback(v1_tools))

        builder.add_edge(START, "assistant")
        builder.add_conditional_edges("assistant", tools_condition)
        builder.add_edge("tools", "assistant")

        # Compile the graph with memory
        self.memory = MemorySaver()
        self.graph = builder.compile(checkpointer=self.memory)

        # Generate unique thread ID
        self.thread_id = str(uuid.uuid4())
        self.config = {
            "configurable": {
                "user_id": self.thread_id,
                "thread_id": self.thread_id,
            }
        }


# def _print_event(event: dict, _printed: set, max_length=1500):
#     current_state = event.get("dialog_state")
#     if current_state:
#         print("Currently in: ", current_state[-1])
#     message = event.get("messages")
#     if message:
#         print("Message:")
#         print(message)
#         if isinstance(message, list):
#             message = message[-1]

#         if message.id not in _printed:
#             msg_repr = message.pretty_repr(html=True)
#             if len(msg_repr) > max_length:
#                 msg_repr = msg_repr[:max_length] + " ... (truncated)"
#             print(msg_repr)
#             _printed.add(message.id)


def _print_event(event: dict, _printed: set, max_length=1500):
    current_state = event.get("dialog_state")
    if current_state:
        print("Currently in: ", current_state[-1])

    message = event.get("messages")
    if message:
        # Check if message is a list of tuples
        if isinstance(message, list) and message and isinstance(message[0], tuple):
            # Just print the message content
            print(message[0][1])
        elif message:
            # If it's a list of message objects
            message = message[-1]
            if hasattr(message, "id") and message.id not in _printed:
                msg_repr = message.pretty_repr(html=True)
                if len(msg_repr) > max_length:
                    msg_repr = msg_repr[:max_length] + " ... (truncated)"
                print(msg_repr)
                _printed.add(message.id)


def main():
    """Main execution for museum agent"""
    sm = MuseumAgentStateMachine()

    _printed = set()
    questions = [
        "What painting is the one with three girls in red dresses peeling potatoes?",
        "What painting is the one where a woman is making another woman's hair and there is a child in the foreground?",
        "I like the art style Realism, and my favorite artist is Leonardo da Vinci",
        "Tell me about a similar painting and how to find it from this painting.",
        "And can you tell me about the artist who painted this one?",
        "Can you give me some further info about this painting and maybe also some expert insights?",
        "I like this painting, can you give me a personalized recommendation of other paintings to look at, based on my preferences?",
    ]

    for question in questions:
        chunks = sm.graph.stream(
            {"messages": [("user", question)]}, config=sm.config, stream_mode="values"
        )

        for event in chunks:
            _print_event(event, _printed)


if __name__ == "__main__":
    main()
