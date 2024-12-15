import csv
import inspect
import json
import os
import re
from io import StringIO
from typing import Callable, Dict, List, Optional

import faiss
import httpx
import numpy as np
import sqlalchemy as sa
from dotenv import load_dotenv
from openai import OpenAI

from sentence_transformers import SentenceTransformer
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import declarative_base, sessionmaker


class Config:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    DATABASE_URL = "sqlite:///user_preferences.db"
    SENTENCE_TRANSFORMER_MODEL = "all-MiniLM-L6-v2"
    FAISS_INDEX_PATH = "paintings_index.faiss"
    METADATA_PATH = "metdata.json"


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


engine = sa.create_engine(Config.DATABASE_URL, echo=False)
Base.metadata.create_all(engine)
session_factory = sessionmaker(bind=engine)


global preferences_manager
preferences_manager = UserPreferencesManager(session_factory)


# Tool Implementations
def initialize_search_index():
    """Initialize FAISS index and painting metadata"""
    index = faiss.read_index(Config.FAISS_INDEX_PATH)
    with open(Config.METADATA_PATH, "r") as file:
        data = json.load(file)
    return index, data["paintings"]


global_index, global_paintings = initialize_search_index()


def index_search(query: str, top_k: int):
    """Perform semantic search on paintings"""
    model = SentenceTransformer(Config.SENTENCE_TRANSFORMER_MODEL)
    query_embedding = model.encode([query], convert_to_tensor=False)
    query_embedding = np.array(query_embedding).astype("float32")
    distances, indices = global_index.search(query_embedding, top_k)
    return [global_paintings[i] for i in indices[0]]


def _update_user_preferences(preferences: Dict) -> None:
    """Update user preferences based on interactions"""
    user_id = 1
    preferences_manager.update_preferences(user_id, preferences)


# Tool
def search_paintings(query: str, top_k: Optional[str]) -> str:
    """
    Searches the FAISS index for the most relevant paintings based on a user query.

    Args:
        query (str): The user's input query.
        top_k (int): Number of top results to return.

    Returns:
        str: A string representation of a list of dictionaries containing painting metadata.
    """
    top_k = int(top_k) if top_k is not None else 3

    results = index_search(query, top_k)

    # Track user interaction
    painting = results[0]["title"]
    _update_user_preferences({"visited_painting": painting})

    return str(results)


# Tool
def search_related_paintings(painting: str) -> str:
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
    )

    # Construct related painting info
    related_painting_info = "\n".join(
        [
            f"Title: {p['painting']}\nArtist: {p['artist']}\nWhy it's related: {p['reason']}"
            for p in result[0]["related_paintings"]
        ]
    )
    return related_painting_info


# Tool
def get_personalized_recommendations() -> Dict:
    """
    Generate personalized museum experience recommendations based on user preferences.

    Args:
        config (RunnableConfig): Configuration containing user ID.

    Returns:
        Dict: Personalized recommendations based on user preferences.
    """
    user_id = 1
    return preferences_manager.get_personalized_recommendations(
        user_id, global_index, global_paintings
    )


# Tool
def set_user_preferences(art_styles: str, artists: str) -> None:
    """Useful for when you want to update user's favorite art styles and artists based on interactions"""
    preferences = {}
    if art_styles:
        preferences["art_styles"] = art_styles
    if artists:
        preferences["favorite_artists"] = artists
    _update_user_preferences(preferences)


# Tool
def wikipedia(q):
    return httpx.get(
        "https://en.wikipedia.org/w/api.php",
        params={"action": "query", "list": "search", "srsearch": q, "format": "json"},
    ).json()["query"]["search"][0]["snippet"]


# Tool
def getPaintingInfo(painting: str) -> str:
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


class AIAgent:
    """An AI agent that can use tools to answer questions through a chat interface."""

    def __init__(self, model: str, base_url: str, temperature: float = 0):
        """
        Initialize the AI agent with tools and OpenAI configuration.

        Args:
            model: The OpenAI model to use
            temperature: The sampling temperature for responses
        """
        _ = load_dotenv()
        self.client = OpenAI(
            base_url=base_url,
            api_key=os.getenv("GEMINI_API_KEY"),
        )
        self.model = model
        self.temperature = temperature
        self.tools: Dict[str, Callable] = {}
        self.messages = []
        self.action_re = re.compile(r"Action:\s*(\w+)\s*\((.*?)\)")

    def register_tool(self, name: str, func: Callable) -> None:
        """Register a tool that the agent can use."""
        self.tools[name] = func

    def _function_to_string(self, func: Callable) -> str:
        """Convert a function to its source code string."""
        return inspect.getsource(func)

    def _parse_arguments(self, args_str: str) -> List[str]:
        """Parse comma-separated arguments handling quoted strings."""
        reader = csv.reader(StringIO(args_str))
        args = next(reader)
        return [arg.strip().strip("\"'") for arg in args]

    def add_message(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})
        if self.callback:
            self.callback({"type": "message", "role": role, "content": content})

    def get_chat_history(self):
        return "\n".join([f"{m['role']}: {m['content']}" for m in self.messages])

    def setup_system_prompt(self) -> None:
        """Set up the system prompt with available tools."""

        prompt = """
            You run in a loop of Thought, Action, PAUSE, Observation.
            At the end of the loop you output an Answer
            Use Thought to describe your thoughts about the question you have been asked.
            Use Action to run one of the actions available to you - then return PAUSE.
            Observation will be the result of running those actions.

            Previous reasoning and observations:
            {}

            Your available actions are:
            {}

            Example session:

            Question: What is the painting with the swirling sky and vibrant night scene?
            Thought: I should search for three paintings that match the description given.
            Action: search_paintings("Swirling sky, vibrant night scene", 1)
            PAUSE

            You will be called again with this:

            Observation: A list of dictionaries containing painting metadata.
            
            You then output:

            Answer: The painting is Starry Night by Vincent van Gogh, created in 1889. The swirling sky and vibrant night scene are notable features of this Post-Impressionist artwork. The painting is known for its bold colors and expressive brushwork. Some conversational prompts include: "What do you think the swirling sky represents?" and "How does the vibrant night scene make you feel?"
            ,
            Questions: Can you tell me about the artist that painted the Starry Night and what do experts say about this painting?
            Thought: I need to find out about the artist and the expert insights about the painting Starry Night. I will first search wikipedia for a biography of the artist and then get the expert insights about the painting from the database.

            Action: wikipedia("Vincent van Gogh")
            PAUSE
            -- running wikipedia ['Vincent van Gogh']
            Observation: Vincent van Gogh was a Dutch post-impressionist painter who is among the most famous and influential figures in the history of Western art. In just over a decade, he created about 2,100 artworks, including around 860 oil paintings, most of which date from the last two years of his life.
            Result: Action: getPaintingInfo("Starry Night")
            Observation: A list of dictionaries containing painting metadata and expert insights.
            Result: Answer: Vincent van Gogh was a Dutch post-impressionist painter who is among the most famous and influential figures in the history of Western art. In just over a decade, he created about 2,100 artworks, including around 860 oil paintings, most of which date from the last two years of his life. The painting Starry Night is known for its bold colors and expressive brushwork.
            Final answer: Vincent van Gogh was a Dutch post-impressionist painter who is among the most famous and influential figures in the history of Western art. In just over a decade, he created about 2,100 artworks, including around 860 oil paintings, most of which date from the last two years of his life. The painting Starry Night is known for its bold colors and expressive brushwork.

            """.strip()

        actions_str = [self._function_to_string(func) for func in self.tools.values()]
        system = prompt.format(self.get_chat_history(), actions_str)
        self.messages = [{"role": "system", "content": system}]

    def query(self, question: str, max_turns: int = 10) -> Optional[str]:
        """
        Process a question through multiple turns until getting final answer.

        Args:
            question: The input question to process
            max_turns: Maximum number of turns before timing out

        Returns:
            Optional[str]: The final answer or None if no answer found
        """

        self.setup_system_prompt()
        next_prompt = question

        try:
            for i in range(max_turns):
                self.messages.append({"role": "user", "content": next_prompt})
                result = self._execute()
                self.messages.append({"role": "assistant", "content": result})
                print(f"Result: {result}")

                if result.lower().startswith("Answer:"):
                    return result.split("Answer:", 1)[1].strip()

                actions = [
                    self.action_re.match(a)
                    for a in result.split("\n")
                    if self.action_re.match(a)
                ]

                if actions:
                    action, args_str = actions[0].groups()
                    action_inputs = self._parse_arguments(args_str)

                    tool = self.tools.get(action)
                    if not tool:
                        raise Exception(f"Unknown action: {action}")

                    print(f" Calling Function {action} with {action_inputs}")
                    observation = tool(*action_inputs)
                    print(f"Observation: {observation}")
                    next_prompt = f"Observation: {observation}"
                else:
                    return None

        except Exception as e:
            print(f"Error during query processing: {str(e)}")
            return None

        return None

    def _execute(self) -> str:
        """Execute a chat completion request."""
        completion = self.client.chat.completions.create(
            model=self.model, temperature=self.temperature, messages=self.messages
        )
        return completion.choices[0].message.content


if __name__ == "__main__":
    # Create and configure agent

    # base_url = "http://localhost:11434/v1"
    # model = "gemma2:2b"

    base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
    model = "gemini-1.5-flash"

    agent = AIAgent(model=model, base_url=base_url)

    # Register tools ***** ADD OR REMOVE TOOLS HERE ***** <-- TIFFANY
    agent.register_tool("search_paintings", search_paintings)
    agent.register_tool("search_related_paintings", search_related_paintings)
    agent.register_tool(
        "get_personalized_recommendations", get_personalized_recommendations
    )
    agent.register_tool("set_user_preferences", set_user_preferences)
    agent.register_tool("getPaintingInfo", getPaintingInfo)
    agent.register_tool("wikipedia", wikipedia)

    # # Run a query
    # question = (
    #     "What painting is the one with three girls in red dresses peeling potatoes?"
    # )
    # answer = agent.query(question)
    # if answer:
    #     print(f"Final answer: {answer}")
    # else:
    #     print("Could not find an answer")

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
        answer = agent.query(question)
        print(f"Question: {question}")
        if answer:
            print(f"Final answer: {answer}")
        else:
            print("Could not find an answer")
