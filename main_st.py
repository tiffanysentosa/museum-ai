import streamlit as st
import sys
import os

# Ensure the directory containing memory_agent.py is in the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from memory_agent import (
    AIAgent,
    search_paintings,
    search_related_paintings,
    get_personalized_recommendations,
    set_user_preferences,
    getPaintingInfo,
    wikipedia,
)


def initialize_agent():
    # base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
    # model = "gemini-1.5-flash"

    base_url = "http://localhost:11434/v1"
    model = "gemma2:2b"

    agent = AIAgent(model=model, base_url=base_url)
    agent.register_tool("search_paintings", search_paintings)
    agent.register_tool("search_related_paintings", search_related_paintings)
    # agent.register_tool(
    #     "get_personalized_recommendations", get_personalized_recommendations
    # )
    # agent.register_tool("set_user_preferences", set_user_preferences)
    agent.register_tool("getPaintingInfo", getPaintingInfo)
    # agent.register_tool("wikipedia", wikipedia)

    return agent


def main():
    st.title("🎨 Art Explorer Chatbot")

    # Initialize session state for agent and chat history
    if "agent" not in st.session_state:
        st.session_state.agent = initialize_agent()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("Ask me about art, paintings, or artists!"):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)

        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Get agent response
        with st.spinner("Thinking..."):
            response = st.session_state.agent.query(prompt)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response or "I couldn't find a good answer to that query.")

        # Add assistant response to chat history
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": response or "I couldn't find a good answer to that query.",
            }
        )


if __name__ == "__main__":
    main()

# Instructions for running:
# 1. Ensure you have the required dependencies installed:
#    pip install streamlit openai python-dotenv sentence-transformers faiss-cpu sqlalchemy httpx
# 2. Set up your .env file with the necessary API keys
# 3. Run the app with: streamlit run art_agent_streamlit.py
