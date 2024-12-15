import streamlit as st
import sys
import os
from tools import (
    guess_painting,
    top_matches,
    get_painting_details,
    get_painting_response,
    get_painting_recommendation,
    get_related_paintings,
    get_painting_qa,
    interpret_user_response,
)
from openai import OpenAI
from config import Config
from transformers import AutoTokenizer


def init_session_state():
    if "current_painting_id" not in st.session_state:
        st.session_state.current_painting_id = None
    if "current_painting_details" not in st.session_state:
        st.session_state.current_painting_details = None
    if "visited_paintings" not in st.session_state:
        st.session_state.visited_paintings = set()
    if "stage" not in st.session_state:
        st.session_state.stage = "initial"
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "What catches your eye?"}
        ]
    if "explanation" not in st.session_state:
        st.session_state.explanation = []


def user_confirmation(user_input, context=None):
    """
    User confirmation with LLM interpretation

    Args:
        prompt (str): The question or confirmation request
        context (dict, optional): Additional context for interpretation

    Returns:
        bool: Interpreted user response
    """
    while True:
        # print(f"\n{prompt}")
        user_input = user_input.strip()

        # Interpret the response
        interpretation = interpret_user_response(user_input, context)

        if "affirmative" in interpretation["intent"].lower():
            return True, interpretation["explanation"]
        elif "question" in interpretation["intent"].lower():
            return True, interpretation["explanation"]
        elif "negative" in interpretation["intent"].lower():
            return False, interpretation["explanation"]
        else:
            # If not clear, provide guidance
            print(
                f"I'm not sure I understood. {interpretation.get('explanation', 'Could you clarify?')}"
            )
            print("Please provide a clearer yes or no response.")


def main():
    st.title("Museum AI Guide")
    init_session_state()

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Get user input
    user_input = st.chat_input("Your response...")

    if user_input:
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Initial stage - What catches your eye?
        if st.session_state.stage == "initial":
            matches = top_matches(user_input)
            response = guess_painting(user_input, matches)
            st.session_state.messages.append(
                {"role": "assistant", "content": response["response"]}
            )
            st.session_state.current_painting_id = response["painting_id"]
            st.session_state.current_painting_details = get_painting_details(
                response["painting_id"]
            )
            st.session_state.visited_paintings.add(response["painting_id"])
            st.session_state.stage = "want_background"
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": "Would you like to know more about this painting?",
                }
            )

        # Want background stage
        elif st.session_state.stage == "want_background":
            context = {
                "current_action": "painting_background",
                "painting_id": st.session_state.current_painting_id,
            }
            # if user_input.lower() == "yes":
            confirmation, explanation = user_confirmation(user_input.lower(), context)
            if confirmation:
                painting_response = get_painting_response(
                    st.session_state.current_painting_details, explanation
                )
                st.session_state.messages.append(
                    {"role": "assistant", "content": painting_response}
                )
                st.session_state.stage = "want_qa"
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": "Would you like to ask any questions about this painting? (yes/no)",
                    }
                )
            # elif user_input.lower() == "no":
            else:
                st.session_state.stage = "want_qa"
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": "Would you like to ask any questions about this painting? (yes/no)",
                    }
                )

        # Want Q&A stage
        elif st.session_state.stage == "want_qa":
            # if user_input.lower() == "yes":
            context = {
                "current_action": "painting_qa",
                "painting_id": st.session_state.current_painting_id,
            }
            confirmation, explanation = user_confirmation(user_input.lower(), context)

            st.session_state.explanation.append(explanation)
            if confirmation:
                st.session_state.stage = "qa"

                st.session_state.messages.append(
                    {"role": "assistant", "content": "What would you like to know?"}
                )
            # elif user_input.lower() == "no":
            else:
                st.session_state.stage = "want_recommendation"
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": "Could I recommend another painting I think you'd like?",
                    }
                )

        # Q&A stage
        elif st.session_state.stage == "qa":
            answer = get_painting_qa(
                st.session_state.current_painting_id,
                user_input,
            )
            st.session_state.messages.append({"role": "assistant", "content": answer})
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": "Would you like to ask another question? (yes/no)",
                }
            )
            st.session_state.stage = "continue_qa"

        # Continue Q&A stage
        elif st.session_state.stage == "continue_qa":
            # if user_input.lower() == "no":
            context = {
                "current_action": "continuing_painting_qa",
                "painting_id": st.session_state.current_painting_id,
            }
            confirmation, explanation = user_confirmation(user_input.lower(), context)
            if not confirmation:
                st.session_state.stage = "want_recommendation"
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": "Could I recommend another painting I think you'd like?",
                    }
                )
            # elif user_input.lower() == "yes":
            else:
                st.session_state.stage = "qa"
                st.session_state.messages.append(
                    {"role": "assistant", "content": "What would you like to know?"}
                )

        # Want recommendation stage
        elif st.session_state.stage == "want_recommendation":
            # if user_input.lower() == "yes":
            confirmation, explanation = user_confirmation(user_input.lower())
            if confirmation:
                recommendation = get_painting_recommendation(
                    get_related_paintings(st.session_state.current_painting_id),
                    st.session_state.current_painting_details,
                    "yes",
                    st.session_state.visited_paintings,
                )
                if recommendation:
                    st.session_state.messages.append(
                        {"role": "assistant", "content": recommendation["response"]}
                    )
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": "Would you like directions to this painting?",
                        }
                    )
                    st.session_state.recommendation = recommendation
                    st.session_state.stage = "want_directions"
                else:
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": "I'm sorry, I couldn't find any new recommendations. Would you like to find a different painting?",
                        }
                    )
                    st.session_state.stage = "find_different"
            # elif user_input.lower() == "no":
            else:
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": "Would you like to find another painting or end the service? (find/end)",
                    }
                )
                st.session_state.stage = "find_or_end"

        # Want directions stage
        elif st.session_state.stage == "want_directions":
            # if user_input.lower() == "yes":
            confirmation, explanation = user_confirmation(user_input.lower())
            if confirmation:
                directions = st.session_state.recommendation["recommended_painting"][
                    "location_from_painting"
                ]
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": f"{directions}\n\nDo you see the painting?",
                    }
                )
                st.session_state.stage = "found_painting"
            # elif user_input.lower() == "no":
            else:
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": "Would you like me to recommend another related painting?",
                    }
                )
                st.session_state.stage = "want_recommendation"

        # Found painting stage
        elif st.session_state.stage == "found_painting":
            # if user_input.lower() == "yes":
            confirmation, explanation = user_confirmation(user_input.lower())
            if confirmation:
                st.session_state.current_painting_id = st.session_state.recommendation[
                    "recommended_painting_id"
                ]
                st.session_state.current_painting_details = get_painting_details(
                    st.session_state.recommendation["recommended_painting_id"]
                )
                st.session_state.visited_paintings.add(
                    st.session_state.recommendation["recommended_painting_id"]
                )
                st.session_state.stage = "want_background"
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": "Would you like to know more about this painting?",
                    }
                )
            # elif user_input.lower() == "no":
            else:
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": "Let me give you more detailed directions. Feel free to ask a gallery attendant! Would you like to try finding it again?",
                    }
                )
                st.session_state.stage = "try_again"

        # Try again stage
        elif st.session_state.stage == "try_again":
            # if user_input.lower() == "yes":
            confirmation, explanation = user_confirmation(user_input.lower())
            if confirmation:
                directions = st.session_state.recommendation["recommended_painting"][
                    "location_from_painting"
                ]
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": f"{directions}\n\nDo you see the painting?",
                    }
                )
                st.session_state.stage = "found_painting"
            # elif user_input.lower() == "no":
            else:
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": "Would you like me to recommend another related painting?",
                    }
                )
                st.session_state.stage = "want_recommendation"

        # Find or end stage
        elif st.session_state.stage == "find_or_end":
            # if user_input.lower() == "find":
            confirmation, explanation = user_confirmation(user_input.lower())
            if confirmation:
                st.session_state.stage = "initial"
                st.session_state.messages.append(
                    {"role": "assistant", "content": "What catches your eye?"}
                )
            # elif user_input.lower() == "end":
            else:
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": "Thank you for using this guide. I hope you had a good experience!",
                    }
                )
                st.session_state.stage = "end"

        # Find different stage
        elif st.session_state.stage == "find_different":
            # if user_input.lower() == "yes":
            confirmation, explanation = user_confirmation(user_input.lower())
            if confirmation:
                st.session_state.stage = "initial"
                st.session_state.messages.append(
                    {"role": "assistant", "content": "What catches your eye?"}
                )
            # elif user_input.lower() == "no":
            else:
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": "Thank you for using this guide. I hope you had a good experience!",
                    }
                )
                st.session_state.stage = "end"

        st.rerun()


if __name__ == "__main__":
    config = Config()

    # Update configuration values
    config.update_config(
        api_key=os.environ["GEMINI_API_KEY"],
        dataset_path="metdata.json",
        # base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        # model="gemini-1.5-flash",
        base_url="http://localhost:11434/v1",
        model="gemma2:2b",
        tokenizer=AutoTokenizer.from_pretrained("google/gemma-2-2b"),
        CLIENT=OpenAI(
            api_key=os.environ["GEMINI_API_KEY"],
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        ),
    )
    main()
