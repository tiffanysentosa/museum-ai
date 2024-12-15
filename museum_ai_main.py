import json
import streamlit as st
import sys
import os

# import google.generativeai as genai
import time
from openai import OpenAI
from config import Config
from transformers import AutoTokenizer


# Ensure the directory containing tools.py is in the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tools import (
    guess_painting,
    interpret_user_response,
    top_matches,
    get_painting_details,
    get_painting_response,
    get_painting_recommendation,
    get_related_paintings,
    get_painting_qa,
)


def process_painting_description(user_input):
    # Get matches and guess the painting
    matches = top_matches(user_input)
    response = guess_painting(user_input, matches)
    # print("-"*100)
    # print(response)
    print(response["response"])
    painting_id = response["painting_id"]
    # print(painting_id)
    return response, painting_id


def get_initial_input():
    nothing_count = 0
    while True:
        print("\nWhat catches your eye?")
        user_input = input().strip()

        # Interpret the input
        interpretation = interpret_user_response(user_input)

        if interpretation["intent"].lower() in ["unclear", "error"] or not user_input:
            nothing_count += 1
            if nothing_count >= 2:
                print(
                    "No problem! Feel free to come back when something catches your eye! Hope you had a good experience!"
                )
                sys.exit()
            print(
                "Oh no! Why don't you take some time to look around! Let me know when you find a painting that catches your eye!"
            )
            continue

        return user_input


def user_confirmation(prompt, context=None):
    """
    User confirmation with LLM interpretation

    Args:
        prompt (str): The question or confirmation request
        context (dict, optional): Additional context for interpretation

    Returns:
        bool: Interpreted user response
    """
    while True:
        print(f"\n{prompt}")
        user_input = input().strip()

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


def painting_background(painting_id):
    context = {"current_action": "painting_background", "painting_id": painting_id}
    want_details, context = user_confirmation(
        "Would you like to know more about this painting?", context
    )

    if want_details:
        painting_details = get_painting_details(painting_id)
        painting_response = get_painting_response(painting_details, context)
        print(painting_response)
        return painting_response
    return False


def handle_painting_qa(painting_id):
    context = {"current_action": "painting_qa", "painting_id": painting_id}
    want_qa, _ = user_confirmation(
        "Do you have any questions about this painting?", context
    )

    if want_qa:
        while True:
            print("\nWhat would you like to know?")
            question = input().strip()

            # Interpret the question's intent
            interpretation = interpret_user_response(question, context)

            if "affirmative" in interpretation["intent"].lower():
                question_answer = get_painting_qa(painting_id, question)
                print(question_answer)
            elif "negative" in interpretation["intent"].lower():
                break

            # Ask if they want to continue asking questions
            continue_qa, _ = user_confirmation(
                "Do you have any other questions?", context
            )
            if not continue_qa:
                break
        return True
    return False


def handle_painting_recommendation(painting_id, painting_details, visited_paintings):
    context = {
        "current_action": "painting_recommendation",
        "current_painting_id": painting_id,
    }
    want_recommendation, _ = user_confirmation(
        "Could I recommend another painting I think you'd like?", context
    )

    if want_recommendation:
        related_paintings = get_related_paintings(painting_id)

        if not related_paintings:
            print("I'm sorry, I couldn't find any related paintings.")
            return False, None

        while True:
            # print("DEBUG - Visited paintings:", visited_paintings)
            # print(
            #     "DEBUG - Available related paintings:",
            #     [p["painting"] for p in related_paintings["related_paintings"]],
            # )
            recommendation = get_painting_recommendation(
                related_paintings, painting_details, "yes", visited_paintings
            )

            if not recommendation:
                print("I'm sorry, we've already visited all the related paintings!")
                return False, None

            print("\n" + recommendation["response"])

            # More advanced confirmation for directions
            context["recommended_painting"] = recommendation
            want_directions, _ = user_confirmation(
                "Would you like directions to this painting?", context
            )

            if want_directions:
                success, recommended_painting_id = handle_painting_directions(
                    painting_id, recommendation
                )
                if success:
                    return True, recommendation["recommended_painting_id"]

            # Ask about another recommendation
            another_recommendation, _ = user_confirmation(
                "Would you like me to recommend another related painting?", context
            )

            if not another_recommendation:
                print(
                    "Why don't you let me know when you have another painting in mind!"
                )
                return False, None

    return False, None


def handle_painting_directions(current_painting_id, recommendation):
    print("\nI'll give you a few directions to get there...")
    recommended_painting = recommendation["recommended_painting"]

    # Check which location key is available
    if "location_to_painting" in recommended_painting:
        directions = recommended_painting["location_to_painting"]
    elif "location_from_painting" in recommended_painting:
        directions = recommended_painting["location_from_painting"]
    else:
        print("I'm sorry, I couldn't find directions to that painting.")
        return False, None

    while True:
        print(directions)

        # More advanced confirmation for finding the painting
        context = {
            "current_action": "finding_painting",
            "current_painting": recommended_painting,
        }
        found_painting, _ = user_confirmation("Do you see the painting?", context)

        if not found_painting:
            print("Feel free to ask a gallery attendant!")

            # Ask if they want to try again
            try_again, _ = user_confirmation(
                "Would you like to try finding it again?", context
            )

            if not try_again:
                return False, None
        else:
            return True, recommendation["recommended_painting_id"]


def main():
    visited_paintings = set()  # Create a set to track visited paintings

    while True:  # Outer loop for new paintings
        # Initial interaction
        user_input = get_initial_input()

        # Process painting description
        response, painting_id = process_painting_description(user_input)
        painting_details = get_painting_details(painting_id)

        # Add current painting to visited set
        visited_paintings.add(painting_id)

        while True:  # Inner loop for each painting
            # Handle painting background
            painting_background(painting_id)

            # Handle Q&A
            handle_painting_qa(painting_id)

            # Context for overall interaction
            context = {
                "current_action": "painting_exploration",
                "current_painting_id": painting_id,
            }

            # Ask about recommendation
            want_recommendation, _ = user_confirmation(
                "Could I recommend another painting I think you'd like?", context
            )

            if want_recommendation:
                success, recommended_painting_id = handle_painting_recommendation(
                    painting_id, painting_details, visited_paintings
                )
                if success and recommended_painting_id:
                    painting_id = recommended_painting_id
                    painting_details = get_painting_details(painting_id)
                    visited_paintings.add(painting_id)
                    continue
                break

            # Ask about next steps
            continue_exploring, _ = user_confirmation(
                "Would you like to find another painting that catches your eye?",
                {"current_action": "service_continuation"},
            )

            if not continue_exploring:
                print(
                    "\nThank you for using this guide. I hope you had a good experience!"
                )
                return


if __name__ == "__main__":
    config = Config()

    base_url = ("https://generativelanguage.googleapis.com/v1beta/openai/",)
    base_model = ("gemini-1.5-flash",)
    # base_url = "http://localhost:11434/v1"
    # base_model = "gemma2:2b"
    # Update configuration values
    config_update_params = {
        "api_key": os.environ["GEMINI_API_KEY"],
        "dataset_path": "metdata.json",
        "base_url": base_url,
        "model": base_model,
        "tokenizer": AutoTokenizer.from_pretrained("google/gemma-2-2b"),
        "CLIENT": OpenAI(
            api_key=os.environ["GEMINI_API_KEY"],
            base_url=base_url,
        ),
    }
    config.update_config(**config_update_params)
    main()
