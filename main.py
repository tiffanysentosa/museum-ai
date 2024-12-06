import streamlit as st
import sys
import os
import google.generativeai as genai
import time
class Config:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    DATABASE_URL = "sqlite:///user_preferences.db"
    SENTENCE_TRANSFORMER_MODEL = "all-MiniLM-L6-v2"
    FAISS_INDEX_PATH = "paintings_index.faiss"
    METADATA_PATH = "metdata.json"

# Ensure the directory containing tools.py is in the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tools import (
    guess_painting,
    top_matches,
    get_painting_details,
    get_painting_response,
    get_painting_recommendation,
    get_related_paintings,
    get_painting_directions
)

def main():
    user_input = "I see a painting with 3 girls in red dresses peeling potatoes."
    matches = top_matches(user_input)
    response = guess_painting(user_input, matches)
    print("-"*100)
    print("Guess Painting Response\n")
    print(response['response'])
    print(response['painting_id'])
    painting_details = get_painting_details(response['painting_id'])
    print("-"*100)
    print("Painting Details\n")
    print(f"Title: {painting_details['title']}")
    print("-"*3)
    print(f"Artist: {painting_details['artist']}")
    print("-"*3)
    print(f"Visual Description: {painting_details['visual description']}")
    print("-"*3)
    print(f"Narrative: {painting_details['narrative']}")

    painting_response = get_painting_response(painting_details)
    print("-"*100)
    print("Painting Response\n")
    print(painting_response)
    related_paintings = get_related_paintings(response['painting_id'])
    painting_recommendation = get_painting_recommendation(related_paintings, painting_details, "yes")
    print("-"*100)
    print("Painting Recommendation\n")
    print(painting_recommendation)
    recommended_painting_name = painting_recommendation['recommended_painting_name']
    painting_directions = get_painting_directions(response['painting_id'], recommended_painting_name)
    print("-"*100)
    print("Painting Name\n")
    print(recommended_painting_name)
    print("-"*100)
    print("Painting Directions\n")
    print(painting_directions)
    
    wait_time = 60
    print(f"Waiting for {wait_time} seconds before proceeding...")
    time.sleep(wait_time)
    
    # Ask if they see the painting
    print("Do you see the painting? (yes/no)")
    see_painting = input().lower()
    
    if see_painting == "yes":
        print("Would you like to know more about it? (yes/no)")
        learn_more = input().lower()
        
        if learn_more == "yes":
            painting_details = get_painting_details(painting_recommendation['recommended_painting_id'])
            painting_response = get_painting_response(painting_details)
            print("\nHere's what makes this painting special:")
            print(painting_response)
    else:
        print("No problem! Take your time to locate the painting. Let me know when you find it.")

if __name__ == "__main__":
    main()

# Instructions for running:
# 1. Ensure you have the required dependencies installed: