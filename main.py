import streamlit as st
import sys
import os
import google.generativeai as genai
import time
class Config:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
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
    get_painting_qa
)

def get_initial_input():
    nothing_count = 0  # Add counter for "nothing" responses
    while True:
        print("\nWhat catches your eye?")
        user_input = input().strip()
        
        if not user_input or user_input.lower() == "nothing":
            nothing_count += 1
            if nothing_count >= 2:
                print("No problem! Feel free to come back when something catches your eye! Hope you had a good experience!")
                sys.exit()  # Exit the program
            print("Oh no! Why don't you take some time to look around! Let me know when you find a painting that catches your eye!")
            continue
        return user_input

def process_painting_description(user_input):
    # Get matches and guess the painting
    matches = top_matches(user_input)
    response = guess_painting(user_input, matches)
    #print("-"*100)
    #print(response)
    print(response['response'])
    painting_id = response['painting_id']
    #print(painting_id)
    return response, painting_id

def painting_background(painting_id):
    while True:
        print("\nWould you like to know more about this painting? Please answer with 'yes' or 'no'")
        user_response = input().strip().lower()
        if user_response == 'yes':
            painting_details = get_painting_details(painting_id)
            painting_response = get_painting_response(painting_details)
            print(painting_response)
            return painting_response
        elif user_response == 'no':
            return False
        else:
            print("Please answer with 'yes' or 'no'")

def painting_recommendation(response, painting_details):
    # if yes, get related paintings 
    related_paintings = get_related_paintings(response['painting_id'])
    painting_recommendation = get_painting_recommendation(related_paintings, painting_details, "yes")
    print(painting_recommendation)
    return painting_recommendation

def handle_painting_qa(painting_id):
    while True:
        print("\nDo you have any questions about this painting? Please answer with 'yes' or 'no'")
        response = input().strip().lower()
        
        if response == 'no':
            return False
        elif response == 'yes':
            while True:
                print("\nWhat would you like to know?")
                question = input().strip()
                question_answer = get_painting_qa(painting_id, question)
                print(question_answer)
            
                print("\nDo you have any other questions? Please answer with 'yes' or 'no'")
                more_questions = input().strip().lower()
                if more_questions != 'yes':
                    return True
                continue
        else:
            print("Please answer with 'yes' or 'no'")

def handle_painting_recommendation(painting_id, painting_details, visited_paintings):
    related_paintings = get_related_paintings(painting_id)
    
    if not related_paintings:
        print("I'm sorry, I couldn't find any related paintings.")
        return False, None
    
    while True:  # Loop to handle multiple recommendations
        # Debug prints
        print("DEBUG - Visited paintings:", visited_paintings)
        print("DEBUG - Available related paintings:", [p['painting'] for p in related_paintings['related_paintings']])
        
        recommendation = get_painting_recommendation(related_paintings, painting_details, "yes", visited_paintings)  # Pass visited_paintings
        if not recommendation:
            print("I'm sorry, we've already visited all the related paintings!")
            return False, None
            
        print("\n" + recommendation['response'])
        
        print("\nWould you like directions to this painting? Please answer with 'yes' or 'no'")
        want_directions = input().strip().lower()
        
        if want_directions == 'yes':
            success, recommended_painting_id = handle_painting_directions(painting_id, recommendation)
            if success:
                return True, recommendation['recommended_painting_id']
            else:
                print("\nWould you like me to recommend another related painting? Please answer with 'yes' or 'no'")
                another_recommendation = input().strip().lower()
                if another_recommendation == 'yes':
                    continue
                else:
                    return False, None
        elif want_directions == 'no':
            print("\nWould you like me to recommend another related painting? Please answer with 'yes' or 'no'")
            another_recommendation = input().strip().lower()
            if another_recommendation == 'yes':
                continue
            else:
                print("Why don't you let me know when you have another painting in mind!")
                return False, None

def handle_painting_directions(current_painting_id, recommendation):
    print("\nI'll give you a few to get there...")
    recommended_painting = recommendation['recommended_painting']
    
    # Check which location key is available
    if 'location_to_painting' in recommended_painting:
        directions = recommended_painting['location_to_painting']
    elif 'location_from_painting' in recommended_painting:
        directions = recommended_painting['location_from_painting']
    else:
        print("I'm sorry, I couldn't find directions to that painting.")
        return False, None
        
    while True:
        print(directions)
        print("\nDo you see the painting? Please answer with 'yes' or 'no'")
        found_painting = input().strip().lower()
        
        if found_painting == 'no':
            print("Feel free to ask a gallery attendant!")
            print("\nWould you like to try finding it again? Please answer with 'yes' or 'no'")
            try_again = input().strip().lower()
            if try_again == 'yes':
                continue
            else:
                return False, None
        elif found_painting == 'yes':
            return True, recommendation['recommended_painting_id']
        else:
            print("Please answer with 'yes' or 'no'")

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
            has_questions = handle_painting_qa(painting_id)
            
            # Ask before giving recommendation
            print("\nCould I recommend another painting I think you'd like? Please answer with 'yes' or 'no'")
            want_recommendation = input().strip().lower()
            
            if want_recommendation == 'yes':
                success, recommended_painting_id = handle_painting_recommendation(painting_id, painting_details, visited_paintings)  # Pass visited_paintings
                if success and recommended_painting_id:
                    painting_id = recommended_painting_id
                    painting_details = get_painting_details(painting_id)
                    visited_paintings.add(painting_id)  # Add new painting to visited set
                    continue
                break
            if want_recommendation == 'no':
                print("\nWould you like to find another painting that catches your eye or end the service? Please answer with 'find' or 'end'")
                choice = input().strip().lower()
                if choice == 'end':
                    print("\nThank you for using this guide. I hope you had a good experience!")
                    return
                elif choice == 'find':
                    print("\nGreat! Let me know what catches your eye!")
                    break
                else:
                    print("Please answer with 'find' or 'end'")
            else:
                print("\nThank you for using this guide. I hope you had a good experience!")
                return

if __name__ == "__main__":
    main()

# Instructions for running:
# 1. Ensure you have the required dependencies installed
# 2. Set the GEMINI_API_KEY environment variable
# 3. Run the script
