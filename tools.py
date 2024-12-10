import faiss
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import json
import google.generativeai as genai
from dotenv import load_dotenv
import os
import random

load_dotenv()

class Config:
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
    DATASET_PATH = "metdata.json"

# Initialize Gemini at module level
genai.configure(api_key=Config.GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-1.5-pro')

DATASET_PATH = Config.DATASET_PATH
dataset = None

def load_dataset():
    global dataset
    if dataset is None:
        with open(DATASET_PATH, 'r') as f:
            data = json.load(f)
            dataset = data['paintings']
    return dataset

def top_matches(user_input, top_k=3):
    """
    Identifies the best-matching painting based on user input using FAISS.

    Args:
        user_input (str): The description of the painting provided by the user.
        top_k (int): Number of top matches to retrieve.

    Returns:
        list: The top-k closest paintings with their details.
    """
    paintings_data = load_dataset() 

    # Check dataset integrity
    for i, entry in enumerate(paintings_data):
        if not isinstance(entry, dict):
            raise ValueError(f"Dataset entry at index {i} is not a dictionary: {entry}")

    text_data = [
        f"{entry['description']} {entry['visual_description']['notable_elements']}"
        for entry in paintings_data
    ]

    # Use TF-IDF to convert text data to embeddings
    vectorizer = TfidfVectorizer()
    embeddings = vectorizer.fit_transform(text_data).toarray()

    # Initialize FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings, dtype=np.float32))

    # Convert user input to the same embedding space
    user_embedding = vectorizer.transform([user_input]).toarray().astype(np.float32)

    # Perform FAISS search
    _, indices = index.search(user_embedding, top_k)

    # Retrieve the top-k matching paintings
    results = [paintings_data[idx] for idx in indices[0]]

    return results

def guess_painting(user_input, results):
    """
    Agent guesses the painting based on the top matches given to us by top_matches.

    Args:
        results (list): The top-k closest paintings with their details.
        model: Gemini generative model instance.

    Returns:
        str: LLM response with a guess of the painting.
    """
    descriptions = "\n".join(
        [
            f"- Title: {painting['title']}, Artist: {painting['artist']}, "
            f"Description: {painting['description']}"
            for painting in results
        ]
    )

    prompt = f"""
    You are a art guide. Based on the following top 3 painting matches and the user's input, choose the most likely match 
    and craft a short, engaging response. The response should include:
    - The painting's title in quotes (e.g., "The Three Sisters").
    - The painting's artist.
    - A brief description from the details provided.

    User's input: {user_input}

    Here are the top 3 matches:
    {descriptions}

    Example output:
    Great painting! That is "The Three Sisters", a painting by Leon Frederic. It depicts three girls in red dresses peeling potatoes. The painting is noted for its realistic style and the way it captures a quiet moment of domestic life. 
    """

    try:
        response = model.generate_content(prompt)
        crafted_response = response.text.strip()
        
        # Find the matching painting from results based on the response
        selected_painting = None
        for painting in results:
            if painting['title'].lower() in crafted_response.lower():
                selected_painting = painting
                break
        
        # Store the painting ID if found
        painting_id = selected_painting['id'] if selected_painting else None
        
        crafted_response += " Would you like to know more about it?"

        # Return both the response and the painting ID
        return {
            'response': crafted_response,
            'painting_id': painting_id
        }
    except Exception as e:
        return {
            'response': f"An error occurred while generating the response: {str(e)}",
            'painting_id': None
        }

def get_painting_details(painting_id):
    """
    Retrieves the details of a painting from the database based on its ID.
    """
    paintings_data = load_dataset()
    for painting in paintings_data:
        if painting['id'] == painting_id:
            painting_details = {
                'title': painting['title'],
                'artist': painting['artist'], 
                'visual description': painting['visual_description'],
                'narrative': painting['narrative']
            }
            return painting_details
    return None

def get_painting_response(painting_details):
    """
    Generates a response to the user based on the painting details.

    Args:
        painting_details (dict): The details of the painting, including title, artist, visual description, and narrative.
        model: The generative model instance (e.g., Gemini).

    Returns:
        str: A detailed description of the painting.
    """
    # Construct the prompt using the painting details
    prompt = f"""
    You are an expert art guide. Craft a detailed and engaging description of the painting based on the following information.
    
    Style guide:
    - Do not include markdown formatting in the response. 

    Painting Details:
    - Title: {painting_details['title']}
    - Artist: {painting_details['artist']}
    - Visual Description: {painting_details['visual description']}
    - Narrative: {painting_details['narrative']}

    Provide an output in the following style:
    "The Three Sisters" by Leon Frederic (1896) portrays a tender, quiet moment of domestic life through the scene of three sisters peeling potatoes. Their bright red dresses, paired with glowing red-gold and blond hair, contrast with the earthy tones of their surroundings, bringing warmth and life to an otherwise humble task. Their serene expressions and downcast eyes evoke a sense of introspection, reminiscent of the Virgin Mary’s portrayal in sixteenth-century Flemish art, which Frederic admired. 

    Painted in the Realist style with oil on canvas, the work draws attention to the beauty in simplicity—such as the twirling potato skins, akin to locks of hair. While the identities of the sisters remain unknown, their shared task emphasizes family unity and the nurturing bonds formed through everyday routines. The painting's symbolism suggests themes of innocence, dedication, and the sacred nature of ordinary life. 

    Through this work, Frederic bridges the honesty of Realist domestic scenes with a spiritual undertone, offering a poignant glimpse into the timeless significance of familial connections.

    Use a similar tone and style. Highlight the painting's key visual elements, symbolic meaning, and narrative context. Keep the response concise but descriptive.
    """

    try:
        response = model.generate_content(prompt)
        crafted_response = response.text.strip()
        crafted_response = crafted_response
        return crafted_response
    except Exception as e:
        return f"An error occurred while generating the response: {str(e)}"
    
def get_related_paintings(painting_id):
    """
    Retrieves related paintings based on the painting ID.
    """
    paintings_data = load_dataset()
    for painting in paintings_data:
        if painting['id'] == painting_id:
            related_paintings = {
                'related_paintings': painting['related_paintings']
            }
            return related_paintings
    return None

def get_painting_recommendation(related_paintings, painting_details, choice, visited_paintings):
    # First, get IDs for all related paintings
    paintings_data = load_dataset()
    available_paintings = []
    
    # Build list of available paintings with their IDs
    for related in related_paintings['related_paintings']:
        for painting in paintings_data:
            if painting['title'].lower() == related['painting'].lower():
                available_paintings.append({
                    'id': painting['id'],
                    'title': painting['title'],
                    'reason': related['reason'],
                    'location_from_painting': related.get('location_from_painting', '')
                })
                break
    
    # Filter out visited paintings by ID
    unvisited_paintings = [p for p in available_paintings if p['id'] not in visited_paintings]
    
    print("DEBUG - Visited IDs:", visited_paintings)
    print("DEBUG - Available painting IDs:", [p['id'] for p in available_paintings])
    print("DEBUG - Unvisited painting IDs:", [p['id'] for p in unvisited_paintings])
    
    if not unvisited_paintings:
        print("DEBUG - No unvisited paintings remaining")
        return None
    
    # Randomly select one unvisited painting
    selected_painting = random.choice(unvisited_paintings)
    
    # Craft the response
    crafted_response = f"If you enjoyed {painting_details['title']}, I highly recommend viewing '{selected_painting['title']}'. {selected_painting['reason']}"
    
    return {
        'response': crafted_response,
        'recommended_painting_name': selected_painting['title'],
        'recommended_painting': {
            'painting': selected_painting['title'],
            'location_from_painting': selected_painting['location_from_painting']
        },
        'recommended_painting_id': selected_painting['id']
    }

def get_painting_directions(painting_id, recommended_painting_name):   
    """
    Retrieves the directions to the recommended painting from the database.
    
    Args:
        painting_id (str): ID of the original painting
        recommended_painting_name (str): Name of the recommended painting
        
    Returns:
        str: Directions to the recommended painting's location, or error message if not found
    """
    paintings_data = load_dataset()
    
    # Find the original painting
    for painting in paintings_data:
        if painting['id'] == painting_id:
            # Search through related paintings
            for related in painting['related_paintings']:
                if related['painting'].lower() == recommended_painting_name.lower():
                    response = related['location_from_painting'] + "\n\nI'll give you a minute to get there."
                    return response
    return "I'm sorry, I don't have directions to that painting at the moment."

def get_painting_qa(painting_id, question):
    """
    Retrieves the answer to a question by grabbing painting details and sending it to the LLM to answer the question.
    
    Args:
        painting_id (str): The ID of the painting being discussed
        question (str): The user's question about the painting
        
    Returns:
        str: The AI-generated answer to the question
    """
    # Get painting details
    #doesn't work right now
    painting_details = get_painting_details(painting_id)
    if not painting_details:
        return "I'm sorry, I couldn't find information about that painting."
    
    prompt = f"""
    You are an expert art guide. Answer the following question about this painting based on the provided details.
    Keep your response concise but informative and engaging.
    
    Painting Details:
    - Title: {painting_details['title']}
    - Artist: {painting_details['artist']}
    - Visual Description: {painting_details['visual description']}
    - Narrative: {painting_details['narrative']}
    
    Question: {question}
    
    Style guide:
    - Be conversational and friendly
    - Stick to facts from the provided details
    - If the question cannot be answered from the provided details, politely say so
    """
    
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"I apologize, but I encountered an error while answering your question: {str(e)}"

    
