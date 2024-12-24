import faiss
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import json

# import google.generativeai as genai
from dotenv import load_dotenv
import os
import random
from openai import OpenAI
import time
import csv

load_dotenv()
from transformers import AutoTokenizer
from config import Config

# Access the singleton instance of Config
Config = Config()




DATASET_PATH = Config.DATASET_PATH
dataset = None


def count_tokens(input_text):
    # Tokenize input text
    tokens = Config.TOKENIZER.tokenize(input_text)
    # Count the number of tokens
    num_tokens = len(tokens)
    return num_tokens


class ModelConfig:
    def __init__(self, model: str, api_key: str, base_url: str):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        # self.client = Config.CLIENT

   
    def generate_content(self, content: str, user_input, csv_file="metrics_log.csv"):
        system_prompt = """
        You are an expert art guide. Answer the following question about this painting based on the provided details.
        Keep your response concise but informative and engaging."""

        start_time = time.time()
        completion = Config.CLIENT.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
            stream=True,
        )

        collected_messages = []
        latencies = []  # To store (input tokens, output tokens, latency)
        num_output_tokens = 0  # Total number of output tokens
        time_to_first_token = 0  # Time to first token

        for chunk in completion:
            chunk_time = time.time() - start_time

            # Capture time to first token
            if len(latencies) == 0:
                time_to_first_token = chunk_time

            # Extract content if present
            if chunk.choices[0].delta.content is not None:
                chunk_text = chunk.choices[0].delta.content
                collected_messages.append(chunk_text)
                num_output_tokens += len(Config.TOKENIZER.tokenize(chunk_text))
                latencies.append(chunk_time)

                print(chunk_text)

        total_time = (
            latencies[-1] if latencies else 0
        )  # Total latency (last token's time)
        time_per_output_token = (
            (latencies[-1] - latencies[0]) / (len(latencies) - 1)
            if len(latencies) > 1
            else 0
        )
        num_total_output_tokens = len(
            Config.TOKENIZER.tokenize("".join(collected_messages))
        )
        throughput = num_total_output_tokens / total_time if total_time > 0 else 0

        # Print metrics
        print(f"Time to first token (s): {round(time_to_first_token, 2)}")
        print(f"Total time for output (s): {round(total_time, 2)}")
        print(f"Time per output token (ms): {round(time_per_output_token * 1000, 2)}")
        print(f"Throughput (tokens/sec): {round(throughput, 2)}")

        # Write metrics to a CSV file
        metrics = {
            "input": user_input,
            "time_to_first_token": round(time_to_first_token, 2),
            "total_time": round(total_time, 2),
            "time_per_output_token": round(time_per_output_token * 1000, 2),
            "throughput": round(throughput, 2),
            "num_total_output_tokens": num_total_output_tokens,
        }

        file_exists = os.path.isfile(csv_file)
        with open(csv_file, mode="a", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=metrics.keys())
            if not file_exists:
                writer.writeheader()  # Write the header if file is new
            writer.writerow(metrics)  # Append the metrics

        return "".join(collected_messages)


model = ModelConfig(
    model=Config.MODEL, api_key=Config.GEMINI_API_KEY, base_url=Config.BASE_URL
)





def interpret_user_response(
    user_input, context=None, csv_file="interpretation_log.csv"
):
    """
    Use the LLM to interpret user input and determine intent.

    Args:
        user_input (str): User's raw input
        context (dict, optional): Additional context about the current interaction
        csv_file (str): Path to the CSV file where metrics will be logged

    Returns:
        dict: Interpreted response with intent and details
    """
    try:
        # Create a context-aware prompt to interpret user intent
        prompt = f"""You are an AI assistant in an art gallery, helping a visitor navigate and understand paintings.
        Interpret the following user input and determine the intent:

        User Input: "{user_input}"
        
        Possible Intents:
        1. Affirmative (wants to proceed)
        2. Negative (wants to stop or decline)
        
        Provide a JSON response with the following structure:
        {{
            "intent": "...", # One of the intents above
            "confidence": 0.0, # Confidence level (0.0 to 1.0)
            "explanation": "...", # Brief explanation of how you interpreted the input
        }}
        
        Context (if available): {context}
        """

        # Record start time for response generation
        start_time = time.time()

        # Generate response from the LLM
        completion = Config.CLIENT.chat.completions.create(
            model=Config.MODEL,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_input},
            ],
            max_tokens=300,  # Limit response length
        )

        # Measure response time
        response_time = time.time() - start_time
        print(f"Time to interpret user's decision: {response_time:.2f}s")

        # Extract and parse the response
        response_text = completion.choices[0].message.content

        try:
            # Attempt to parse the JSON response
            interpretation = json.loads(response_text)
            print("INTERPRETATION", interpretation)
        except json.JSONDecodeError:
            # Fallback to a default interpretation if JSON parsing fails
            interpretation = {
                "intent": "unclear",
                "confidence": 0.5,
                "explanation": "Could not parse the exact intent",
            }

        # Prepare metrics for logging
        metrics = {
            "user_input": user_input,
            "llm_output": interpretation["explanation"],
            "response_time": round(response_time, 2),
        }

        # Log metrics to CSV file
        file_exists = os.path.isfile(csv_file)
        with open(csv_file, mode="a", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=metrics.keys())
            if not file_exists:
                writer.writeheader()  # Write the header if the file is new
            writer.writerow(metrics)  # Append the metrics

        return interpretation

    except Exception as e:
        print(f"Error in interpreting response: {e}")
        return {
            "intent": "error",
            "confidence": 0.0,
            "explanation": "Error in processing user input",
        }


def load_dataset():
    global dataset
    if dataset is None:
        with open(DATASET_PATH, "r") as f:
            data = json.load(f)
            dataset = data["paintings"]
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
        response = model.generate_content(prompt, user_input)
        crafted_response = response.strip()

        # Find the matching painting from results based on the response
        selected_painting = None
        for painting in results:
            if painting["title"].lower() in crafted_response.lower():
                selected_painting = painting
                break

        # Store the painting ID if found
        painting_id = selected_painting["id"] if selected_painting else None

        crafted_response += " Would you like to know more about it?"

        # Return both the response and the painting ID
        return {"response": crafted_response, "painting_id": painting_id}
    except Exception as e:
        return {
            "response": f"An error occurred while generating the response: {str(e)}",
            "painting_id": None,
        }


def get_painting_details(painting_id, context=None):
    """
    Retrieves the details of a painting from the database based on its ID.
    """
    paintings_data = load_dataset()
    for painting in paintings_data:
        if painting["id"] == painting_id:
            painting_details = {
                "title": painting["title"],
                "artist": painting["artist"],
                "year": painting["year"],
                "visual description": painting["visual_description"],
                "narrative": painting["narrative"],
            }
            if context:
                return painting_details, context
            return painting_details
    return None


def get_painting_response(painting_details, context=None):
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
    - Year: {painting_details['year']}
    - Visual Description: {painting_details['visual description']}
    - Narrative: {painting_details['narrative']}

    Provide an output in the following style:
    "The Three Sisters" by Leon Frederic (1896) portrays a tender, quiet moment of domestic life through the scene of three sisters peeling potatoes. Their bright red dresses, paired with glowing red-gold and blond hair, contrast with the earthy tones of their surroundings, bringing warmth and life to an otherwise humble task. Their serene expressions and downcast eyes evoke a sense of introspection, reminiscent of the Virgin Mary’s portrayal in sixteenth-century Flemish art, which Frederic admired. 

    Painted in the Realist style with oil on canvas, the work draws attention to the beauty in simplicity—such as the twirling potato skins, akin to locks of hair. While the identities of the sisters remain unknown, their shared task emphasizes family unity and the nurturing bonds formed through everyday routines. The painting's symbolism suggests themes of innocence, dedication, and the sacred nature of ordinary life. 

    Through this work, Frederic bridges the honesty of Realist domestic scenes with a spiritual undertone, offering a poignant glimpse into the timeless significance of familial connections.

    Use a similar tone and style. Highlight the painting's key visual elements, symbolic meaning, and narrative context. Keep the response concise but descriptive.

    Limit the response to the what the user asks for (if available): {context}

    """

    try:
        response = model.generate_content(
            prompt, f"painting details summary with context: {context}"
        )
        crafted_response = response.strip()
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
        if painting["id"] == painting_id:
            related_paintings = {"related_paintings": painting["related_paintings"]}
            return related_paintings
    return None


def get_painting_recommendation(
    related_paintings, painting_details, choice, visited_paintings
):
    # First, get IDs for all related paintings
    paintings_data = load_dataset()
    available_paintings = []

    # Build list of available paintings with their IDs
    for related in related_paintings["related_paintings"]:
        for painting in paintings_data:
            if painting["title"].lower() == related["painting"].lower():
                available_paintings.append(
                    {
                        "id": painting["id"],
                        "title": painting["title"],
                        "reason": related["reason"],
                        "location_from_painting": related.get(
                            "location_from_painting", ""
                        ),
                    }
                )
                break

    # Filter out visited paintings by ID
    unvisited_paintings = [
        p for p in available_paintings if p["id"] not in visited_paintings
    ]

    print("DEBUG - Visited IDs:", visited_paintings)
    print("DEBUG - Available painting IDs:", [p["id"] for p in available_paintings])
    print("DEBUG - Unvisited painting IDs:", [p["id"] for p in unvisited_paintings])

    if not unvisited_paintings:
        print("DEBUG - No unvisited paintings remaining")
        return None

    # Randomly select one unvisited painting
    selected_painting = random.choice(unvisited_paintings)

    # Craft the response
    crafted_response = f"If you enjoyed {painting_details['title']}, I highly recommend viewing '{selected_painting['title']}'. {selected_painting['reason']}"

    return {
        "response": crafted_response,
        "recommended_painting_name": selected_painting["title"],
        "recommended_painting": {
            "painting": selected_painting["title"],
            "location_from_painting": selected_painting["location_from_painting"],
        },
        "recommended_painting_id": selected_painting["id"],
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
        if painting["id"] == painting_id:
            # Search through related paintings
            for related in painting["related_paintings"]:
                if related["painting"].lower() == recommended_painting_name.lower():
                    response = (
                        related["location_from_painting"]
                        + "\n\nI'll give you a minute to get there."
                    )
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
    # doesn't work right now
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
        response = model.generate_content(prompt, question)
        return response.strip()
    except Exception as e:
        return f"I apologize, but I encountered an error while answering your question: {str(e)}"
