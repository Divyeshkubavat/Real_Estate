import streamlit as st
import pickle
import pandas as pd
import openai
import json
import re
import numpy as np

# Load the trained model
MODEL_PATH = "model/pipeline.pkl" 
try:
    with open(MODEL_PATH, "rb") as model_file:
        model = pickle.load(model_file)
except Exception as e:
    st.error(f"❌ Error loading model: {str(e)}")


# Configure Gemini API
client = openai.OpenAI(
    api_key="",
    base_url=""  # Important: Use GroqCloud's base URL
)

# Define model input columns
COLUMNS = [
    'property_type', 'sector', 'bedRoom', 'bathroom', 'balcony', 
    'agePossession', 'built_up_area', 'servant room', 'store room', 
    'furnishing_type', 'luxury_category', 'floor_category'
]

# Default values
DEFAULT_VALUES = {
    "property_type": "flat",
    "sector": "sector 36",
    "bedRoom": 2,
    "bathroom": 1,
    "balcony": '1',
    "agePossession": "New Property",
    "built_up_area": 1000,
    "servant room": 0,
    "store room": 0,
    "furnishing_type": "unfurnished",
    "luxury_category": "Low",
    "floor_category": "Low Floor"
}

def extract_details(user_input):
    """
    Extracts structured real estate details from user input using Gemini AI.
    Ensures missing values are replaced with defaults.
    """
    try:
        
        details = {}

        # Extract property type (check for "house")
        Response=extract_property_data(user_input)
        print(Response)
        if Response["Property Type"]:
            details["property_type"]=str(Response["Property Type"]).lower() 
        else:
            details["property_type"]=DEFAULT_VALUES['property_type']

        # Extract sector
        if Response["Sector"]:
            details["sector"] = "sector "+str(Response["Sector"])
        else:
            details["sector"]=DEFAULT_VALUES['sector']

        # Extract number of bedrooms (e.g., "5 BHK")
        if Response["Bedroom Count"]:
            details["bedRoom"]=Response["Bedroom Count"]
        else:
            details["bedRoom"]=DEFAULT_VALUES['bedRoom']

        # Extract number of bathrooms (e.g., "3 bathroom")
        if Response["Bathroom Count"]:
            details["bathroom"]=Response["Bathroom Count"]
        else:
            details["bathroom"]=DEFAULT_VALUES['bathroom']

        # Extract balcony count (e.g., "3 balcony")
        if Response["Balcony Count"]:
            if Response["Balcony Count"] > 3:
                details["balcony"]="3+"
            else:
                details["balcony"]=str(Response["Balcony Count"])
        else:
            details["balcony"]=DEFAULT_VALUES['balcony']

        # Extract furnishing type (e.g., "semifurnished")
        if Response["Furnishing Type"]:
            details["furnishing_type"]=Response["Furnishing Type"].lower()
        else:
            details["furnishing_type"]=DEFAULT_VALUES['furnishing_type']

        # Age
        if Response["Age/Possession"]:
            details["agePossession"]=Response["Age/Possession"]
        else:
            details["agePossession"]=DEFAULT_VALUES['agePossession']

        # Extract floor category (e.g., "high floor")
        if Response["Floor Category"]:
            details["floor_category"]=Response["Floor Category"]
        else:
            details["floor_category"]=DEFAULT_VALUES['floor_category']

        # servant room
        if Response["Servant Room"]:
            details["servant room"]=Response["Servant Room"]
        else:
            details["servant room"]=DEFAULT_VALUES['servant room']

        #Store Room
        if Response["Store Room"]:
            details["store room"]=Response["Store Room"]
        else:
            details["store room"]=DEFAULT_VALUES['store room']

        # Call Gemini API to extract built-up area
        if Response["Built-up Area"]:
            details["built_up_area"]=Response["Built-up Area"]
        else:
            details["built_up_area"]=DEFAULT_VALUES['built_up_area']

        # luxury Item
        if Response["Luxury Category"]:
            details["luxury_category"]=Response["Luxury Category"]
        else:
            details["luxury_category"]=DEFAULT_VALUES['luxury_category']

        # Default values handling for missing fields
        for key, value in DEFAULT_VALUES.items():
            if key not in details:
                details[key] = value  

        return details

    except Exception as e:
        st.write(f"❌ Unexpected error: {str(e)}")
        return DEFAULT_VALUES.copy()


def extract_property_data(description: str):
    """
    Extracts structured property data from a given real estate description.

    Args:
        description (str): The property listing description.

    Returns:
        dict: A dictionary with extracted property details.
    """
    
    prompt = f"""
    You are an expert in structured data extraction. Extract the following fields from the given real estate description. Ensure appropriate classification for categorical fields:

    1. **Property Type**: Extract and classify as 'Flat' or 'House'.
    2. **Sector**: Extract an integer if present, otherwise infer the locality.
    3. **Bedroom Count**: Extract an integer.
    4. **Bathroom Count**: Extract an integer.
    5. **Balcony Count**: Extract an integer.
    6. **Age/Possession**: Classify as:
       - 'New Property' (if the description mentions newly built, under construction, or ready to move in)
       - 'Relatively New' (if 0-5 years old)
       - 'Moderately Old' (if 6-15 years old)
       - 'Old Property' (if older than 15 years or described as old)
    7. **Built-up Area**: Extract in square feet (sqft).
    8. **Servant Room**: Return 1 if mentioned, else 0.
    9. **Store Room**: Return 1 if mentioned, else 0.
    10. **Furnishing Type**: Classify as:
       - 'unfurnished' (if explicitly stated or no furniture details are given)
       - 'semifurnished' (if some furniture, modular kitchen, or wardrobes are mentioned)
       - 'furnished' (if fully furnished or detailed furniture is included)
    11. **Luxury Category**: Classify as:
       - 'Low' (if basic or affordable)
       - 'Medium' (if mid-range or decent features)
       - 'High' (if premium, luxury, or upscale)
    12. **Floor Category**: Classify as:
       - 'Low Floor' (if floor number is 0-3)
       - 'Mid Floor' (if floor number is 4-10)
       - 'High Floor' (if floor number is above 10)

    Given the property description:
    "{description}"

    **Ensure that terms like 'newly built' or 'newly constructed' are mapped to 'New Property' and similar adjustments for other categorical fields. Return ONLY the extracted details in JSON format without additional text.**
    """

    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "user", "content": prompt}]
    )

    raw_response = response.choices[0].message.content.strip()

    try:
        # Extract JSON part using regex in case extra text is included
        match = re.search(r"\{.*\}", raw_response, re.DOTALL)
        if match:
            json_text = match.group(0)
            extracted_data = json.loads(json_text)  # Load as dictionary
            return extracted_data
        else:
            return {"error": "No JSON found in response", "raw_response": raw_response}
    
    except json.JSONDecodeError as e:
	    return {"error": f"Failed to parse JSON: {str(e)}", "raw_response": raw_response}



st.set_page_config(page_title="Real Estate Chatbot", layout="wide")
st.title("🏡 Real Estate Chatbot ")
st.write("🔹 Ask about property prices! Example: 'What’s the price of a 3 BHK in Sector 36?'")
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display chat history
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Type your query...")

def predict_price(details):
    """Predicts the price based on extracted details, with error handling."""
    try:
        input_df = pd.DataFrame([details])
        predicted_price = np.expm1(model.predict(input_df)[0])  # Model prediction
        return predicted_price
    except Exception as e:
        return None  # Indicate failure


if user_input:
    st.session_state["messages"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    details = extract_details(user_input)
    price = predict_price(details)  # Call the updated function

    if price is not None:
        low = price - 0.22
        high = price + 0.22
        bot_response = f"The estimated price for a {details['bedRoom']} BHK {details['property_type']} in {details['sector']} is between **₹{low:,.2f} Cr.** to **₹{high:,.2f} Cr.**."
    else:
        bot_response = "Oops! It looks like we don't have enough data for the given details. Try modifying your input and reattempt."

    st.session_state["messages"].append({"role": "assistant", "content": bot_response})
    with st.chat_message("assistant"):
        st.markdown(bot_response)


