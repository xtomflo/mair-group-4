
# Find the best matching restaurant

# Function to Manage State Transitions.
def state_transitions(current_state, user_utterance):
- current dialog state. 
- current user utterance




    return dialog_state, system_utterance



# Infer Dialogue Act
call predict on log_reg. 


# Get Keyword(s) from Sentence
# Fill the Slots - Area, FoodType, PriceRange 
from fuzzywuzzy import fuzz

def fuzzy_keyword_match(keyword, text, threshold=80):
    
    words = text.lower().split()
    
    for word in words:
        similarity = fuzz.ratio(keyword, word)
        if similarity > threshold:
            return True
    return False

    
    
    
def extract_preferences(utterance):
    
    preferences = {
        "location" : None,
        "food" : None,
        "price_range" : None
    }
    
    # Define list of keywords that we're looking to match
    food = ['british', 'modern european', 'italian', 'romanian', 'seafood',
       'chinese', 'steakhouse', 'asian oriental', 'french', 'portuguese',
       'indian', 'spanish', 'european', 'vietnamese', 'korean', 'thai',
       'moroccan', 'swiss', 'fusion', 'gastropub', 'tuscan',
       'international', 'traditional', 'mediterranean', 'polynesian',
       'african', 'turkish', 'bistro', 'north american', 'australasian',
       'persian', 'jamaican', 'lebanese', 'cuban', 'japanese', 'catalan']
    location = ['west', 'north', 'south', 'centre', 'east']
    price_range = ['moderate', 'expensive', 'cheap']
    
    
    # Check for food types
    for food in food_types:
        if fuzzy_keyword_match(food, utterance):
            preferences['food_type'] = food
            break

    # Check for locations
    for location in locations:
        if fuzzy_keyword_match(location, utterance):
            preferences['location'] = location
            break

    # Check for price range
    for price in price_range:
        if fuzzy_keyword_match(price, utterance):
            preferences['price_range'] = price
            break

    return preferences
