import subprocess
import pandas as pd
import numpy as np
from models import train_models, vectorizer, le
from Levenshtein import distance as levenshtein_distance
from fuzzywuzzy import fuzz
from serealizeStateMachine import StateMachine

CONFIG_TTS = False



def infer_properties(restaurant_properties):
    # Check if properties meet the rules, return the consequences
    inferred_properties = {}
    
    # We apply rules in order
    # For ex. if restaurant is cheap, good and romanian, it being romanian will overwrite touristic status to False
    for antedecent, consequent, value in rules:
        if all(prop in restaurant_properties for prop in antedecent):
            inferred_properties[consequent] = value

    return inferred_properties

def fuzzy_keyword_match(keyword, text, threshold=80):
    
    words = text.lower().split()
    
    # Look for exact match first
    if keyword in words:
        return True

    for word in words:
        similarity = fuzz.ratio(keyword, word)
        if similarity > threshold:
            return True
    return False


class RestaurantRecommender:
    def __init__(self,state_machine):
        # Initial state
        self.state = 'Welcome'
        self.state_machine=state_machine
        # User preferences
        self.area = None
        self.food_type = None
        self.price_range = None
        self.special_features = None
        self.matched_restaurant=None
        self.isInformationGiven=False
        # Placeholder for restaurant data (You can replace this with real data later)
        self.restaurant_df = pd.read_csv('restaurant_info.csv')
        
        self.log_reg = train_models()

    def clear_state(self):
    # Reset preferences, when no matching restaurant is found
        self.area = None
        self.food_type = None
        self.price_range = None
        
    def extract_preferences(self, utterance):
                
        # Define list of keywords that we're looking to match
        keywords = {
            'food_type': ['british', 'modern european', 'italian', 'romanian', 'seafood',
                'chinese', 'steakhouse', 'asian oriental', 'french', 'portuguese',
                'indian', 'spanish', 'european', 'vietnamese', 'korean', 'thai',
                'moroccan', 'swiss', 'fusion', 'gastropub', 'tuscan',
                'international', 'traditional', 'mediterranean', 'polynesian',
                'african', 'turkish', 'bistro', 'north american', 'australasian',
                'persian', 'jamaican', 'lebanese', 'cuban', 'japanese', 'catalan'], 
            'area': ['west', 'north', 'south', 'centre', 'east'],
            'price_range': ['moderate', 'expensive', 'cheap'],
            'special_feature': ['touristic', 'assigned seats', 'children', 'romantic']
        }

        for key, options in keywords.items():

            # Check for exact match
            for option in options:
                if option in utterance:
                    setattr(self, key, option) 
                    print(f"{key} updated to {option}")
                    print(f" Area {self.area} Food {self.food_type} Price {self.price_range}")
                    break

            # If no exact match, check fuzzy match  
            if not hasattr(self, key):
                for option in options:
                    if fuzzy_keyword_match(option, utterance):
                        setattr(self, key, option)
                        print(f"{key} fuzzily updated to {option}")
                        break

       
            
    def predict_dialog_act(self, utterance):
        custom_message_vec = vectorizer.transform([utterance])

        prediction = self.log_reg.predict(custom_message_vec)
        prediction_1d = np.array([prediction]).ravel() # Change shape to pacify a warning from LabelEncoder
        prediction_label = le.inverse_transform(prediction_1d)
        print(f"PREDICTED: {prediction_label}")
        
        return prediction_label
    
    def find_restaurant(self):
        
        restaurant_match_df = self.restaurant_df.copy()
        no_match_for = [] 
        
        # Filter the Restaurants for an exact match to preferences
        restaurant_match_df = restaurant_match_df[restaurant_match_df['area'].str.lower() == self.area.lower()]
        restaurant_match_df = restaurant_match_df[restaurant_match_df['food'].str.lower() == self.food_type.lower()]
        restaurant_match_df = restaurant_match_df[restaurant_match_df['pricerange'].str.lower() == self.price_range.lower()]

        # Return the restaurant(s) with a note 
        if not restaurant_match_df.empty:
            return restaurant_match_df, "exact_match"

        # Find closest matches with info on what criteria is not met.
        closest_match_df = self.restaurant_df.copy()
        if self.area:
            closest_match_df = closest_match_df[closest_match_df['area'].str.lower() == self.area.lower()]
        if self.food_type:
            closest_match_df = closest_match_df[closest_match_df['food'].str.lower() == self.food_type.lower()]
        
        if not closest_match_df.empty:
            return closest_match_df, "price_range"
        
        closest_match_df = self.restaurant_df.copy()
        if self.area:
            closest_match_df = closest_match_df[closest_match_df['area'].str.lower() == self.area.lower()]
        if self.price_range:
            closest_match_df = closest_match_df[closest_match_df['pricerange'].str.lower() == self.price_range.lower()]
        
        if not closest_match_df.empty:
            return closest_match_df, "food_type"

        closest_match_df = self.restaurant_df.copy()
        if self.food_type:
            closest_match_df = closest_match_df[closest_match_df['food'].str.lower() == self.food_type.lower()]
        if self.price_range:
            closest_match_df = closest_match_df[closest_match_df['pricerange'].str.lower() == self.price_range.lower()]
        
        if not closest_match_df.empty:
            return closest_match_df, "area"
        
        return closest_match_df
    



    def giveInformation(self):
        if self.mismatch_reason == 'exact_match':
            output_utterance(f"Restaurant {self.matched_restaurant.restaurantname.values[0].title()} serves {self.matched_restaurant.food.values[0].title()} food in the {self.matched_restaurant.area.values[0].title()} of town, and has {self.matched_restaurant.pricerange.values[0].title()} prices")
        elif self.mismatch_reason == 'area':
            output_utterance(f"Sorry, we didn't find a matching restaurant in the {self.area} of town, but")
            output_utterance(f"Restaurant {self.matched_restaurant.restaurantname.values[0].title()} serves {self.matched_restaurant.food.values[0].title()} food, in the {self.matched_restaurant.area.values[0].title()}  of town and has {self.matched_restaurant.pricerange.values[0].title()} prices")
        elif self.mismatch_reason == 'food_type': 
            output_utterance(f"Sorry, we didn't find a restaurant serving {self.food_type} type of food, but")
            output_utterance(f"Restaurant {self.matched_restaurant.restaurantname.values[0].title()} serves {self.matched_restaurant.food.values[0].title()} food, in the {self.matched_restaurant.area.values[0].title()}  of town and has {self.matched_restaurant.pricerange.values[0].title()} prices")
        elif self.mismatch_reason == 'price_range':
            output_utterance(f"Sorry, we didn't find a matching restaurant in {self.price_range} price range, but")
            output_utterance(f"Restaurant {self.matched_restaurant.restaurantname.values[0].title()} serves {self.matched_restaurant.food.values[0].title()} in the {self.matched_restaurant.area.values[0].title()} part of town and prices are {self.matched_restaurant.pricerange.values[0].title()}")
        
    def classifyRequest(self,utterance):
        if "address"  in utterance:
            return "address"
        if "postcode"  in utterance:
            return "postcode"
        if "phone" in utterance: 
            return "phone"

    def make_transition(self):  
        transition=None
        stateUpdated=False
        currentState=self.state_machine.getState(self.state_machine.currentState)
        print(currentState.name)
        utterance=None
        if currentState.type=='Process': #these are the rectangular states, here the machine should say something
            if currentState.id==23:
                output_utterance(currentState.utterances[0])
                utterance = input(currentState.utterances[0]).lower()
                return True
            elif currentState.id ==14:
                if (not self.isInformationGiven):
                    self.giveInformation()
                    self.isInformationGiven=True
                output_utterance(currentState.utterances[0])
                utterance = input(currentState.utterances[0]).lower()
                transition = self.predict_dialog_act(utterance)
                if transition=='request':
                    requestType=self.classifyRequest(utterance)
                    if requestType=='address':
                        self.state_machine.currentState=17
                    elif requestType=='postcode':
                        self.state_machine.currentState=18  
                    elif requestType=='phone':
                        self.state_machine.currentState=16
                    return False
            elif currentState.id ==19:
                output_utterance(f"The phone number of restaurant {self.matched_restaurant.restaurantname} is {self.matched_restaurant.phone}")
            elif currentState.id ==20:
                output_utterance(f"The address of restaurant {self.matched_restaurant.restaurantname} is {self.matched_restaurant.addr}")
            elif currentState.id ==21:
                output_utterance(f"The postcode of restaurant {self.matched_restaurant.restaurantname} is {self.matched_restaurant.postcode}")
            elif currentState.id == 10:
                utterance = input(currentState.utterances[0])
                self.extract_preferences(utterance)
                print(self.special_features)
            else:
                utterance = input(currentState.utterances[0]).lower()
                output_utterance(currentState.utterances[0])
                dialog_act = self.predict_dialog_act(utterance)
                self.extract_preferences(utterance)
        elif currentState.type=='Decision': #these are the romboid states, here the machine should check something
            if currentState.id == 2:
                transition='Yes' if self.area is not None  else 'No'
            elif currentState.id == 4:
                transition='Yes' if self.food_type is not None  else 'No'
            elif currentState.id == 6:
                transition='Yes' if self.price_range is not None  else 'No'
            elif currentState.id == 11:
                transition='Yes' if self.special_features is not None else 'Yes'
            elif currentState.id == 18:
                transition='Yes' if self.matched_restaurant.postcode is not None  else 'No'
            elif currentState.id == 16:
                transition='Yes' if self.matched_restaurant.phone is not None  else 'No'
            elif currentState.id == 17:
                transition='Yes' if self.matched_restaurant.addr is not None  else 'No'
            elif currentState.id == 8:
                self.matching_restaurants, reason = self.find_restaurant()
                if not self.matching_restaurants.empty:
                    transition='Yes' 
                    self.matched_restaurant=self.matching_restaurants.head(1)
                    self.mismatch_reason = reason
                else:
                    transition='No'
            elif currentState.id == 12:
                if not self.matching_restaurants.empty:
                    for restaurant in self.matching_restaurants.iterrows():
                        inferred_properties = infer_properties({restaurant})
                        print(inferred_properties)
                
            
        for t in currentState.transitions: 
        # here we check for any conditional transitions and move on accordingly 
            if t[2]==transition:
                    self.state_machine.currentState=t[1]                    
                    stateUpdated=True
                    break
        if not stateUpdated:
            for t in currentState.transitions: 
            # if there are no conditional transitions or the conditions for these transitions have not been met we move towards an unconditional transition 
                if t[2] =='' :
                    self.state_machine.currentState=t[1]
                    stateUpdated=True
                    break
        return False
if __name__ == "__main__":

    s=StateMachine()
    s.serealize()
    # Create an instance of the RestaurantRecommender class
    recommender = RestaurantRecommender(s)
   
    # Start the state machine
    
    conversation_over=False
    while not conversation_over:
        conversation_over=recommender.make_transition()
        
