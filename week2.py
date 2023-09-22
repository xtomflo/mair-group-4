import pandas as pd
import numpy as np
from main import train_models, vectorizer, le
from Levenshtein import distance as levenshtein_distance
from fuzzywuzzy import fuzz
from serealizeStateMachine import StateMachine

def fuzzy_keyword_match(keyword, text, threshold=80):
    
    words = text.lower().split()
    
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
        self.matched_restaurant=None
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
        food_types = ['british', 'modern european', 'italian', 'romanian', 'seafood',
        'chinese', 'steakhouse', 'asian oriental', 'french', 'portuguese',
        'indian', 'spanish', 'european', 'vietnamese', 'korean', 'thai',
        'moroccan', 'swiss', 'fusion', 'gastropub', 'tuscan',
        'international', 'traditional', 'mediterranean', 'polynesian',
        'african', 'turkish', 'bistro', 'north american', 'australasian',
        'persian', 'jamaican', 'lebanese', 'cuban', 'japanese', 'catalan']
        areas = ['west', 'north', 'south', 'centre', 'east']
        price_range = ['moderate', 'expensive', 'cheap']
        
        
        # Check for food types
        for food in food_types:
            if fuzzy_keyword_match(food, utterance):
                self.food_type = food
                print(f"Food updated {food}!")
                break

        # Check for area
        for area in areas:
            if fuzzy_keyword_match(area, utterance):
                self.area = area
                print(f"Area updated {area}!")
                break

        # Check for price range
        for price in price_range:
            if fuzzy_keyword_match(price, utterance):
                self.price_range = price
                print(f"Price_Range updated {price}!")
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
        if self.area is not None and self.price_range is not None and self.food_type is not None:
            print(f"Restaurant {self.matched_restaurant.restaurantname.values[0]} serves {self.matched_restaurant.food.values[0]} food in the {self.matched_restaurant.area.values[0]} part of town, in a {self.matched_restaurant.pricerange.values[0]}")
        elif self.area is None:
            print(f"Restaurant {self.matched_restaurant.restaurantname.values[0]} serves {self.matched_restaurant.food.values[0]}  in a {self.matched_restaurant.pricerange.values[0]}")
        elif self.food_type is None:         
            print(f"Restaurant {self.matched_restaurant.restaurantname.values[0]} in the {self.matched_restaurant.area.values[0]} part of town, in a {self.matched_restaurant.pricerange.values[0]}")
        elif self.price_range is None:
            print(f"Restaurant {self.matched_restaurant.restaurantname.values[0]} serves {self.matched_restaurant.food.values[0]} in the {self.matched_restaurant.area.values[0]} part of town")

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
       # print(currentState.name)
        utterance=None
        if currentState.type=='Process': #these are the rectangular states, here the machine should say something
            if currentState.id==14: # state 14 is the exit state so here we return false in order to get out of the loop 
                return False
            elif currentState.id ==11: # state 11 is the state that gives the recommendation based ot a picked restaurant 
                self.giveInformation()
            elif currentState.id ==16: # state 16 is the state where we are supposed to give information about the phone number of the recommended restaurant 
                    print(f"The phone number of restaurant {self.matched_restaurant.restaurantname.values[0]} is {self.matched_restaurant.phone.values[0]}")
            elif currentState.id ==23: # state 23 is the state where we are supposed to give information about the address of the recommended restaurant 
                    print(f"The address of restaurant {self.matched_restaurant.restaurantname.values[0]} is {self.matched_restaurant.addr.values[0]}")
            elif currentState.id ==24:# state 24 is the state where we are supposed to give information about the postcode of the recommended restaurant 
                    print(f"The postcode of restaurant {self.matched_restaurant.restaurantname.values[0]} is {self.matched_restaurant.postcode.values[0]}")
            else:
                utterance = input(currentState.utterances[0]).lower()
                transition = self.predict_dialog_act(utterance)
                if currentState.id==12 and transition=='request': # state 12 is the state where we we wait for new request from the user after giving him a recommendation if his next uttterance is classify as a request we need to go to a new state according to how we classify this request 
                    requestType=self.classifyRequest(utterance)
                    if requestType=='address':
                        self.state_machine.currentState=13
                    if requestType=='postcode':
                        self.state_machine.currentState=21  
                    if requestType=='phone':
                        self.state_machine.currentState=22
                    return True
                self.extract_preferences(utterance)
        elif currentState.type=='Decision': #these are the romboid states, here the machine should check something
            if currentState.id==4: # decision 4 is where we see if we already have information about the area
                transition='Yes' if self.area is not None  else 'No'
            elif currentState.id==6: # decision 4 is where we see if we already have information about the food type 
                transition='Yes' if self.food_type is not None  else 'No'
            elif currentState.id==8: # decision 4 is where we see if we already have information about the price range 
                transition='Yes' if self.price_range is not None  else 'No'
            elif currentState.id==21: # decision 21 Iis where we see if we  have information about the postocde of the recommended restaurant
                transition='Yes' if self.matched_restaurant.postcode is not None  else 'No'
            elif currentState.id==22: # decision 22 is where we see if we  have information about the  phone number  the recommended restaurant
                transition='Yes' if self.matched_restaurant.phone is not None  else 'No'
            elif currentState.id==13: # decision 13 is where we see if we  have information about the address of the recommended restaurant
                transition='Yes' if self.matched_restaurant.addr is not None  else 'No'
            elif currentState.id==10: # state 10 assumes that we have all the needed information to give a recommendation and calls a function that selects an appropriate one 
                matching_restaurants, reason = self.find_restaurant()
                if not matching_restaurants.empty:
                    transition='Yes' 
                    self.matched_restaurant=matching_restaurants.head(1)
                else:
                    transition='No'
            
        for t in currentState.transitions: # here we check for any conditional transitions and move on accordingly 
            if t[2]==transition:
                    self.state_machine.currentState=t[1]                    
                    stateUpdated=True
                    break
        if not stateUpdated:
                for t in currentState.transitions: # if there are no conditional transitions or the conditions for these transitions have not been met we move towards an unconditional transition 
                    if t[2] =='' :
                        self.state_machine.currentState=t[1]
                        stateUpdated=True
                        break
        return True
if __name__ == "__main__":

    s=StateMachine()
    s.serealize()
    # Create an instance of the RestaurantRecommender class
    recommender = RestaurantRecommender(s)
   
    # Start the state machine
    
    conversation_over=True
    while conversation_over:
        conversation_over=recommender.make_transition()
        
