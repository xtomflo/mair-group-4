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
    
    def transition(self):
        while True:
            if self.state == 'Welcome':
                utterance = input("Welcome! I can help you find a restaurant. Have you already decided on the area? \n >").lower()
                print(utterance)
                dialog_act = self.predict_dialog_act(utterance)
                self.extract_preferences(utterance)
                self.state = 'Area expressed?'

            elif self.state == 'Area expressed?':
                # Placeholder for CHECK logic
                
                if self.area is not None:
                    self.state = 'Food type expressed?'
                else:
                    self.state = 'Ask for area'

            elif self.state == 'Ask for area':
                # Placeholder for TALK logic 
                utterance = input("What is your preferred area? \n >").lower()
                dialog_act = self.predict_dialog_act(utterance)
                self.extract_preferences(utterance)
                self.state = 'Area expressed?'

            elif self.state == 'Food type expressed?':
                # Placeholder for CHECK logic
                if self.food_type is not None:
                    self.state = 'Price range expressed?'
                else:
                    self.state = 'Ask for food type'

            elif self.state == 'Ask for food type':
                # Placeholder for TALK logic 
                utterance = input("What type of cuisine would you like to eat? \n >").lower()
                dialog_act = self.predict_dialog_act(utterance)
                self.extract_preferences(utterance)

                self.state = 'Food type expressed?'

            elif self.state == 'Price range expressed?':
                # Placeholder for CHECK logic
                if self.price_range is not None:
                    self.state = 'Check if restaurant exists'
                else:
                    self.state = 'Ask for price range'

            elif self.state == 'Ask for price range':
                # Placeholder for TALK logic 
                utterance = input("What would be a fitting price range? \n >").lower()
                self.extract_preferences(utterance)

                self.state = 'Price range expressed?'

            elif self.state == 'Check if restaurant exists':
                # Placeholder for CHECK logic
                # Here, you would typically check if a restaurant exists that fits the criteria
                matching_restaurants, reason = self.find_restaurant()  # Placeholder, replace with actual filtering logic
                if not matching_restaurants.empty:
                    self.state = 'Give recommendation'
                else:
                    self.state = 'Inform no matching restaurant available. Ask to state new preference'

            elif self.state == 'Inform no matching restaurant available. Ask to state new preference':
                print("Sorry, no matching restaurant is available. Please state new preferences.")
                self.clear_state()
                self.state = 'Area expressed?'

            elif self.state == 'Give recommendation':
                print(f"We've found a matching restaurant. It's name is {matching_restaurants.restaurantname}")
                print(f"It's located on {matching_restaurants.addr}. \n You can reach them by calling {matching_restaurants.phone}")
                # Placeholder for TALK logic (user interaction)
                # Here, you would typically show the recommended restaurant from the dataframe

                self.state == 'Wait'
                
            elif self.state == 'Wait':
                utterance = input("Can I help with more information?")
                dialog_act = self.predict_dialog_act(utterance)
                
                if dialog_act == 'request':
                    self.state = 'Provide Info'
                    
              #  elif dialog_act == ''
                    
            elif self.state == 'Provide Info':
                print(f"It's located on {matching_restaurants.addr}. \n You can reach them by calling {matching_restaurants.phone}")
            
            elif self.state == 'Exit':
                print("Was great talking to you!")
                break


    def giveInformation(self):
        if self.area is not None and self.price_range is not None and self.food_type is not None:
            print(f"Restaurant {self.matched_restaurant.restaurantname} serves {self.matched_restaurant.food} in the {self.matched_restaurant.area} part of town, in a {self.matched_restaurant.pricerange}")
        elif self.area is None:
            print(f"Restaurant {self.matched_restaurant.restaurantname} serves {self.matched_restaurant.food}  in a {self.matched_restaurant.pricerange}")
        elif self.food_type is None:         
            print(f"Restaurant {self.matched_restaurant.restaurantname} in the {self.matched_restaurant.area} part of town, in a {self.matched_restaurant.pricerange}")
        elif self.price_range is None:
            print(f"Restaurant {self.matched_restaurant.restaurantname} serves {self.matched_restaurant.food} in the {self.matched_restaurant.area} part of town")

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
                utterance = input(currentState.utterances[0]).lower()
                return True
            elif currentState.id ==14:
                if (not self.isInformationGiven):
                    self.giveInformation()
                    self.isInformationGiven=True
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
                    print(f"The phone number of restaurant {self.matched_restaurant.restaurantname} is {self.matched_restaurant.phone}")
            elif currentState.id ==20:
                    print(f"The address of restaurant {self.matched_restaurant.restaurantname} is {self.matched_restaurant.addr}")
            elif currentState.id ==21:
                    print(f"The postcode of restaurant {self.matched_restaurant.restaurantname} is {self.matched_restaurant.postcode}")
            elif currentState.id ==10:
                    self.state_machine.currentState=14
                    return False
            else:
                utterance = input(currentState.utterances[0]).lower()
                transition = self.predict_dialog_act(utterance)
                self.extract_preferences(utterance)
        elif currentState.type=='Decision': #these are the romboid states, here the machine should check something
            if currentState.id==2:
                transition='Yes' if self.area is not None  else 'No'
            elif currentState.id==4:
                transition='Yes' if self.food_type is not None  else 'No'
            elif currentState.id==6:
                transition='Yes' if self.price_range is not None  else 'No'
            elif currentState.id==18:
                transition='Yes' if self.matched_restaurant.postcode is not None  else 'No'
            elif currentState.id==16:
                transition='Yes' if self.matched_restaurant.phone is not None  else 'No'
            elif currentState.id==17:
                transition='Yes' if self.matched_restaurant.addr is not None  else 'No'
            elif currentState.id==8:
                matching_restaurants, reason = self.find_restaurant()
                if not matching_restaurants.empty:
                    transition='Yes' 
                    self.matched_restaurant=matching_restaurants.head(1)
                else:
                    transition='No'
            
        for t in currentState.transitions:
            if t[2]==transition:
                    self.state_machine.currentState=t[1]                    
                    stateUpdated=True
                    break
        if not stateUpdated:
                for t in currentState.transitions:
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
   
    #recommender.transition()
    # Start the state machine
    
    conversation_over=False
    while not conversation_over:
        conversation_over=recommender.make_transition()
        
