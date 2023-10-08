import subprocess
import threading
import time
import string

from enum import Enum, auto
import pandas as pd
import numpy as np
import utils

from models import le, vectorizer
import models

# Dictionary to store user preferences
preferences = {
    'area': None,
    'food_type': None
}

SETTINGS = {
   'tts': False,
   'closest_match': False,
   'model': 'LOG_REG',
   'use_special_features': False
}

def collect_config():
    print("Choose configuration options:")

    # Text-to-speech
    tts = input("Enable text-to-speech? (y/n)\n")
    if tts.lower() == 'y':
        SETTINGS['tts'] = True

    # Closest Match   
    stt = input("Enable Closest Match? (y/n)\n")
    if stt.lower() == 'y':
        SETTINGS['closest_match'] = True
    
    # Choice of model
    model = input("Choose model (A - LOG_REG/B - KNN)\n")
    if model.lower() == 'a':
        SETTINGS['model'] = 'LOG_REG'
    elif model.lower() == 'b':
        SETTINGS['model'] = 'KNN'

    use_special_features = input("Do you want to be able to add special features? (y/n)")
    if use_special_features.lower() == 'y':
        SETTINGS['use_special_features'] = True
    
 

    # Print summary
    print("\nSelected configuration:")
    print(f"- TTS: {SETTINGS['tts']}")   
    print(f"- Closest Match: {SETTINGS['closest_match']}")
    print(f"- Model: {SETTINGS['model']}")
    print(f"- Special features: {SETTINGS['use_special_features']}")


class State(Enum):

    # "Process" States
    WELCOME = auto()
    ASK_AREA = auto()
    ASK_FOOD = auto()
    ASK_PRICE = auto()
    ASK_REQUIREMENTS = auto()
    PROVIDE_RECOMMENDATION = auto()
    NO_RESTAURANT = auto()
    CLASSIFY_REQUEST = auto()
    PROVIDE_PHONE = auto()
    PROVIDE_ADDRESS = auto()
    PROVIDE_POSTCODE = auto()
    APOLOGIZE = auto()
    EXIT = auto()
    
    
    CHECK_AREA = auto()
    CHECK_FOOD = auto() 
    CHECK_PRICE = auto()
    CHECK_RESTAURANT = auto()
    CHECK_RESTAURANT_NO2 = auto()
    VALID_PREFERENCES = auto()
    CHECK_ADDRESS = auto()
    CHECK_POSTCODE = auto()
    CHECK_PHONE = auto()
    FILTER_RESTAURANT = auto()

class RestaurantRecommender():
    
    def __init__(self):
    ### Initialization of the class instance
        self.current_state = State.WELCOME
        # User preferences
        self.area = None
        self.food_type = None
        self.price_range = None
        self.special_feature = None
        self.matching_restaurants = None
        self.current_index = 0  # To track the current restaurant being recommended
        self.info_provided = False
        
        if SETTINGS.get('model') == 'LOG_REG':
            self.log_reg = models.train_log_reg()
        elif SETTINGS.get('model') == 'KNN':
            self.knn = models.train_knn()
        
        self.restaurant_df = pd.read_csv('restaurant_info.csv')
    
    def clear_state(self):
    ### When no match is found, state is clear to start the process of collecting preferences again
        self.area = None
        self.food_type = None
        self.price_range = None
    
    def output_utterance(self, system_utterance):
    ### Generic output to handle Text-To-Speech functionality
        if SETTINGS.get('tts') is True:
            print(system_utterance)
            utils.speak(system_utterance)
        else:
            print(system_utterance)
            
    def collect_input(self, system_utterance):
    ### Collecting user input and extracting the dialog_act
        if SETTINGS.get('tts') is True:
            print(system_utterance)
            utils.speak(system_utterance)
            user_utterance = input(">")
        else:
            user_utterance = input(system_utterance + "\n")
        
        dialog_act = self.predict_dialog_act(user_utterance)

        return dialog_act, user_utterance.lower()

    def predict_dialog_act(self, utterance):
    ### Predict a dialog act for a given utterance based on a model that's been configured to use 
        custom_message_vec = vectorizer.transform([utterance])
        if SETTINGS.get('model')== 'LOG_REG':
            prediction = self.log_reg.predict(custom_message_vec)
        elif SETTINGS.get('model') == 'KNN':
            prediction = self.knn.predict(custom_message_vec)
            
        prediction_1d = np.array([prediction]).ravel() # Change shape to pacify a warning from LabelEncoder
        prediction_label = le.inverse_transform(prediction_1d)
        print(f"PREDICTED: {prediction_label}")
        
        return prediction_label
    
    
    def find_restaurant(self):
    ### Find a restaurant that will match stated preference criteria.

        restaurant_match_df = self.restaurant_df.copy()
        
        # Handle 'any' inputs
        if self.area != 'any':
            restaurant_match_df = restaurant_match_df[restaurant_match_df['area'].str.lower() == self.area.lower()]
        if self.food_type  != 'any':
            restaurant_match_df = restaurant_match_df[restaurant_match_df['food'].str.lower() == self.food_type.lower()]
        if self.price_range  != 'any':
            restaurant_match_df = restaurant_match_df[restaurant_match_df['pricerange'].str.lower() == self.price_range.lower()]
        
        # Return exact match or closest match
        if not restaurant_match_df.empty:
            return restaurant_match_df, "exact_match"
        
        # Check if closest match finding enabled
        if SETTINGS.get('closest_match') is False:
            return restaurant_match_df, "empty"

        # Find closest matches
        closest_match_df = self.restaurant_df.copy() 
        if self.area != 'any':
            closest_match_df = closest_match_df[closest_match_df['area'].str.lower() == self.area.lower()]
        if self.food_type != 'any':
            closest_match_df = closest_match_df[closest_match_df['food'].str.lower() == self.food_type.lower()]
        if not closest_match_df.empty:
            return closest_match_df, "price_range"
        if self.price_range != 'any':
            closest_match_df = closest_match_df[closest_match_df['pricerange'].str.lower() == self.price_range.lower()]
        if self.food_type != 'any':
            closest_match_df = closest_match_df[closest_match_df['food'].str.lower() == self.food_type.lower()]
        if not closest_match_df.empty:
            return closest_match_df, "area" 
        if self.area != 'any':
            closest_match_df = closest_match_df[closest_match_df['area'].str.lower() == self.area.lower()]
        if self.price_range != 'any':
            closest_match_df = closest_match_df[closest_match_df['pricerange'].str.lower() == self.food_type.lower()]
        if not closest_match_df.empty:
            return closest_match_df, "area"

        return closest_match_df, "empty"


    def filter_restaurants(self, restaurants):
    ### Filter restaurants for ones matching special feature requests, like touristic, romantic etc. 
    
        for r in restaurants.itertuples():
            inferred = utils.infer_properties([r.pricerange, r.food_quality, r.crowdedness, r.length_of_stay])
            if inferred.get(self.special_feature) is True:
                return restaurants[restaurants.restaurantname == r.restaurantname]


    def give_recommendation(self):
    ### Output correct Restaurant Recommendation
        # Save the currently recommended restaurant
        self.current_recommendation = self.matching_restaurants.iloc[0]

        if self.mismatch_reason == 'exact_match':
            self.output_utterance(f"""Restaurant {string.capwords(self.current_recommendation.restaurantname) } serves {string.capwords(self.current_recommendation.food) } food in the {string.capwords(self.current_recommendation.area) } of town, and has {string.capwords(self.current_recommendation.pricerange) } prices""")
        if SETTINGS.get('closest_match') is True:
            if self.mismatch_reason == 'area':
                self.output_utterance(f"Sorry, we didn't find a matching restaurant in the {string.capwords(self.area) } of town, but")
                self.output_utterance(f""""Restaurant {string.capwords(self.current_recommendation.restaurantname) } serves {string.capwords(self.current_recommendation.food) } food, in the {string.capwords(self.current_recommendation.area) }  of town and has {string.capwords(self.current_recommendation.pricerange) } prices""")
            elif self.mismatch_reason == 'food_type': 
                self.output_utterance(f"Sorry, we didn't find a restaurant serving {string.capwords(self.food_type) } type of food, but")
                self.output_utterance(f""""Restaurant {string.capwords(self.current_recommendation.restaurantname) } serves {string.capwords(self.current_recommendation.food) } food, in the {string.capwords(self.current_recommendation.area) } of town and has {string.capwords(self.current_recommendation.pricerange) } prices""")
            elif self.mismatch_reason == 'price_range':
                self.output_utterance(f"Sorry, we didn't find a matching restaurant in {string.capwords(self.price_range) } price range, but")
                self.output_utterance(f""""Restaurant {string.capwords(self.current_recommendation.restaurantname) } serves {string.capwords(self.current_recommendation.food) } in the {string.capwords(self.current_recommendation.area) } part of town and prices are {string.capwords(self.current_recommendation.pricerange) }""")
        
        if self.special_feature is not None:
            reasoning=utils.getReasoningInWords(self.special_feature)
            res="Restaurant "+str(self.current_recommendation.restaurantname) +" is also "+ str(self.special_feature)+ ", because "
            res+=reasoning
            self.output_utterance(res)
        # Remove the recommendation from the list of remaining ones
        self.matching_restaurants = self.matching_restaurants.iloc[1:]
        self.info_provided = True
        
        
    def extract_preferences(self, utterance):
    ### Extract preferences from the utterance of the user by means of fuzzy keyword matching
        print(f"Extracting Preference {utterance}")
        
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
            'special_feature': ['touristic', 'assigned seats', 'children', 'romantic'],
        }
        
        # Workaround for handling test-cases with 'world' cuisine
        if 'world' in utterance:
            utterance = utterance.replace("world", "international")
        if 'asian' in utterance:
            utterance = utterance.replace("asian", "asian oriental")
                        
        preferences=utils.pattern_matching(utterance)
        for key, options in keywords.items():
            # Check for exact match
            for option in options:
                if option in utterance:
                    if key not in preferences:
                        preferences[key]=option
                    break
                else: 
                    continue
            if not bool (preferences):
                # If no exact match, check fuzzy match  
                #print("fuzzy check")
                for option in options:
                    if utils.fuzzy_keyword_match(option, utterance):
                        if key is not preferences:
                            preferences[key]=option
                            #print(f"{key} fuzzily updated to {option}")
                        break
        if bool (preferences):
            for key, option in preferences.items(): 
                setattr(self, key, option)
                #print(f"{key} updated to {option}")
                    

    def get_next_state(self, current_state, dialog_act:str = "none", user_utterance:str = ""):
        
        #print(f"Current State: {current_state}, Dialog Act: {dialog_act}")
        if dialog_act == 'bye':
            return State.EXIT
        
        if current_state == State.WELCOME:
            if dialog_act == 'hello':
                return State.WELCOME
            elif dialog_act == 'inform':
                return State.ASK_AREA

        elif current_state == State.ASK_AREA:
            if dialog_act == 'inform':
                self.extract_preferences(user_utterance)
                return State.CHECK_AREA
            else:
                return State.ASK_AREA

        elif current_state == State.CHECK_AREA:
            if self.area is not None:
                return State.ASK_FOOD
            else:
                return State.ASK_AREA
            
        elif current_state == State.ASK_FOOD:
            if dialog_act == 'inform':
                self.extract_preferences(user_utterance)
                return State.CHECK_FOOD
            else:  
                return State.ASK_FOOD

        elif current_state == State.CHECK_FOOD:
            if self.food_type is not None:
                return State.ASK_PRICE
            else:
                return State.ASK_FOOD

        elif current_state == State.ASK_PRICE:
            if dialog_act == 'inform':
                self.extract_preferences(user_utterance)
                return State.CHECK_PRICE
            else:
                return State.ASK_PRICE

        elif current_state == State.CHECK_PRICE:
            if self.price_range is not None:
                return State.CHECK_RESTAURANT
            else:
                return State.ASK_PRICE
            
        elif current_state == State.CHECK_RESTAURANT:
            self.matching_restaurants, self.mismatch_reason = self.find_restaurant()
            
            if self.matching_restaurants.empty is True:
                return State.NO_RESTAURANT
            else:
                if SETTINGS['use_special_features']:
                    return State.ASK_REQUIREMENTS
                else:
                    return State.PROVIDE_RECOMMENDATION
        elif current_state == State.NO_RESTAURANT:
            self.clear_state() 
            
            return State.ASK_AREA
        
        elif current_state == State.ASK_REQUIREMENTS:
            if dialog_act == 'inform':
                self.extract_preferences(user_utterance)
                if self.special_feature is not None:
                    return State.FILTER_RESTAURANT
            elif dialog_act == 'negate':
                return State.PROVIDE_RECOMMENDATION
            else:  
                return State.ASK_REQUIREMENTS

        elif current_state == State.FILTER_RESTAURANT:
            self.matching_restaurants = self.filter_restaurants(self.matching_restaurants)
            return State.CHECK_RESTAURANT_NO2
        
        elif current_state == State.CHECK_RESTAURANT_NO2:
            if self.matching_restaurants is not None:
                return State.PROVIDE_RECOMMENDATION
            else:
                return State.NO_RESTAURANT
                
        elif current_state == State.PROVIDE_RECOMMENDATION:
            if dialog_act == 'request':
                return State.CLASSIFY_REQUEST
            elif dialog_act == 'reqalts':
                return State.PROVIDE_RECOMMENDATION
            elif dialog_act in ['bye', 'thankyou', 'negate']:
                return State.EXIT
            else:
                return State.PROVIDE_RECOMMENDATION 
            
        elif current_state == State.CLASSIFY_REQUEST:
            request_type = utils.classify_request(user_utterance)
            if request_type == 'phone':
                return State.CHECK_PHONE
            elif request_type == 'address':
                return State.CHECK_ADDRESS 
            else:
                return State.CHECK_POSTCODE

        elif current_state == State.CHECK_PHONE:
            # Check if phone available
            if self.matching_restaurants.phone is not None:
                return State.PROVIDE_PHONE
            else:
                return State.APOLOGIZE

        elif current_state == State.PROVIDE_PHONE:
            return State.PROVIDE_RECOMMENDATION

        elif current_state == State.CHECK_ADDRESS:
            # Check if address available
            if self.matching_restaurants.addr is not None:
                return State.PROVIDE_ADDRESS
            else:
                return State.APOLOGIZE

        elif current_state == State.PROVIDE_ADDRESS:
            return State.PROVIDE_RECOMMENDATION

        elif current_state == State.CHECK_POSTCODE:
            # Check if postcode available
            if self.matching_restaurants.postcode is not None:
                return State.PROVIDE_POSTCODE
            else:
                return State.APOLOGIZE

        elif current_state == State.PROVIDE_POSTCODE:
            return State.PROVIDE_RECOMMENDATION

        elif current_state == State.APOLOGIZE:
            return State.PROVIDE_RECOMMENDATION

        elif current_state == State.EXIT:
            return State.EXIT
        
        return current_state

        
    def run(self):
        
        last_state = None
        current_state = State.WELCOME
        
        dialog_act, user_utterance = self.collect_input("Welcome to UU restaurant system. We can provide recommendations based on area, food type and price range. How may I help you?")

        while current_state != State.EXIT:
            
            #user_utterance = get_user_input()         
            next_state = self.get_next_state(current_state, dialog_act, user_utterance)
            
            if next_state == State.WELCOME:
                if last_state is State.WELCOME:
                    dialog_act, user_utterance = self.collect_input("Great to have you here! How can I help?")
                else:
                    dialog_act, user_utterance = self.collect_input("Hello! How may I help you?")

            elif next_state == State.ASK_AREA and self.area is None:
                
                # Ask for area preference
                if last_state is State.ASK_AREA:
                    dialog_act, user_utterance = self.collect_input("What area would you like to eat in? Choose from centre, east, west, south and north.")
                else:
                    dialog_act, user_utterance = self.collect_input("In which area are you looking for a restaurant?")
            
            elif next_state == State.ASK_FOOD and self.food_type is None:
                # Ask for food preference
                if last_state is State.ASK_FOOD:
                    dialog_act, user_utterance = self.collect_input("We don't seem to have restaurants of that type, try a different one?")
                else:
                    dialog_act, user_utterance = self.collect_input("What type of food do you prefer?")
                
            elif next_state == State.ASK_PRICE and self.price_range is None:
                # Ask for price range
                if last_state is State.ASK_PRICE:
                    dialog_act, user_utterance = self.collect_input("What prices do you prefer? You can choose from cheap, moderate and expensive.")
                else:
                    dialog_act, user_utterance = self.collect_input("What price range do you prefer?")
                
            elif next_state == State.ASK_REQUIREMENTS:
                # Ask for any other preferences
                if last_state is State.ASK_REQUIREMENTS:
                    dialog_act, user_utterance = self.collect_input("Do you have special wishes? You can choose from touristic, romantic, assigned seats or children friendly.")
                else:
                    dialog_act, user_utterance = self.collect_input("Do you have any additional requirements for the restaurant?")
                
            elif next_state == State.PROVIDE_RECOMMENDATION:
                # Provide restaurant recommendation
                if dialog_act == 'reqalts':
                    if self.matching_restaurants.empty is True:
                        self.output_utterance("Unfortunately there are no other restaurants matching your criteria.")
                        dialog_act, user_utterance = self.collect_input("Can I help you with anything else?")
                    else:
                        self.give_recommendation()
                elif self.info_provided is True:
                    dialog_act, user_utterance = self.collect_input("Can I help you with anything else?")
                else:
                    self.give_recommendation()
                    dialog_act, user_utterance = self.collect_input("Can I help you with anything else?")
                    
            elif next_state == State.NO_RESTAURANT:
                # Inform there's no matches
                self.output_utterance(f"Unfortunately we do dont have a {string.capwords(self.food_type)} restaurant in the {string.capwords(self.area)} area in {string.capwords(self.price_range)} price range.")
                self.output_utterance("Please try searching with differrent criteria")
                
            elif next_state == State.PROVIDE_PHONE:
                # Provide restaurant phone number
                self.output_utterance(f"The phone number of restaurant {string.capwords(self.current_recommendation.restaurantname) } is {self.current_recommendation.phone}")

            elif next_state == State.PROVIDE_ADDRESS:
                # Provide restaurant address
                self.output_utterance(f"The address of restaurant {string.capwords(self.current_recommendation.restaurantname) } is {self.current_recommendation.addr}")

            elif next_state == State.PROVIDE_POSTCODE:
                # Provide restaurant postcode
                self.output_utterance(f"The postcode of restaurant {string.capwords(self.current_recommendation.restaurantname) } is {self.current_recommendation.postcode}")

            elif next_state == State.APOLOGIZE:
                # Apologize that info is not available
                self.output_utterance("Unfortunately this information is not available, I apologise for the inconvenience.")
                
            elif next_state == State.EXIT:
                # Say goodbye
                self.output_utterance("Hope you enjoyed using our system, see you next time!")
                exit
                
            
            #dialog_act = self.predict_dialog_act(user_utterance)
            
            last_state = current_state
            current_state = next_state
            
            #print("Goodbye!")



def main():
    
    collect_config()
    
    recommender = RestaurantRecommender()
    
    recommender.run()
    
    
if __name__ == "__main__":
    main()
   
   