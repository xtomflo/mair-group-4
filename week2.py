import pandas as pd
from fuzzywuzzy import fuzz

from main import train_models, vectorizer, le

def fuzzy_keyword_match(keyword, text, threshold=80):
    
    words = text.lower().split()
    
    for word in words:
        similarity = fuzz.ratio(keyword, word)
        if similarity > threshold:
            return True
    return False

class RestaurantRecommender:
    def __init__(self):
        # Initial state
        self.state = 'Welcome'
        
        # User preferences
        self.area = None
        self.food_type = None
        self.price_range = None
        
        # Placeholder for restaurant data (You can replace this with real data later)
        self.restaurant_df = pd.read_csv('restaurant_info.csv')


    def extract_preferences(self, utterance):
                
        # Define list of keywords that we're looking to match
        food_types = ['british', 'modern european', 'italian', 'romanian', 'seafood',
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
                self.food_type = food
                break

        # Check for locations
        for location in locations:
            if fuzzy_keyword_match(location, utterance):
                self.location = location
                break

        # Check for price range
        for price in price_range:
            if fuzzy_keyword_match(price, utterance):
                self.price_range = price
                break

        return preferences

    def predict_dialog_act(utterance):
        custom_message_vec = vectorizer.transform([utterance])

        prediction = log_reg.predict(custom_message_vec)
        prediction_1d = np.array([prediction]).ravel() # Change shape to pacify a warning from LabelEncoder
        prediction_label = le.inverse_transform(prediction_1d)
        print(prediction_label)
        
        return prediction_label
    
    def transition(self):
        while True:
            if self.state == 'Welcome':
                utterance = input("Welcome to the Restaurant Recommender! \n \
                    Please state your preferences in terms of the cuisine type, area and price range, I'll help you pick the best restaurant! \n :")
                print(utterance)
                self.extract_preferences(utterance)
                self.state = 'Area expressed?'

            elif self.state == 'Area expressed?':
                # Placeholder for CHECK logic
                
                if self.area is not None:
                    self.state = 'Food type expressed?'
                else:
                    self.state = 'Ask for area'

            elif self.state == 'Ask for area':
                # Placeholder for TALK logic (user interaction)
                utterance = input("What is your preferred area?")
                self.extract_preferences(utterance)
                self.state = 'Area expressed?'

            elif self.state == 'Food type expressed?':
                # Placeholder for CHECK logic
                if self.food_type is not None:
                    self.state = 'Price range expressed?'
                else:
                    self.state = 'Ask for food type'

            elif self.state == 'Ask for food type':
                # Placeholder for TALK logic (user interaction)
                utterance = input("Please enter your preferred food type: ")
                self.extract_preferences(utterance)

                self.state = 'Food type expressed?'

            elif self.state == 'Price range expressed?':
                # Placeholder for CHECK logic
                if self.price_range is not None:
                    self.state = 'Check if restaurant exists'
                else:
                    self.state = 'Ask for price range'

            elif self.state == 'Ask for price range':
                # Placeholder for TALK logic (user interaction)
                utterance = input("Please enter your preferred price range: ")
                self.extract_preferences(utterance)

                self.state = 'Price range expressed?'

            elif self.state == 'Check if restaurant exists':
                # Placeholder for CHECK logic
                # Here, you would typically check if a restaurant exists that fits the criteria
                matching_restaurants = self.restaurant_df  # Placeholder, replace with actual filtering logic
                if not matching_restaurants.empty:
                    self.state = 'Give recommendation'
                else:
                    self.state = 'Inform no matching restaurant available. Ask to state new preference'

            elif self.state == 'Inform no matching restaurant available. Ask to state new preference':
                print("Sorry, no matching restaurant is available. Please state new preferences.")
                self.state = 'Area expressed?'

            elif self.state == 'Give recommendation':
                # Placeholder for TALK logic (user interaction)
                # Here, you would typically show the recommended restaurant from the dataframe
                print("Recommended restaurant: ", "Example Restaurant")
                break



if __name__ == "__main__":

    log_reg = train_models()
    vectorizer = vectorizer
    # Create an instance of the RestaurantRecommender class
    recommender = RestaurantRecommender()

    # Start the state machine
    recommender.transition()
