# Define the state transition function
def state_transition(current_state, input_data, context):
    """
    Transition between states based on the current state, input data, and context.
    
    Parameters:
    - current_state (str): The current state name.
    - input_data (str): The input from the user.
    - context (dict): A dictionary holding context information.
    
    Returns:
    - tuple: The next state, the output message, and the updated context.
    """
    next_state = ""
    output_message = ""
    if current_state == "Welcome":
        next_state = "Area expressed?"
        output_message = "Welcome! I can help you find a restaurant. Have you already decided on the area?"
        
    elif current_state == "Area expressed?":
        if input_data.lower() == 'yes':
            next_state = "Food type expressed?"
            output_message = "Great! What type of food are you in the mood for?"
        else:
            next_state = "Ask for area"
            output_message = "No problem, where would you like to dine?"
            
    elif current_state == "Ask for area":
        context['area_counter'] = context.get('area_counter', 0) + 1
        context['area'] = input_data
        next_state = "Area expressed?"
        if context['area_counter'] == 2:
            output_message = "Got it. Let's try again. Have you already decided on the area?"
        elif context['area_counter'] >= 4:
            next_state = "Exit"
            output_message = "It seems we're having difficulty with the area. Let's try again later."
        else:
            output_message = "Got it. Have you already decided on the area?"
            
    elif current_state == "Food type expressed?":
        if input_data.lower() == 'yes':
            next_state = "Price range expressed?"
            output_message = "Awesome! What is your preferred price range?"
        else:
            next_state = "Ask for food type"
            output_message = "No worries, what type of food are you interested in?"
            
    elif current_state == "Ask for food type":
        context['food_type_counter'] = context.get('food_type_counter', 0) + 1
        context['food_type'] = input_data
        next_state = "Food type expressed?"
        if context['food_type_counter'] == 2:
            output_message = "Got it. Let's try again. What type of food are you in the mood for?"
        elif context['food_type_counter'] >= 4:
            next_state = "Exit"
            output_message = "It seems we're having difficulty with the food type. Let's try again later."
        else:
            output_message = "Got it. What type of food are you in the mood for?"
            
    elif current_state == "Price range expressed?":
        if input_data.lower() == 'yes':
            next_state = "Check if restaurant exists"
            output_message = "Let me check if a restaurant matches your criteria."
        else:
            next_state = "Ask for price range"
            output_message = "No worries, what is your preferred price range?"
            
    elif current_state == "Ask for price range":
        context['price_range_counter'] = context.get('price_range_counter', 0) + 1
        context['price_range'] = input_data
        next_state = "Price range expressed?"
        if context['price_range_counter'] == 2:
            output_message = "Got it. Let's try again. What is your preferred price range?"
        elif context['price_range_counter'] >= 4:
            next_state = "Exit"
            output_message = "It seems we're having difficulty with the price range. Let's try again later."
        else:
            output_message = "Got it. What is your preferred price range?"
            
    elif current_state == "Check if restaurant exists":
        matches, reason = find_matching_restaurant()
        if matches is not None and not matches.empty:
            next_state = "Give recommendation"
            recommended_restaurant = matches.iloc[0]
            output_message = f"I found a restaurant for you: {recommended_restaurant['restaurantname']} located in {recommended_restaurant['area']}."
        else:
            next_state = "Ask for area"
            output_message = f"Sorry, no matching restaurant found. Reason: {reason}. Can you please state your area preference again?"
            
    elif current_state == "Give recommendation":
        next_state = "Exit"
        output_message = "Hope you enjoy your meal! Goodbye."
        
    elif current_state == "Exit":
        output_message = "Goodbye!"
        
    return next_state, output_message, context

# Initialize variables
context = {}
current_state = "Welcome"
input_data = ""

# Run a sample dialogue
while current_state != "Exit":
    next_state, output_message, context = state_transition(current_state, input_data, context)
    print(f"Bot: {output_message}")
    if next_state == "Exit":
        break
    input_data = input("You: ")
    current_state = next_state
