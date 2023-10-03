from fuzzywuzzy import fuzz
 
rules = [
    (("cheap", "good food"), "touristic", True),
    (("romanian",), "touristic", False),
    (("busy",), "assigned seats", True),
    (("long stay",), "children", False),
    (("busy",), "romantic", False),
    (("long stay",), "romantic", True),
]

def infer_properties(restaurant_properties):
    # Check if properties meet the rules, return the consequences
    inferred_properties = {}
    
    # We apply rules in order
    # For ex. if restaurant is cheap, good and romanian, it being romanian will overwrite touristic status to False
    for antedecent, consequent, value in rules:
        if all(prop in restaurant_properties for prop in antedecent):
            inferred_properties[consequent] = value

    return inferred_properties




def classifyRequest(utterance):
        if "address"  in utterance:
            return "address"
        if "postcode"  in utterance:
            return "postcode"
        if "phone" in utterance: 
            return "phone"
        
def output_utterance(utterance):
    ### Output system utterances to the user
    print(utterance)
    #if CONFIG_TTS:
    #    subprocess.call(["say", utterance])
            
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
