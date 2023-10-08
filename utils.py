import enum
from fuzzywuzzy import fuzz
import pyttsx3
import os
from Levenshtein import distance as levenshtein_distance
    

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
        

rules = [
    (("cheap", "good food"), "touristic", True),
    (("romanian",), "touristic", False),
    (("busy",), "assigned seats", True),
    (("long stay",), "children", False),
    (("busy",), "romantic", False),
    (("long stay",), "romantic", True),
    ((), "none", True),
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


def speak(text):
    os.system(f'say {text}')



def classifyRequest(utterance):
        if "address"  in utterance or "location" in utterance or "place" in utterance:
            return "address"
        if "postcode"  in utterance or 'post code'in utterance or "postal" in utterance:
            return "postcode"
        if "phone" in utterance or "number" in utterance or "contact" in utterance: 
            return "phone"
        

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


def extract_keyword(key,word):
    if key is not None:
        vocab=keywords[key]
        if levenshtein_distance(word,'any')<2:
            return (key,'any')
        for v in vocab:
            if levenshtein_distance(v,word)<3:
                return (key,v)
    else:
        for keyword in keywords:
            vocab=keywords[keyword]
            for v in vocab:
                if levenshtein_distance(v,word)<3:
                    return (key,v)
    return (key,None)
def pattern_matching(utterance):
    res={}
    words = utterance.lower().split()
    for (i,w) in enumerate(words):
        if w=='price' or  w=='price_range' or w=='cost' or w=='priced' :
            key='price_range'
            if i>0:
                key,value=extract_keyword(key,words[i-1])
            if value is not None:
                if key not in res:
                    res[key]=value
            if i<len(words)-1:
                key,value=extract_keyword(key,words[i+1])
            if value is not None:
                if key not in res:
                    res[key]=value
        elif w=='food' or w=='cuisine' or w=='flavour':
            key='food_type'
            if i>0:
                key,value=extract_keyword(key,words[i-1])
            if value is not None:
                if key not in res:
                    res[key]=value
            if i<len(words)-1:
                key,value=extract_keyword(key,words[i+1])
            if value is not None:
                if key not in res:
                    res[key]=value
        elif w=='area' or w=='part' or w=='location' or w=='position':
            key='area'
            if i>0:
                key,value=extract_keyword(key,words[i-1])
            if value is not None:
                if key not in res:
                    res[key]=value
            if i<len(words)-1:
                key,value=extract_keyword(key,words[i+1])
            if value is not None:
                if key not in res:
                    res[key]=value
        elif w=='restaurant':
            if i>0:
                key,value=extract_keyword(None,words[i-1])
            if value is not None:
                if key not in res:
                    res[key]=value
            if i<len(words)-1:
                key,value=extract_keyword(key,words[i+1])
            if value is not None:
                if key not in res:
                    res[key]=value

    return res


u="I'm looking for a moderately priced restaurant in the west part of town"
u2="I'm looking for a restaurant in any area that serves Tuscan food"
print(pattern_matching(u))
print(pattern_matching(u2))


