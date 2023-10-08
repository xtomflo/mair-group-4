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

#the key that we get here can only be area, food_type, price_range, or None

def extract_keyword(key,word):
    #if it is not None we get the vocabulary for this property and we try to match the adjacent words with some of the words that we have
    if key is not None:
        vocab=keywords[key]
        #for the word any we use a smaller leveshtein distance beacuse the word itself is short
        if levenshtein_distance(word,'any')<2:
            return (key,'any')
        for v in vocab:
            if levenshtein_distance(v,word)<3:
                return (key,v)
    #if it is None we try to match it with all the words from the vocabs, no matter the class
    else:
        for keyword in keywords:
            vocab=keywords[keyword]
            for v in vocab:
                if levenshtein_distance(v,word)<3:
                    return (key,v)
    #in case we find no matches we return None for the word, and the same key we got
    return (key,None)

#The idea is that we go through the utterance and search for words that are correlated with the 3 different properties that we are trying to extract
def pattern_matching(utterance):
    value=None
    res={}
    words = utterance.lower().split()
    for (i,w) in enumerate(words):
        if w=='price' or  w=='price_range' or w=='cost' or w=='priced' :
            #we set the key based on the case in which we are in 
            key='price_range'
            #if there is a next word we pass in onto the other function and try to match it with a word that makes sense to be in proximity to this one
            if i>0:
                key,value=extract_keyword(key,words[i-1])
            if value is not None:
                if key not in res:
                    res[key]=value
            #we do the same with next word if such exists
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
        # in case where the adjective which is describing the restaurnat we intially don't know for which property it is supposed to be, so first we need to 
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



