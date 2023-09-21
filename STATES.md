states have a name and type. Type can be TALK or CHECK

1. Welcome of type TALK which transitions to 2. 
2. Area expressed? of type CHECK, check if area fits, transitions to 4. if true and if false to 3.
3. Ask for area of type TALK, wait for reply, transitions to 2. 
4. Food type expressed? check if food_type fits, of type CHECK
5. Ask for food type of type TALK, wait for reply, transitions to 4.
6. Price range expressed? of type CHECK, check if price_range fits, transitions to 8.  if true and if false to 7.
7. Ask for price_range of type TALK,  wait for reply, transitions to 6.
8. Check if restaurant exists, of type CHECK, check if fitting restaurant available in dataframe, if true transition to 10. if false to 9.
9. Inform no matching restaurant available. Ask to state new preference, of type TALK, transition to 2.
10. Give recommendation, of type TALK, display restaurant info from dataframe

3. Welcome
- Welcome to UU restaurant system. We can provide recommendations based on area, food type and price range. How may I help you?

4. Area known? 
- What part of town do you prefer?

6. Food type known?
- What type of food do you prefer?

9. Price range known?
- What price range do you prefer? 
- Do you prefer something in the cheap, moderate, or expensive price range?

X. Loop
- so any part in town?
- so any type of food?
- so any price range?
  
X. Inform user that there is no restaurant.
- Unfortunately there is no restaurant matching your preferences. Do you have something else in mind?
  No match food type
- i am sorry, but there is no restaurant serving [food_type] food.
  No match area
- i am sorry, but there is no restaurant in [area] part of town.
  No match price range
- i am sorry, but there is no restaurant in the [price_range] price range.

12. Give recommendation
[price_range] = cheap, moderate, expensive
[area] = north, south, east, west

when food, price, area known:
- [restaurant] serves [food_type] food in the in the [area] of town in a [price_range] price range.
  
when food and price is known:
- [restaurant] serves [food_type] food in the [price_range] price range.
  
when food and area is known:
- [restaurant] serves [food_type] food in the in the [area] of town

when price and area is known:
- [restaurant] is in the [area] of town and serves food in the [price_range] price range.

19. Output information
phone number:
- the phone number of [restaurant] is [phone_number]
- the address of [restaurant] is [address]
- the postcode of [restaurant] is [postcode]

phone number, address and postcode
- the phone number of [restaurant] is [phone_number] and it is on [address], [postcode].

phone number and address
- the phone number of [restaurant] is [phone_number] and it is on [address].

phone number and postcode
- the phone number of [restaurant] is [phone_number] and the postcode is [postcode].

post code and address
- the address of [restaurant] is [address]and the postcode is [postcode].

18. apologise for inconvenience 
- Unfortunately this information is not available, I apologise for the inconvenience.





# HowDoWeDoThis?
1. World vs International cuisine
We're given test prompts:
*I'm looking for world food
I want a restaurant that serves world food*

But there's no "World" cuisine restaurant in the list, we have just "International", which should probably be the match. 

2. Handling for diff dialog_acts in different states. 
3. Handling for 'don't care' type of preference



