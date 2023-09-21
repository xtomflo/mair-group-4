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






# HowDoWeDoThis?
1. World vs International cuisine
We're given test prompts:
*I'm looking for world food
I want a restaurant that serves world food*

But there's no "World" cuisine restaurant in the list, we have just "International", which should probably be the match. 

2. Handling for diff dialog_acts in different states. 
3. Handling for 'don't care' type of preference



