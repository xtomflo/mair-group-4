import csv

stateTypes=['Process','Decision','Terminator']
transitionTypes=['Line']

validFoodTypes=['any']
validAreas=['any']
validPriceRanges=['any']

allRestaurants=[]

class State:
    def __init__(self,id, name, type):
        self.id=id
        self.name = name
        self.type = type
        self.transitions=[]

class Restaurant:
    def __init__(self,name, priceRange, area,food,phone,addr,postCode):
        self.name=name
        self.priceRange = priceRange
        self.area = area
        self.food = food
        self.phone = phone
        self.addr = addr
        self.postCode = postCode
        

def getRestaurantData():
    file = open('restaurant_info.csv')  
    csvreader = csv.reader(file)
    for row in csvreader:
            name=row[0] 
            priceRange=row[1]
            area=row[2]
            foodType=row[3]
            phone=row[4]
            addr=row[5]
            postCode=row[6]
            r=Restaurant(name, priceRange, area,foodType,phone,addr,postCode)
            allRestaurants.append(r)
            """
            if foodType not in validFoodTypes:
                validFoodTypes.append(foodType)
            if area not in validAreas:
                validAreas.append(area)
            if priceRange not in validPriceRanges:
                validPriceRanges.append(priceRange)
            """


class StateMachine:
    def __init__(self):
        self.states=[]
        self.collectedInformation={}
        self.initialState=3 # 3 is always the initial state, since it is the welcome state
    def serealize(self): 
        #
        file = open('diagram.csv')  
        csvreader = csv.reader(file)
        for row in csvreader:
            id=row[0]
            type=row[1]
            text=row[11]
            if (type in stateTypes):
                state=State(id,text,type)
                self.states.append(state)
            elif type in transitionTypes:
                src=row[6]
                dest=row[7]
                transition=text
                for s in self.states:
                    if s.id==src:
                        s.transitions.append((src,dest,transition))
        file.close()
        getRestaurantData()
    def getState(self,id): # this function returns the state based on a provided id, so as to get the name of the state or the availbale transiitons
        for s in self.states:
            if s.id==id:
                return s
    def makeTransition(self,input):
        #TODO
        # classify input
        # extract info and add it to the collected info
        # if all information is extracted make a recommendation
        # or answer to a request for postcode or address 
        # make transition (change the current state in the class, based on the possible ones for the current state)
        # if newstate type is 'Process' then some message should be generated from the programm
        # if newstate type is 'Decision' then make some check (is all the needed infrmation known, )
        return 

    def print(self):
        for s in self.states:
            print("id",s.id,"name",s.name,"type",s.type)
            for t in s.transitions:
                srcName=self.getState(t[0])
                destName=self.getState(t[1])
                print("src",srcName.name,"dest",destName.name,"transition",t[2])
        



stateMachine=StateMachine()
stateMachine.serealize()
stateMachine.print()

print(validFoodTypes)
print(validAreas)
print(validPriceRanges)