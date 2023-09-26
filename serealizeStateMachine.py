import csv

stateTypes=['Process','Decision','Terminator','Connector']
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
        self.utterances=[]

class StateMachine:
    def __init__(self):
        self.states=[]
        self.currentState=1 # 1 is always the initial state, since it is the welcome state
    def serealize(self): 
        #
        file = open('diagram.csv')  
        csvreader = csv.reader(file)
        for row in csvreader:
            id=row[0]
            type=row[1]
            text=row[11]
            if (type in stateTypes):
                state=State(int(id),text,type)
                self.states.append(state)
            elif type in transitionTypes:
                src=row[6]
                dest=row[7]
                transition=text
                for s in self.states:
                    if s.id==int(src):
                        s.transitions.append((int(src),int(dest),transition))
        file.close()
        self.serealizeMachineQuestions()

    def getState(self,id): # this function returns the state based on a provided id, so as to get the name of the state or the availbale transiitons
        for s in self.states:
            if s.id==id:
                return s
  
    def print(self):
        for s in self.states:
            print("id",s.id,"name",s.name,"type",s.type)
            for t in s.transitions:
                srcName=self.getState(t[0])
                destName=self.getState(t[1])
                print("src",srcName.name,"dest",destName.name,"transition",t[2])

    def serealizeMachineQuestions(self):
        file = open('machine_utterances.txt')
        for row in file:
            data=row.split(';')  
            state=self.getState(int(data[0]))
            for i in range(1,len(data)):
                state.utterances.append(data[i])
                
        



stateMachine=StateMachine()
stateMachine.serealize()
stateMachine.print()
