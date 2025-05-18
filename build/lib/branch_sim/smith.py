#Algorith
#determine the branch outcoume 
#make a prediction based on the counter
#update the counter predictor based on branch outcome 

import math

class smith_n_bit:
    def __init__(self,b): # initialize with b, number of bits of counter
        self.predCount= 0     
        self.misPredCount= 0
        self.size= pow(2, b)  # 2^b for counter size
        self.iniNum= (self.size>>1)  # divide counter size by 2, by shifting to get number to initialize
        self.counter= [self.iniNum for var in range (self.size)] # initialize counter size with initial number
       
        
     #function that return true or false if counter is greater or lesser than initial number
    def predict(self):
        if self.counter[self.iniNum]>= self.iniNum: return True 
        else: return False

    #function to update the counter once the branch is known
    def updateCounter(self, branch):
        if branch== 't':    # if the actual branch from the it 't' stand for taken
            self.counter[self.iniNum]+= 1   # increase the number of the counter
            if self.counter[self.iniNum]> self.size -1: # if counter element is greater than the size of the element
                self.counter[self.iniNum]= self.size- 1 #reset to last element
        if branch== 'n':    #'n' stands for not taken
            self.counter[self.iniNum]-= 1  # reset to zero if counter element is less than zero
            if self.counter[self.iniNum]< 0:
                self.counter[self.iniNum]= 0

    #function that interacts with sim.py and prediction and misprediction
    def predictOutcome(self, branch):
        self.predCount+= 1             # make a prediction as a "baseline"
        predictC= self.predict()       # predict true or false based on the counter state
        if (branch=='t' and predictC== False): self.misPredCount+= 1 
        elif (branch== 'n' and predictC== True): self.misPredCount+= 1
        self.updateCounter(branch)
            
    def counterDisplay(self):
                print("FINAL COUNTER CONTENTS:")
                print(self.counter[self.iniNum])

    def display(self):
        print("number of predictions:")
        print(self.predCount)
        print("number of mispredictions:")
        print(self.misPredCount)
        print("misprediction rate:")
        misPred_rate= self.misPredCount*100/self.predCount
        print("{:.2f}".format(misPred_rate),"%")




    

