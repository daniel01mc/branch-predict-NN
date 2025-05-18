import math 

min_nt = 0 # minimum taken
weakly_t= 4 #weakly taken
max_t= 7  # max taken


class Bimodal:
    # class Bimodal initial
    def __init__(self, m):
        self.indx= 0
        self.predCount= 0 
        self.misPredCount= 0
        self.size= int(math.pow(2, m))  # size is 2^m, m is entry from terminal
        self.mask= self.size -1 #create mask with size-1
        self.CT= [weakly_t for i in range(self.size)] #initialize counter table elements to 4
       
    
        # set index by shifting and masking with size-1 
    def indexSetup(self, address):
        self.indx= address >>2
        self.indx &= self.mask
        

    def update(self, branch, prediction):
        if (branch=='t'):
            if (self.CT[self.indx]< max_t): self.CT[self.indx]+= 1  #increment index if counter table index< 7
            if (prediction == 'n'): 
                self.misPredCount+= 1 # increment misprediction
        elif (branch== 'n'): 
            if(self.CT[self.indx]> min_nt): self.CT[self.indx]-= 1 #decrement index if counter table > 0
            if(prediction== 't'):
                self.misPredCount+= 1 # increment misprediction
        
     
     #function that interacts with bimodal section, return 't' or 'n' based on address and index
    def bm_predict(self, address): 
        self.indexSetup(address) 
        if (self.CT[self.indx]>= weakly_t): return 't'
        else: return 'n'


        #predic based no branches, address
        #input: brances('y', 'n'), and address
        #ouput: update branch prediction based 'y' or 'n'
    def predict(self, branch, address):
        self.predCount+=1
        self.indexSetup(address)
        if (self.CT[self.indx]>= weakly_t): prediction= 't'
        else: prediction= 'n'
        return self.update(branch, prediction)

     
      # display counter contents
    def counterDisplay(self):
        print("FINAL BIMODAL CONTENTS: ")
        for i in range(self.size):
            print(f"{i}       {self.CT[i]}")

    
        # display predictions and mispredictions
    def display(self):
        print("number of predictions:")
        print(self.predCount)
        print("number of mispredictions:")
        print(self.misPredCount)
        print("misprediction rate:")
        misPred_rate= self.misPredCount*100/self.predCount
        print("{:.2f}".format(misPred_rate),"%")