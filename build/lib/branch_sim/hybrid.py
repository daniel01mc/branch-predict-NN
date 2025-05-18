import math 
from bimodal import Bimodal
from gshare import Gshare

strongly_nt=0 #strongly not taken
weakly_nt= 1 #weakly not taken
weakly_t= 2 #weakly taken
strongly_t=3 #strongly taken
bi_weakly_t= 4 # bimodal weakly taken


class Hybrid:
    def __init__(self, k, m1, n, m2):
        self.bm= Bimodal(m2)
        self.gs= Gshare(m1,n)
        self.chtIndx= 0
        self.hb_predCount= 0
        self.hb_misPredCount= 0
        self.size= int(math.pow(2, k))  # size is 2^k, k is user entry defined from terminal
        self.mask= self.size -1 #create mask with size-1
        self.CHT= [weakly_nt for i in range(self.size)] #initialize chooser history table elements to 1
        
    predictBimodal= predictGshare= 'n'
    
    # set index by shifting and masking with size-1 
    def indexSetup(self, address):
        self.chtIndx= address >>2
        self.chtIndx &= self.mask
                
    # Predict counter history table (CHT) based on 
    # counter history table index (chtIndx) status
    def CHTpredict(self):
        if (self.CHT[self.chtIndx] >= weakly_t): return 't' 
        else: return 'n'        

    # main function that interact with sim_py, return both predictions
    # and mispredictions.
    def branchPred_selector(self, address, branch):

        self.hb_predCount+= 1
    
        self.indexSetup(address) 

        # predict bimodal based on address
        predictBimodal= self.bm.bm_predict(address)

        # Predict Gshare based on branch and address
        predictGshare= self.gs.predict(branch, address)
        
        # branch selector is based on status of 
        # counter history register index
        branch_selector = self.CHTpredict()
        
        # if branch selector is taken, upcount gShare
        if branch_selector=='t':
            finalPrediction= predictGshare
            self.gs.update(branch)
        else:
            # if branch selector is not taken, upcount bimodal
            finalPrediction= predictBimodal
            self.bm.update(branch, branch)

        #update global history register regardless of choice
        self.gs.updatetable(finalPrediction,branch)
        
        # Update Counter history register  
        if (predictBimodal== branch and predictGshare!= branch):
            if self.CHT[self.chtIndx]>0:
                self.CHT[self.chtIndx]-= 1
        elif (predictBimodal!= branch and predictGshare== branch):
            if self.CHT[self.chtIndx]<3:
                self.CHT[self.chtIndx]+= 1

        # if final prediction from either gShare or bimodal is 
        # not equal to actual branch, increase misprediction   
        if finalPrediction!= branch:
            self.hb_misPredCount+= 1


    def display(self): 
        print("number of predictions:")
        print(self.hb_predCount)
        print("number of mispredictions:")
        print(self.hb_misPredCount)
        print("misprediction rate:")
        misPred_rate= self.hb_misPredCount*100/self.hb_predCount
        print("{:.2f}".format(misPred_rate),"%")

    def display_hybrid_counter(self):
        print("FINAL CHOOSER CONTENTS:")
        for i in range(self.size):
            print(f"{i}       {self.CHT[i]}")

    def display_gShare_counter(self):
        print("FINAL GSHARE CONTENTS")
        for i in range(self.gs.predictionTableSize):
             print(f"{i}       {self.gs.table[i]}")

    def display_bimodal_counter(self):
        print("FINAL BIMODAL CONTENTS")
        for i in range(self.bm.size):
            print(f"{i}       {self.bm.CT[i]}")


    

