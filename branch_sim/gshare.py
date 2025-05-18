import math 

class Gshare:
    # CONSTRUCTOR GSHARE
    def __init__(self,m,n):
        self.m = m
        self.n = n
        # PREDICTION TABLE SIZE = 2^M
        self.predictionTableSize = int(math.pow(2,self.m))
        self.globalHistoryRegister = 0
        self.predictions = 0
        self.mispredictions = 0
        # SET EXTREMES FOR SATURATION
        self.max=7
        self.taken=4
        # INITIALIZE ALL ENTRIES IN PRED TABLE TO 4 ("WEAKLY TAKEN")
        self.table = [4 for i in range(self.predictionTableSize)]
        self.index =0

    def predict(self, branch, address):
        # INCREMENT THE NUMBER OF TOTAL PREDICTIONS
        self.predictions+=1
        n_bits = int((address/4)%math.pow(2,self.n))
        m_bits = int((address/4)%self.predictionTableSize)
        # EXTRACT THE LOWER M BITS
        m_n_bits = int(m_bits/math.pow(2,self.n))
        xor = n_bits^self.globalHistoryRegister
        # LEFT SHIFT UNTIL THE BITS LINE UP
        self.index = m_n_bits<<self.n
        # USE THE XOR'd BIT MATH TO DETERMINE INDEX
        self.index=self.index+xor
        prec='n'
        # DETERMINE PREDICTION DIRECTION BASED ON VALUE AT INDEX
        if self.table[self.index]>=self.taken:
            prec='t'
        return prec

    def update(self,branch):
        # UPDATE COUNTEER BASED ON PREDICTION DIRECTION AND BRANCH PREDICTION
        # ENSURES THAT SATURATION OCCURS AT EXTREMES (0,7)
        if branch=='t' and self.table[self.index]<self.max:
            self.table[self.index]+=1
        elif branch=='n' and self.table[self.index]>0:
            self.table[self.index]-=1


    def updatetable(self,prec,branch):
        # CHECK IF PREDICTION DIRECTION AND CONTENT IN TABLE MATCH;
        # IF NOT --> MISPREDICTION
        if prec!=branch:
            self.mispredictions+=1
        # SHIFT REGISTER RIGHT BY 1 BIT PER PREDICTION DIRECTION UPDATE
        self.globalHistoryRegister=self.globalHistoryRegister>>1
        if branch=='t':
            self.globalHistoryRegister=self.globalHistoryRegister+int(math.pow(2,self.n-1))

    def counterDisplay(self):
        print("FINAL GSHARE CONTENTS:")
        for i in range(self.predictionTableSize):
             print(f"{i}       {self.table[i]}")

    def display(self):
        print("number of predictions:")
        print(self.predictions)
        print("number of mispredictions:")
        print(self.mispredictions)
        print("misprediction rate:")
        misPred_rate= self.mispredictions*100/self.predictions
        print("{:.2f}".format(misPred_rate),"%")
