from bimodal import Bimodal
from gshare import Gshare
from hybrid import Hybrid
from smith import smith_n_bit
import codecs
import sys

b= m1= m2= n= k= 0
file= ' '
address= []
branch= []


def transform_file():
    file = codecs.open(traceFilename,'r', 'utf-8')
    lines = file.readlines()
    address = []
    branch= []
    T= 0    # initialize taken to 0
    NT= 0   # initialize not taken to 0

    for i in range(len(lines)):
        # if second string is either 't' or 'n'
        if lines[i].split(" ")[1][ :1]=='t' or lines[i].split(" ")[1][ :1]=='n':
            #join first string to address
            address.append(lines[i].split(" ")[0])
            branch.append(lines[i].split(" ")[1][ :1])
            #join second string to branch
            if lines[i].split(" ")[1][ :1]=='t':
                T+=1
            else: 
                NT+=1
    # add 0 to address less than 8
    for i in range(len(address)):
        if len(address[i])<8:
            while len(address[i])!=8:
                address[i]='0'+ address[i]
        
    #convert hex format to decimal format
    for i in range(len(address)):
        address2 = address[i]
        address2 = int(address2,16)
        address[i]= address2
    
    return address, branch


# terminal argument for predictor smith, bimodal, gshare, hybrid
predictor = sys.argv[1]

if predictor== 'smith':
    b= int(int(sys.argv[2]))
    traceFilename= sys.argv[3]

if predictor== 'bimodal':
    m2 = int(int(sys.argv[2]))
    traceFilename = sys.argv[3]
    
elif predictor== 'gshare':
    m1 = int(sys.argv[2])
    n = int(sys.argv[3])
    traceFilename = sys.argv[4]

elif predictor== 'hybrid':
    k= int(sys.argv[2])
    m1= int(sys.argv[3])
    n= int(sys.argv[4])
    m2= int(sys.argv[5])
    traceFilename= sys.argv[6]
    
#initialize bimodal, gshare, and hybrid
sb= smith_n_bit(b)
bm= Bimodal(m2)
gs= Gshare(m1, n)
hb= Hybrid(k, m1, n, m2)

# function that return addres, branch from transform file
# a funcction that seprates address and branch
address, branch= transform_file()

if predictor== 'smith':
    for i in range(len(branch)):
        prediction= sb.predictOutcome(branch[i])
    print("COMMAND")
    print("python3 branch_sim/sim.py smith", b, traceFilename)
    sb.display()
    sb.counterDisplay()
    
    

elif predictor== 'bimodal':
    for i in range(len(branch)):
        prediction = bm.predict(branch[i],address[i])
    print("COMMAND")
    print("python3 branch_sim/sim.py bimodal",m2, traceFilename)
    bm.display()
    bm.counterDisplay()


elif predictor== 'gshare':
    for i in range(len(branch)):
        prec = gs.predict(branch[i],address[i])
        gs.update(branch[i])
        gs.updatetable(prec,branch[i])
    print("COMMAND")
    print("python3 branch_sim/sim.py gshare", m1, n, traceFilename)
    gs.display()
    gs.counterDisplay()
    
elif predictor== 'hybrid':
    for i in range(len(branch)):
        hb.branchPred_selector(address[i],branch[i])
    print("COMMAND")
    print("python3 branch_sim/sim.py hybrid", k, m1, n, m2, traceFilename)
    hb.display()
    hb.display_hybrid_counter()
    hb.display_gShare_counter()
    hb.display_bimodal_counter()
   
