import numpy as np
#!!!TODO test code
def MajVot(ArrayOfClassPred):#[class pred1,class pred2,class pred3]
                             # where class predis an array
    answer=np.zeros((len(ArrayOfClassPred[0],1)),dtype=int)
    votes=np.zeros((len(ArrayOfClassPred[0],3)),dtype=int)

    for i in ArrayOfClassPred:#i ex=class pred1
        index=0
        for j in i: #j element de classpred1
            if j==0:
                votes[index][0]+=1
            elif j==1:
                votes[index][1]+=1
            else:
                votes[index][2]+=1
            index+=1

    for i in range(len(answer)):
        answer[i]=np.argmax(votes[i,:])
        
        
        
        
    
    return answer