import numpy as np

def format__1(digits,num):
        if digits<len(str(num)):
            raise Exception("digits<len(str(num))")
        return ' '*(digits-len(str(num))) + str(num)
        
def format__2(digits,num):
        if digits<len("{0:.4f}".format(num)):
            raise Exception("digits<len(str(num))")
        return ' '*(digits-len("{0:.4f}".format(num))) + "{0:.4f}".format(num)



def printmat(arr): #print a 2d numpy array (maybe) or nested list
    max_chars = max([len(str(item)) for item in np.ndarray.flatten(arr)] ) #the maximum number of chars required to display any item in list
    for row in arr:
        print('[%s]\n' %(' '.join(format__1(max_chars,i) for i in row)))

def stringmat(arr): #print a 2d numpy array (maybe) or nested list
    max_chars = max([len(str(item)) for item in np.ndarray.flatten(arr)] ) #the maximum number of chars required to display any item in list
    s=''
    for row in arr:
        s+='[%s]\n' %(' '.join(format__1(max_chars,i) for i in row))
    return s

def stringmat_compressed(arr): #print a 2d numpy array (maybe) or nested list
    max_chars = max([len("{0:.4f}".format(item)) for item in np.ndarray.flatten(arr)] ) #the maximum number of chars required to display any item in list
    s=''
    for row in arr:
        s+='[%s]\n' %(' '.join(format__2(max_chars,i) for i in row))
    return s



allclose = lambda a, b: np.allclose(a, b, rtol=1e-4, atol=1e-6)
