import numpy as np

array = np.zeros(shape = (5, 2, 3))

with open("test.np", "wb") as f: 
    np.save(f, array)
    
with open("test.np", "rb") as f: 
    a = np.load(f)

print(a)