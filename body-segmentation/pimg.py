from PIL import Image
import numpy as np
import sys
LINE = sys.argv[1]
print(LINE)
f = open('./hi.txt','r').read().split('\n')
index = -1
for i,val in enumerate(f):
    if LINE in val:
        index = i
        break
k = f[index + 3]
print(f[index],f[index+1])
k = k[k.find('{'):]
u = eval(k)
u = [u[i] for i in range(len(u))]
u = np.array(u)
pixels = u
pixels = 255 * (1.0 - pixels)
pixels.resize((667,1000))
im = Image.fromarray(pixels.astype(np.uint8), mode='L')
im.show()
