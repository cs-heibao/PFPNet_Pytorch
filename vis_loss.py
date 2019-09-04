import os
import matplotlib.pyplot as plt
path = './loss.txt'
loss = []
with open(path,'r') as fd:
    files = fd.readlines()
files = files[-142000:]
j = 0
for i in range(len(files)):
    file = files[j].strip()
    data = file.find(')')
    if data > 0:
        loss.append(float(file[file.find('(')+1:data]))
        k = j
    else:
        l1 = float(file[file.find('(')+1:-1])
        k = j+1
        file1 = float(files[k].strip()[:-1])
        loss.append(l1*file1)
    j = k + 1
    if j ==len(files):
        break
plt.figure()
plt.title('loss')
plt.plot(range(len(loss)),loss)
plt.savefig('./loss.jpg')
plt.show()
print('ok')