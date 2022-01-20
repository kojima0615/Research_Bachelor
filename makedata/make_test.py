import encryption
import csv

with open("plain_1000000.csv") as f:
    plain_text = f.readlines()

with open("key_1000000.csv") as f:
    key = f.readlines()

round1 = [0]*400000
#round16 = [0]*1000000
for i in range(1000000):
    round1[i] = encryption.enc(int(plain_text[i]),int(key[i]))
    #print(round16[i])
    #print(round1[i])
    if i % 10000==0:
        print(i)



with open("round1_1000000.csv", "w") as f:
    for ele in round1:
        f.write(str(ele)+'\n')

'''
with open("round16_1000000.csv", "w") as f:
    for ele in round16:
        f.write(str(ele)+'\n')
'''

