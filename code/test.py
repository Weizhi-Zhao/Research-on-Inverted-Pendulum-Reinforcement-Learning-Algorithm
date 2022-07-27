n = [0,0,0,0,0,0,0,0,0,0,0,1]
j = [0,0,0,0,0,0,0,0,0,0,0,0]
def c():
    for i in range(11):
        j[i] = j[i] * 0.9
def s():
    for i in range(1,11):
        cha = n[i+1] - n[i]
        j[i] += 1
        
        for k in range(1,11):
            n[k] += j[k]*cha*0.9
        c()
s()
print(n)
for i in range(1,11):
    j[i] = 0
s()
print(n)
for i in range(1,11):
    j[i] = 0
s()
print(n)