#import argparse
#parser = argparse.ArgumentParser(description='')
#parser.add_argument("-f", type=str, required=True)
#args = parser.parse_args()

#f = open(args.f)
f = open("log.txt")
a = f.readlines()
f.close()
num_gpus = 8
m1, n1, N1, m2, n2, N2 = [], [], [], [], [], []
import numpy as np
for i in a:
 if len(i.split()) == 6 and i.split()[0].isdigit():
  m1.append(float(i.split()[0]))
  n1.append(float(i.split()[1]))
  N1.append(float(i.split()[2]))
  m2.append(float(i.split()[3]))
  n2.append(float(i.split()[4]))
  N2.append(float(i.split()[5]))

m1, n1, N1 = np.array(m1), np.array(n1), np.array(N1)
m2, n2, N2 = np.array(m2), np.array(n2), np.array(N2)
print(m1.shape, n1.shape, N1.shape)
print(m2.shape, n2.shape, N2.shape)
#print(np.sum(N1==0), np.sum(N2==0))
#assert False
m1 += 1e-6
n1 += 1e-6
N1 += 1e-6
m2 += 1e-6
n2 += 1e-6
N2 += 1e-6

for i in range(6000+2000, 18000+1, 2000):
    start, finish = (i-8000)*num_gpus, (i-6000)*num_gpus
    print("iter {} - {}, Model A, Precision: {:.4f}, Recall: {:.4f}, Empty Ratio: {:.4f}, m: {:.2f}, n: {:.2f}, N: {:.2f}".format(i-2000, i, np.mean(m1[start:finish]/N1[start:finish]), np.mean(m1[start:finish]/n1[start:finish]), np.sum(N1[start:finish]-1e-6 == 0)/N1[start:finish].shape[0], m1[start:finish].mean(), n1[start:finish].mean(), N1[start:finish].mean()))
    #print("iter {} - {}, Model B, Precision: {:.4f}, Recall: {:.4f}, Empty Ratio: {:.4f}, m: {:.2f}, n: {:.2f}, N: {:.2f}".format(i-2000, i, np.mean(m2[start:finish]/N2[start:finish]), np.mean(m2[start:finish]/n2[start:finish]), np.sum(N2[start:finish]-1e-6 == 0)/N2[start:finish].shape[0], m1[start:finish].mean(), n1[start:finish].mean(), N1[start:finish].mean()))

'''
f = open("log.log")
a = f.readlines()
f.close()
n1, N1, n2, N2 = [], [], [], []
num_gpus = 8
import numpy as np
for i in a:
 if len(i.split()) == 4 and i.split()[0].isdigit():
  n1.append(float(i.split()[0]))
  N1.append(float(i.split()[1]))
  n2.append(float(i.split()[2]))
  N2.append(float(i.split()[3]))

n1, N1, n2, N2 = np.array(n1), np.array(N1), np.array(n2), np.array(N2)
print(n1.shape, N1.shape, n2.shape, N2.shape)

for i in range(6000+2000, 18000+1, 2000):
    start, finish = (i-8000)*num_gpus, (i-6000)*num_gpus
    print("iter {} - {}".format(i-2000, i), "Acc1: {:.4f}, Empty Ratio: {:.4f}".format(((n1+1e-6)/(N1+1e-6))[start:finish].mean(), np.sum(N1[start:finish] == 0)/(finish-start)), "Acc2: {:.4f}, Empty Ratio: {:.4f}".format(((n2+1e-6)/(N2+1e-6))[start:finish].mean(), np.sum(N2[start:finish] == 0)/(finish-start)))
'''
