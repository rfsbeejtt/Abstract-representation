import numpy as np
import os

print('\n KONVNET')

path='./5stck_konvnet/'
files=os.listdir(path)
succes_list=[]
for file in files:
    konvnetresult=np.load(path+file)
    unique, counts = np.unique(konvnetresult, return_counts=True)
    given_values=dict(zip(unique, counts))
    one_konvnet = given_values[1]
    zero_konvnet = given_values[0]
    tried_konvnet=zero_konvnet+one_konvnet
    succes_list.append(one_konvnet / tried_konvnet)


resiliance_list=[]
for i in succes_list:
    resiliance_list.append(1-i)



print('Mean Resilience:  '+str(np.mean(resiliance_list)))
print('Mean Attack Succes:  '+str(np.mean(succes_list)))
print('Variance: '+str(np.std(resiliance_list)))
print('Max Attack: '+str(np.amax(succes_list)))
print('Min Attack: '+str(np.amin(succes_list)))
print('Median Attack: '+str(np.median(succes_list)))
print(len(succes_list))

path='./model_acc_bn/'
files=os.listdir(path)
acc_list=[]
for file in files:
    konvnetacc=np.load(path+file)
    acc_list.append(konvnetacc)


print('Accuracy Average:: '+str(np.mean(acc_list)))
print('Accuracy Variance:  '+ str(np.std(acc_list)))
print(len(acc_list))


path='./successrate_abs_model/'

files=os.listdir(path)
succes_list_abs=[]
for file in files:
    abresult=np.load(path+file)
    unique_ab, counts_ab = np.unique(abresult, return_counts=True)
    given_values_abs=dict(zip(unique_ab, counts_ab))
    one_abs = given_values_abs[1]
    zero_abs = given_values_abs[0]
    tried_abs=zero_abs+one_abs
    succes_list_abs.append(one_abs / tried_abs)



resiliance_list_abs=[]
for i in succes_list_abs:
    resiliance_list_abs.append(1-i)

print('\n \n \n ABSTRACT NET:')

print('Mean Resilience:  '+str(np.mean(resiliance_list_abs)))
print('Mean Attack Succes:  '+str(np.mean(succes_list_abs)))
print('Variance: '+str(np.std(resiliance_list_abs)))
print('Max Attack: '+str(np.amax(succes_list_abs)))
print('Min Attack: '+str(np.amin(succes_list_abs)))
print('Median Attack: '+str(np.median(succes_list_abs)))
print(len(succes_list_abs))

path=r'./model_acc_abs/'
files=os.listdir(path)
abs_acc_list=[]
for file in files:
    abs_acc=np.load(path+file)
    abs_acc_list.append(abs_acc)

print('Accuracy Average:: '+str(np.mean(abs_acc)))
print('Accuracy Variance:  '+ str(np.std(abs_acc)))
