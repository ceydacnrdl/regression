import numpy as np

a=np.array([[3,5,-9],[-8,-6,5],[10,3,-6]])
b= np.array([[-3],[10],[-8]])

c=np.dot(np.linalg.inv(a),b)
print('Coefficients:')
print(c)

c2=np.linalg.lstsq(a,b,rcond=None)[0]
print('Coefficients of Least Square method:')
print(c2)

print('-'*50)
print('Predict:')
print(np.dot(a,c))
print('Predict of Least Square method:')
print(np.dot(a,c2))

