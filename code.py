import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.linear_model import LinearRegression

#Converting the received power equation into dBm for ease of computation.
#Using Linear regression to compute Eta and K values.

#Estimating Sigma using Standard deviation formula:
#sqrt[(summation(Pr-Pr_prime)^2)/Number of Valid Pr]

num_of_valid_Pr=0

#Reading the transmitter and receiver coordinates from the CSV files.
trans_coord=pd.read_csv('transmitterXY.csv', header=None)
recv_coord=pd.read_csv('receiverXY.csv', header=None)

#Creating a dataframe to store the distance and their corresponding received power
df=pd.DataFrame(columns=['log_dist','recv_pow'])

#Experiment 7
recv_power=pd.read_csv('wifiExp7.csv', header=None)    #Reading the received power from the CSV files.
del recv_power[0]   #Deleting the colimn with timestamps
recv_power.columns=[0,1,2,3,4,5,6,7]    #Renumbering the columns

for row in range(0,8):
    #Computing the distance between Transmitter and Receiver using Pythagoras theorem.
    d=np.sqrt((float(trans_coord[0][0])-float(recv_coord[0][row]))**2 + (float(trans_coord[1][0])-float(recv_coord[1][row]))**2) 
    d=10*math.log10(d)
    for i in range(0,len(recv_power)):
        if recv_power[row][i] != 500:
            #Storing the distance and the corresponding power values(Valid ones) in the dataframe.
            df=df.append({'log_dist':d,'recv_pow':-recv_power[row][i]}, ignore_index=True)
            num_of_valid_Pr+=1  #Computing the number of power values that are valid - Non 500dBm values. 


#Experiment 8
recv_power=pd.read_csv('wifiExp8.csv', header=None)
del recv_power[0]
recv_power.columns=[0,1,2,3,4,5,6,7]
for row in range(0,8):
    d=np.sqrt((float(trans_coord[0][1])-float(recv_coord[0][row]))**2 + (float(trans_coord[1][1])-float(recv_coord[1][row]))**2)
    d=10*math.log10(d)
    for i in range(0,len(recv_power)):
        if recv_power[row][i] != 500:
            df=df.append({'log_dist':d,'recv_pow':-recv_power[row][i]}, ignore_index=True)
            num_of_valid_Pr+=1

#Experiment 9
recv_power=pd.read_csv('wifiExp9.csv', header=None)
del recv_power[0]
recv_power.columns=[0,1,2,3,4,5,6,7]
for row in range(0,8):
    d=np.sqrt((float(trans_coord[0][2])-float(recv_coord[0][row]))**2 + (float(trans_coord[1][2])-float(recv_coord[1][row]))**2)
    d=10*math.log10(d)
    for i in range(0,len(recv_power)):
        if recv_power[row][i] != 500:
            df=df.append({'log_dist':d,'recv_pow':-recv_power[row][i]}, ignore_index=True)
            num_of_valid_Pr+=1

#Experiment 10
recv_power=pd.read_csv('wifiExp10.csv', header=None)
del recv_power[0]
recv_power.columns=[0,1,2,3,4,5,6,7]
for row in range(0,8):
    d=np.sqrt((float(trans_coord[0][3])-float(recv_coord[0][row]))**2 + (float(trans_coord[1][3])-float(recv_coord[1][row]))**2)
    d=10*math.log10(d)
    for i in range(0,len(recv_power)):
        if recv_power[row][i] != 500:
            df=df.append({'log_dist':d,'recv_pow':-recv_power[row][i]}, ignore_index=True)
            num_of_valid_Pr+=1


#Experiment 11
recv_power=pd.read_csv('wifiExp11.csv', header=None)
del recv_power[0]
recv_power.columns=[0,1,2,3,4,5,6,7]
for row in range(0,8):
    d=np.sqrt((float(trans_coord[0][4])-float(recv_coord[0][row]))**2 + (float(trans_coord[1][4])-float(recv_coord[1][row]))**2)
    d=10*math.log10(d)
    for i in range(0,len(recv_power)):
        if recv_power[row][i] != 500:
            df=df.append({'log_dist':d,'recv_pow':-recv_power[row][i]}, ignore_index=True)
            num_of_valid_Pr+=1

#Experiment 12
recv_power=pd.read_csv('wifiExp12.csv', header=None)
del recv_power[0]
recv_power.columns=[0,1,2,3,4,5,6,7]
for row in range(0,8):
    d=np.sqrt((float(trans_coord[0][5])-float(recv_coord[0][row]))**2 + (float(trans_coord[1][5])-float(recv_coord[1][row]))**2)
    d=10*math.log10(d)
    for i in range(0,len(recv_power)):
        if recv_power[row][i] != 500:
            df=df.append({'log_dist':d,'recv_pow':-recv_power[row][i]}, ignore_index=True)
            num_of_valid_Pr+=1


#Experiment 13
recv_power=pd.read_csv('wifiExp13.csv', header=None)
del recv_power[0]
recv_power.columns=[0,1,2,3,4,5,6,7]
for row in range(0,8):
    d=np.sqrt((float(trans_coord[0][6])-float(recv_coord[0][row]))**2 + (float(trans_coord[1][6])-float(recv_coord[1][row]))**2)
    d=10*math.log10(d)
    for i in range(0,len(recv_power)):
        if recv_power[row][i] != 500:
            df=df.append({'log_dist':d,'recv_pow':-recv_power[row][i]}, ignore_index=True)
            num_of_valid_Pr+=1


#Experiment 14
recv_power=pd.read_csv('wifiExp14.csv', header=None)
del recv_power[0]
recv_power.columns=[0,1,2,3,4,5,6,7]
for row in range(0,8):
    d=np.sqrt((float(trans_coord[0][7])-float(recv_coord[0][row]))**2 + (float(trans_coord[1][7])-float(recv_coord[1][row]))**2)
    d=10*math.log10(d)
    for i in range(0,len(recv_power)):
        if recv_power[row][i] != 500:
            df=df.append({'log_dist':d,'recv_pow':-recv_power[row][i]}, ignore_index=True)
            num_of_valid_Pr+=1


#Experiment 15
recv_power=pd.read_csv('wifiExp15.csv', header=None)
del recv_power[0]
recv_power.columns=[0,1,2,3,4,5,6,7]
for row in range(0,8):
    d=np.sqrt((float(trans_coord[0][8])-float(recv_coord[0][row]))**2 + (float(trans_coord[1][8])-float(recv_coord[1][row]))**2)
    d=10*math.log10(d)
    for i in range(0,len(recv_power)):
        if recv_power[row][i] != 500:
            df=df.append({'log_dist':d,'recv_pow':-recv_power[row][i]}, ignore_index=True)
            num_of_valid_Pr+=1

#Experiment 16
recv_power=pd.read_csv('wifiExp16.csv', header=None)
del recv_power[0]
recv_power.columns=[0,1,2,3,4,5,6,7]
for row in range(0,8):
    d=np.sqrt((float(trans_coord[0][9])-float(recv_coord[0][row]))**2 + (float(trans_coord[1][9])-float(recv_coord[1][row]))**2)
    d=10*math.log10(d)
    for i in range(0,len(recv_power)):
        if recv_power[row][i] != 500:
            df=df.append({'log_dist':d,'recv_pow':-recv_power[row][i]}, ignore_index=True)
            num_of_valid_Pr+=1

#Experiment 17
recv_power=pd.read_csv('wifiExp17.csv', header=None)
del recv_power[0]
recv_power.columns=[0,1,2,3,4,5,6,7]
for row in range(0,8):
    d=np.sqrt((float(trans_coord[0][10])-float(recv_coord[0][row]))**2 + (float(trans_coord[1][10])-float(recv_coord[1][row]))**2)
    d=10*math.log10(d)
    for i in range(0,len(recv_power)):
        if recv_power[row][i] != 500:
            df=df.append({'log_dist':d,'recv_pow':-recv_power[row][i]}, ignore_index=True)
            num_of_valid_Pr+=1


#Experiment 18
recv_power=pd.read_csv('wifiExp18.csv', header=None)
del recv_power[0]
recv_power.columns=[0,1,2,3,4,5,6,7]
for row in range(0,8):
    d=np.sqrt((float(trans_coord[0][11])-float(recv_coord[0][row]))**2 + (float(trans_coord[1][11])-float(recv_coord[1][row]))**2)
    d=10*math.log10(d)
    for i in range(0,len(recv_power)):
        if recv_power[row][i] != 500:
            df=df.append({'log_dist':d,'recv_pow':-recv_power[row][i]}, ignore_index=True)
            num_of_valid_Pr+=1



x=df[['log_dist']]
y=df['recv_pow']

reg=LinearRegression().fit(x,y)     #Using Linear regression to estimate the values of Eta and K
coeff=reg.coef_
intercept=reg.intercept_

eta= -coeff[0]
k=intercept+27

print("Estimated Eta: %f" %eta)
print("Estimated K:%f"%k)

Sum_of_squares_of_diff=0
i=0

#Computing estimated value of sigma using the standard deviation formula.
for dist in df['log_dist']:
    pr_prime = coeff*dist +intercept
    pr=df['recv_pow'][i]
    Sum_of_squares_of_diff+=(pr-pr_prime)**2
    i+=1

sigma= np.sqrt(Sum_of_squares_of_diff/num_of_valid_Pr)
print('Estimated Sigma:%f' %sigma)

#Plotting a estimated curve that fits the data.
xx=df['log_dist']
yy=xx*coeff + intercept

plt.scatter(x=df['log_dist'], y=df['recv_pow'], s=2)    #Creating a scatter plot using the value pairs from the dataframe df.
plt.plot(xx,yy) #Creating a line plot using the coefficients estimated from Linear Regression.
plt.title('Received power versus distance')
plt.xlabel('10log(d/d0)')
plt.ylabel('Pr-dBm')
plt.text(10, -35, 'Estimated sigma: 10.131388\nEstimated Eta: 2.941358\nEstimated K: -2.785128')

plt.show()