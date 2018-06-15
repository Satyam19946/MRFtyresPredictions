import os 
os.chdir(r"C:\Users\Satyam\Desktop\Programs\Learning tensorflow\mrf")

import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
sns.set(style='darkgrid')
#Number_of_rows = int(input("How many number of rows to be displayed"))
Number_of_rows = 5

pd.options.display.max_rows = Number_of_rows
my_data = pd.read_csv('modifiedstock.csv')
#Run the following line if running for first the Year column is not there
# my_data = my_data.assign(Year = my_data.Data.apply(lambda g: '20'+g[-2:]))
indexes = list(my_data.head(0))

#Call this functions to get graphs
def visualizedata():
	#Graph between Opening Price and High Price
	sns.lmplot(x = 'Open Price', y = 'High Price', data = my_data,fit_reg = False,hue = 'Year')
	plt.title("Open Price vs High Price")
	plt.show()
	plt.close('all')

	
	sns.lmplot(x = 'Open Price', y = 'Close Price', data = my_data,fit_reg = False,hue = 'Year')
	plt.title("Open Price vs Close Price")
	plt.show()
	plt.close('all')

	sns.lmplot(x = 'Open Price', y = 'Low Price', data = my_data,fit_reg = False,hue = 'Year')
	plt.title("Open Prive vs Low Price")
	plt.show()
	plt.close('all')

OpenPrice,HighPrice,LowPrice,ClosePrice = pd.DataFrame(my_data['Open Price']),my_data['High Price'],my_data['Low Price'],my_data['Close Price']
#Number of features
n = 1
#Number of samples = len(Open Price)
m = len(OpenPrice)



#Model
def Trainthemodel(x_data,y_data,numberofsteps):
	x = tf.placeholder(tf.float64,shape=[None,n])
	y_true = tf.placeholder(tf.float64,shape = None)

	with tf.name_scope('inference') as scope:
		w = tf.Variable([[1.2]],dtype = tf.float64,name = 'weights')
		bias = tf.Variable(0,dtype = tf.float64,name='bias')
		y_pred = tf.matmul(w,tf.transpose(x)) + bias
	
	with tf.name_scope('loss') as scope:
		loss = tf.reduce_mean(tf.square(y_true-y_pred))
		
	with tf.name_scope('train') as scope:
		learning_rate = 0.00000000001
		optimizer = tf.train.GradientDescentOptimizer(learning_rate)
		train = optimizer.minimize(loss)
	
	
	#This many steps will be taken by the linear regression.
	init = tf.global_variables_initializer()
	with tf.Session() as sess:
		sess.run(init)
		for step in range(numberofsteps):
			sess.run(train,{x:x_data,y_true:y_data})
		return sess.run(bias),sess.run(w)	


op = np.array(OpenPrice)
bh,wh = Trainthemodel(np.array(OpenPrice),np.array(HighPrice),1000)
bl,wl = Trainthemodel(np.array(OpenPrice),np.array(LowPrice),1000)
bc,wc = Trainthemodel(np.array(OpenPrice),np.array(ClosePrice),1000)

def PredictHigh(P):
	return P*wh[0][0] + bh
	
def PredictLow(P):
	return P*wl[0][0] + bl 

	
def PredictClose(P):
	return P*wc[0][0] + bc		

def Accuracy(truearray,predarray):
	return (1 - sum((predarray - truearray)**2)/sum((truearray - truearray.mean())**2))*100
	
def visualizepredictions():
	hp = np.array(HighPrice)
	cp = np.array(ClosePrice)
	lp = np.array(LowPrice)
	#Predicting high price;
	predh = hp*wh[0] + bh
	predc = cp*wc[0] + bc
	predl = lp*wl[0] + bl
	plt.scatter(x = range(m), y = HighPrice, s =10,c = 'blue', alpha = 0.5)
	plt.scatter(x = range(m), y = predh,s = 10, c = 'red', alpha = 0.5)
	plt.legend(["Actual High Price","High Price Predicted on Open Price"])
	plt.title("Accuracy of the model is {0:.2f}".format(Accuracy(hp,predh)))
	plt.show()
	
	plt.scatter(x = range(m), y = ClosePrice, s = 10, c ='blue', alpha = 0.5)
	plt.scatter(x = range(m), y = (cp*wc[0]+bc), s =10 , c = 'red', alpha = 0.5)
	plt.legend(['Actual Close Price','Close Price Predicted on Open Price'])
	plt.title("Accuracy of the model is {0:.2f}".format(Accuracy(cp,predc)))
	plt.show()
	
	plt.scatter(x = range(m), y = LowPrice, s = 10, c = 'blue', alpha = 0.5)
	plt.scatter(x = range(m), y = (lp*wl[0]+bl), s = 10, c = 'red', alpha = 0.5)
	plt.legend(['Actual Low Price','Low Price Predicted on Open Price'])
	plt.title("Accuracy of the model is {0:.2f}".format(Accuracy(lp,predl)))
	plt.show()
	
def Main():
	while True:
		print("Enter The Open Price to Get todays High Price, Low Price, Close Price")
		P = float(input())
		print("High Price ", PredictHigh(P))
		print("Low Price ", PredictLow(P))
		print("Close Price ", PredictClose(P))
		print("Run Again?(Y/y)")
		ch = input()
		if ch != 'Y' and ch != 'y':
			break	

Main() 