#使用交替最小二乘法填充矩阵
#作者：王道烩  无52   清华大学电子工程系   2015011006    13020023780     wangdh15@mails.tsinghua.edu.cn
#日期：2018年5月27日


import numpy as np
import scipy.io as sio
import scipy
import datetime

#定义计算损失的函数，传入的参数为标记矩阵W ， 数据矩阵M ， U ， V 以及数据数量data_num，返回均方误差损失 
def compute_lost(  W , M , U ,V , data_num):
	temple = U.dot(V.T)
	lost = scipy.linalg.misc.norm(temple*W - M)**2/data_num
	return lost



#算法主体，传入的参数为特征维度feature_size ， 正则化参数lamda， 训练集标记矩阵W ， 训练集数据M ， 测试集标记矩阵W_test , 测试集数据集M_test
def als(feature_size , lamda , W , M , W_test , M_test):
	#随机初始化U ， V元素为0-1之间的随机数
	U = np.random.rand(943 , feature_size)
	V = np.random.rand(1682 ,feature_size)

	#初始化前一次训练集上的损失和迭代次数
	lost_training_last = 0
	iteration_num = 0

	#进行迭代
	while True:
		#迭代次数加一
		iteration_num += 1

		# 更新U 使用理论计算的结果
		for i in range(943):
			temple = np.linalg.inv(np.eye(feature_size) * lamda + 2 * (W[i , :] * V.T).dot(V)) 
			U[i , :] = 2 * ((W[i , :] * M[i , :]).dot(V)).dot(temple)
		# 更新V 使用理论计算的结果
		for i in range(1682):
			temple = np.linalg.inv(np.eye(feature_size) * lamda +2* ((W.T)[i , :] * U.T).dot(U))
			V[i , :] =2 * (((W.T)[i ,:] * (M.T)[i ,:]).dot(U)).dot(temple)

		#计算更新完U和V之后的在训练集上的新的损失
		lost_training_now = compute_lost( W , M , U  ,V , 80000)

		#对比此次损失和上次迭代的损失，如果小于某个阈值，则停止迭代，否则将此次训练集损失赋予lost_training_last
		if lost_training_last - lost_training_now < 1.0e-5 and lost_training_last - lost_training_now > -1.0e-5:
			break
		else:
			lost_training_last = lost_training_now


	#迭代结束计算在测试集上的损失
	lost_testing = compute_lost(W_test , M_test , U ,V , 10000)

	#将在训练集上的损失和测试集上的损失以及迭代次数,举证U V返回
	return lost_training_now ,  lost_testing , iteration_num ,U , V 



#######################################################
#调用上面的函数来进行计算
#在开始和结束分别设置一个计时器记录当前时间来计算算法执行的时间
#######################################################

#设置开始的计时器
starttime = datetime.datetime.now()

#从自己挑选的训练集和测试集文件中读取数据
temple = sio.loadmat('./data_set_my_best.mat')

#将训练集和测试及分别存在变量training_data 以及testing_data中
training_data = temple['training_data']
testing_data  = temple['testing_data']

#初始化W W_test M M_test
W = np.zeros((943 , 1682) , dtype = np.int16)
M = np.zeros((943 , 1682) , dtype = np.float64)
W_test = np.zeros((943 , 1682) , dtype = np.int16)
M_test = np.zeros((943 , 1682) , dtype = np.float64)

#对training_data进行遍历以填充W M
for each in  training_data:
	W[int(each[0]-1) , int(each[1]-1) ] = 1
	M[int(each[0]-1) , int(each[1]-1) ] = each[2]

#对testing_data进行遍历以填充W_test   M_test
for each in testing_data:
	W_test[int(each[0]-1) , int(each[1]-1) ] = 1
	M_test[int(each[0]-1) , int(each[1]-1) ] = each[2]

#选择正则化参数lamda  特征维度feature_size
lamda = 15
feature_size = 3

#调用函数als
lost_training , lost_testing , iteration_num , U , V = als(feature_size , lamda , W , M , W_test , M_test)

#设置结束定时器
endtime = datetime.datetime.now()

#计算总运行时间
run_time = (endtime-starttime).seconds


#打印出信息：训练集上的损失，测试集上的损失，迭代次数以及运行时间
print('the lost on training_data: %f'%lost_training)
print('the lost on testing_data: %f'%lost_testing)
print('the number of iteration : %d'%iteration_num)
print('the time run : %d seconds'%run_time)

#将求的矩阵U  V  以及结果U.dot(V.T)保存在文件result.mat中
sio.savemat('./result.mat' , {'U':U , 'V':V,  'result':U.dot(V.T)})
