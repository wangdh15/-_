#从数据集中挑选训练集和测试集 一共使用了三种方法(1)随机法  (2)尽量均匀选择测试集  (3)尽量均匀选择训练集
#作者:王道烩    无52  清华大学电子工程系    2015011006    13020023780    wangdh15@mails.tsinghua.edu.cn
#日期：2018年5月30日
import scipy.io as sio
import numpy as np 
import scipy
import random

#随机选择训练集以及测试集
def random_select():
	
	#读入数据
	temple = sio.loadmat('./data_train.mat')
	data_train = temple['data_train']

	#初始化training_data以及testing_data为空
	testing_data = []
	training_data = []

	#在90000个数据中随机选择10000个数据作为测试集
	index = list(range(90000))
	index_testing_data = random.sample(index , 10000)
	for each in index:
		if each in index_testing_data:
			testing_data.append(data_train[each])
		else:
			training_data.append(data_train[each])

	#转换为numpy的array类型
	training_data = np.array(training_data)
	testing_data = np.array(testing_data)

	#将挑选的数据保存在data_set_my.mat中，供ALS.py使用
	sio.savemat('./data_set_my.mat' , {'training_data':training_data , 'testing_data':testing_data})




#均匀选择训练集
def precious_select_testing_data():

	#读入数据
	temple = sio.loadmat('./data_train.mat')
	data_train = temple['data_train']

	#初始化训练集和测试集
	testing_data = []
	training_data = []

	#记录每一行有多少数据
	index_num = np.zeros(943 , dtype = np.int32)
	for each in data_train:
		index_num[int(each[0]-1)] += 1

	#从每一行中交替随机挑选10或11个数据作为测试集
	for i in range(943):
		num_flag = 0
		small_index = index_num[0:i].sum()
		big_index = small_index + index_num[i]
		if i % 2 == 0 and num_flag <= 373:
			select_num = 10
			num_flag += 1
		else:
			select_num = 11
		index_testing_data  = random.sample(list(range(int(index_num[i]))) , select_num)
		data_temple = data_train[small_index : big_index]
		for j in range(int(index_num[i])):
			if j in index_testing_data:
				testing_data.append(data_temple[j])
			else:
				training_data.append(data_temple[j])
	
	#将数据保存在文件中
	training_data = np.array(training_data)
	testing_data = np.array(testing_data)
	sio.savemat('./data_set_my.mat' , {'training_data':training_data , 'testing_data':testing_data})



#均匀选择训练集
def precious_select_training_data():

	#读入数据
	temple = sio.loadmat('./data_train.mat')
	data_train = temple['data_train']

	#初始化
	testing_data = []
	training_data = []

	#确定每一行和每一列至少要包含数数据数目来保证选择的训练集尽量分布均匀
	row_minim = 35
	column_minim =35 

	#确定每一行的数据数目
	index_num_row = np.zeros(943 , dtype = np.int32)
	for each in data_train:
		index_num_row[int(each[0])-1] += 1

	#定义data_remain用来保存第一次循环没有进入训练集的数据
	data_remain = []

	#先从每一行进行挑选，如果这一行的数据量比row_minim小，就全部加入训练集，否则随机算这row_minim个数据添加到训练集中，并将剩下的数据保存在data_remain中
	for i in range(943):
		small_index = index_num_row[0:i].sum()
		big_index = small_index + index_num_row[i]
		if index_num_row[i] <= row_minim:
			data_temple = data_train[small_index : big_index]
			for each in data_temple:
				training_data.append(each)
		else:
			#random select row_minim item
			index_training_data  = random.sample(list(range(int(index_num_row[i]))) , row_minim)
			data_temple = data_train[small_index : big_index]
			for j in range(int(index_num_row[i])):
				if j in index_training_data:
					training_data.append(data_temple[j])
				else:
					data_remain.append(data_temple[j])


	#将剩下的顺寻按照列排序
	data_remain = np.array(data_remain)
	a_arg = np.argsort(data_remain[:,1])
	data_remain = data_remain[a_arg]

	#计算此时训练集中每一列的数量
	index_num_column = np.zeros(1682 , dtype = np.int32)
	for each in training_data:
		index_num_column[int(each[1])-1] += 1

	#计算此时剩下的数据中每一列的数据个数
	index_num_column_remain  = np.zeros(1682 , dtype = np.int32)
	for each in data_remain:
		index_num_column_remain[int(each[1])-1] += 1

	#定义data——remain_second来保存第二次循环后没有在训练集中的数据
	data_remain_second = []

	#开始循环，如果训练集中某一列的数据数目小于column_minim,那么就在剩下的数据添加到训练集中，并将多的数据保存在data_remain_second中
	for i in range(1682):
		if index_num_column[i] < column_minim:
			if index_num_column[i] + index_num_column_remain[i] <= column_minim:
				small_index = index_num_column_remain[0:i].sum()
				big_index = small_index + index_num_column_remain[i]
				data_temple = data_remain[small_index : big_index]
				for each in data_temple:
					training_data.append(each)
			else:
				small_index = index_num_column_remain[0:i].sum()
				big_index = small_index + index_num_column_remain[i]
				data_temple = data_remain[small_index : big_index]
				index_training_data  = random.sample(list(range(int(index_num_column_remain[i]))) , column_minim - index_num_column[i])
				for j in range(int(index_num_column_remain[i])):
					if j in index_training_data:
						training_data.append(data_temple[j])
					else:
						data_remain_second.append(data_temple[j])
		else:
			small_index = index_num_column_remain[0:i].sum()
			big_index = small_index + index_num_column_remain[i]
			data_temple = data_remain[small_index : big_index]
			for each in data_temple:
				data_remain_second.append(each)

	#在第二次循环后剩下的数据中随机挑选10000个数据作为测试集，并将其余的添加到训练集中
	index_testing_data = random.sample(list(range(len(data_remain_second))) , 10000)
	for i in range(len(data_remain_second)):
		if i in index_testing_data:
			testing_data.append(data_remain_second[i])
		else:
			training_data.append(data_remain_second[i])

	#将数据保存在文件中
	training_data = np.array(training_data)
	testing_data = np.array(testing_data)
	sio.savemat('./data_set_my.mat' , {'training_data':training_data , 'testing_data':testing_data})




#通过添加取消注释来选择挑选数据的方式，经测试, 均匀选择训练集>随机法>均匀选择测试集
#random_select()
#precious_select_testing_data()
precious_select_training_data()