#画出在训练集和测试集上的损失与正则化参数lamda以及特征维度feature_size的关系
#作者：王道烩  无52  清华大学电子工程系     2015011006  13020023780   wangdh15@mails.tsinghua.edu.cn
#日期：2018年5月29日



from matplotlib import pyplot as plt 
import scipy.io as sio
import numpy as np

#从文件中读入数据
temple = sio.loadmat('./find_lamda_featuresize.mat')
lamda = temple['lamda']
feature_size = temple['feature_size']

result = temple['result_test']

#第一列为训练集上的损失  第二列为测试集上的损失 没10个数据对应一个feature_size ， 10个数据之间为不同的lamda
lost_training_feature_size = np.array([result[i*10:i*10+10,0].sum()/10 for i in range(5)])
lost_testing_feature_size = np.array([result[i*10:i*10+10 ,1].sum()/10 for i in range(5)])
lost_training_lamda = result[20:30 , 0]
lost_testing_lamda = result[20:30 , 1]

#画出图像
plt.figure(1)
plt.plot(feature_size[0] , lost_training_feature_size , 'r--' , feature_size[0], lost_testing_feature_size,'b-')
plt.xlabel('feature_size')
plt.ylabel('lost')
plt.show()
plt.figure(2)
plt.plot(lamda[0], lost_training_lamda,  'r--' , lamda[0] , lost_testing_lamda,'b-')
plt.xlabel('lamda')
plt.ylabel('lost')
plt.show()
