# ------------目的：数据可视化探索-----------------

# 所需的工具包放在最前边
import pandas as pd
import seaborn as sns  # 画热力图要用
import matplotlib.pyplot as plt

# 1.先查看一下数据的格式
trainData = pd.read_csv('train.csv')
testData = pd.read_csv('test.csv')
print("Train data:")
print(trainData.head(5))
print("Test data:")
print(testData.head(5))

# 2.查找缺失值
# 2.1 合并训练集和测试集一块处理数据，减少工作量
n_trainData = trainData.shape[0]  # 记录下训练集的行数
n_testData = testData.shape[0]    # 记录下测试集的行数
y_train = trainData.SalePrice.values  # 训练集最后一列（房价）
all_data = pd.concat([trainData,testData])  # axis默认为0，表示按照行的维度拼接，列数不变，行数相加
# drop函数：剔除表中的列或者行。axis默认0表示删除行；inplace=true表示使用剔除掉列之后的数据替换原表，默认为false
all_data.drop(['SalePrice','Id'],axis=1,inplace=True) # 剔除掉Id列和房价列
print("all_data's shape：{}".format(all_data.shape))
# 2.2 计算缺失率
all_data_na = (all_data.isnull().sum()/len(all_data))*100
all_data_na = all_data_na.drop(all_data_na[all_data_na==0].index).sort_values(ascending=False) # 剔除掉缺失率为0的特征索引
missing_data = pd.DataFrame({'MissingData':all_data_na})
print(missing_data.head(10))
# 2.3 可视化缺失情况
x = all_data_na.index
y = all_data_na
plt.xticks(rotation=90)
plt.bar(x,y)
plt.xlabel('missing features')
plt.ylabel('percent of missing values')
plt.title('Percent missing data')
plt.grid()
plt.show()

# 3.查找异常值
# 3.1 探索变量间的相关性（以热力图的形式展现）
trainData_exceptID = trainData.drop(['Id'],axis=1)
corr = trainData_exceptID.corr() # 计算变量间的相关系数
fig = plt.figure(figsize=(12,9))
sns.heatmap(corr,square=True)  # square参数设置图像是否为正方形；linewidths可以设置矩阵间隔
plt.show()
fig.savefig('Correlation_Coefficient.jpg')
# 3.2 可视化
# 3.2.1 SalePrice随OverallQual变化（箱线图更合适，这里先用一下散点图）
overallQual = trainData_exceptID['OverallQual']
salePrice = trainData_exceptID['SalePrice']
plt.scatter(overallQual,salePrice)
plt.xlabel('OverallQual')
plt.ylabel('SalePrice')
plt.grid()
plt.show()
# 3.2.2 SalePrice随GrLivArea（居住面积）变化的散点图
gArea = trainData_exceptID['GrLivArea']
plt.scatter(gArea,salePrice)
plt.xlabel('GrLivArea')
plt.ylabel('SalePrice')
plt.grid()
plt.show()
# 3.2.3 SalePrice随GarageArea（车库面积）变化的散点图
ggArea = trainData_exceptID['GarageArea']
plt.scatter(ggArea,salePrice)
plt.xlabel('GarageArea')
plt.ylabel('SalePrice')
plt.grid()
plt.show()
# 3.2.4 SalePrice随OverallQual变化，每一个特征按照箱线图展示
price_qual = pd.concat([trainData_exceptID['SalePrice'],trainData_exceptID['OverallQual']],axis=1) # axis=1按照列维度拼接，行数不变
sns.boxplot(x='OverallQual',y='SalePrice',data=price_qual)
plt.grid()
plt.show()


