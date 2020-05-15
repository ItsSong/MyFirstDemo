# ------------目的：数据可视化探索-----------------

# 所需的工具包放在最前边
import pandas as pd
import seaborn as sns  # 画热力图、箱线图等(画图使用：在matplot的基础上进行了进一步封装）
import matplotlib.pyplot as plt
from scipy import stats # 假设检验要用
from scipy.stats import norm,skew
import numpy as np
from sklearn.preprocessing import LabelEncoder

# 1.先查看一下数据的格式----------------------------------------------------------------------------
trainData = pd.read_csv('train.csv')
testData = pd.read_csv('test.csv')
print("Train data:")
print(trainData.head(5))
print("Test data:")
print(testData.head(5))

# 2.查找缺失值--------------------------------------------------------------------------------------
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

# 3.查找异常值-----------------------------------------------------------------------------
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

# 4.异常值和缺失值处理-----------------------------------------------------------------------------
# 4.1 异常值
# 以居住面积为例
trainData_exceptID = trainData_exceptID.drop(trainData_exceptID[(gArea>4000) & (salePrice<300000)].index)
# 重新画一下图
plt.scatter(trainData_exceptID['GrLivArea'],trainData_exceptID['SalePrice'])
plt.xlabel('GrLivArea')
plt.ylabel('SalePrice')
plt.grid()
plt.show()
# 4.2 缺失值
# 根据上边缺失率的计算，PoolQC、MiscFeature、Alley的缺失值都在90%以上，可以考虑直接删掉这些特征
all_data = all_data.drop(['PoolQC'],axis=1)
all_data = all_data.drop(['MiscFeature'],axis=1)
all_data = all_data.drop(['Alley'],axis=1)
print(all_data.shape)
# 缺失率排名第4和第5的是Fence(栅栏)、FireplaceQu(壁炉):缺失可能代表没有，用none填充
all_data['Fence'] = all_data['Fence'].fillna('None')
all_data['FireplaceQu'] = all_data['FireplaceQu'].fillna('None')
# 缺失率排名第6的是LotFrontage(房屋前街道的长度):考虑用均值填充缺失值
all_data['LotFrontage'] = all_data.groupby('Neighborhood')["LotFrontage"].transform(lambda x: x.fillna(x.median()))
# 缺失率排名第7、8、9、10、11（缺失率完全相等）：GarageQual（车库质量）、GarageCond（车库条件）、GarageFinish（车库内部装修）、GarageType（车库位置）
# 考虑同时缺失的情况：房子应该没有车库，离散型变量用None填充缺失值
for i in ('GarageQual','GarageCond','GarageFinish','GarageType'):
    all_data[i] = all_data[i].fillna('None')
# 同理，接下来是车库相关的数值型/连续型变量，用0填充(如GarageYrBlt车库建成时间)
for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
    all_data[col] = all_data[col].fillna(0)
# 地下室相关连续变量缺失，可能因为没有地下室，用0填充
for j in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
    all_data[j] = all_data[j].fillna(0)
# 地下室相关离散变量，同理用None填充
for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    all_data[col] = all_data[col].fillna('None')
# Mas为砖石结构相关变量，缺失值我们同样认为是没有砖石结构，用0和none填补缺失值
all_data["MasVnrType"] = all_data["MasVnrType"].fillna("None")
all_data["MasVnrArea"] = all_data["MasVnrArea"].fillna(0)
# 接下来是MSZoning(标识销售区域划分类别)，考虑使用众数填补缺失值
# print(all_data['MSZoning'].mode())    # mode函数求众数，显示第一个值为众数，第二个值为众数的数据类型
all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])
# 继续往下走(I'm tired hahaha)
all_data = all_data.drop(['Utilities'],axis=1)
all_data['Functional'] = all_data['Functional'].fillna(all_data['Functional'].mode()[0])
all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])
all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])
all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])
all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])
all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])
# 现在来看一下缺失情况是否OK了
all_data_na1 = (all_data.isnull().sum()/len(all_data))*100
all_data_na1 = all_data_na1.drop(all_data_na1[all_data_na1==0].index).sort_values(ascending=False) # 剔除掉缺失率为0的特征索引
missing_data1 = pd.DataFrame({'MissingData1':all_data_na1})
print(missing_data1.head(10))

# 5.假设检验（其实也属于特征工程）-----------------------------------------------------------
# 5.1 检验数据是否符合正态分布(可视化更直观哦)
sns.distplot(trainData_exceptID['SalePrice'],fit=norm)  # seaborn中使用distplot画直方图。fit参数表示拟合分布线
(mu,sigma) = norm.fit(trainData_exceptID['SalePrice']) # 计算均值，方差
print('\n mu={:.2f} and sigma={:.2f} \n'.format(mu,sigma))
# 房价数据分布情况可视化
plt.legend(['Normal dist. ($\mu=${:.2f} and $\sigma=${:.2f}'.format(mu,sigma)],loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice Distribution')
plt.grid()
plt.show()
# probplot概率图，类似于QQ图
fig2 = plt.figure()
result = stats.probplot(trainData_exceptID['SalePrice'],plot=plt) #概率图：概率以理论分布的比例（x轴）显示，y轴包含样本数据的未缩放分位数
plt.show()
# 5.2 将非正态转化为正态分布（使用对数变换）
trainData_exceptID['SalePrice'] = np.log1p(trainData_exceptID['SalePrice'])  # 取对数：log1p = log（x+1）
# 再来看看分布情况
sns.distplot(trainData_exceptID['SalePrice'],fit=norm)
(mu,sigma) = norm.fit(trainData_exceptID['SalePrice'])
print('\n mu={:.2f} and sigma={:.2f} \n'.format(mu,sigma))
plt.legend(['Normal distri.($\mu=${:.2f} and $\sigma=${:.2f}'.format(mu,sigma)],loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice Distribution After Transform')
plt.grid()
plt.show()# 结果不错
# 再画个probplot概率图
fig3 = plt.figure()
result_afterT = stats.probplot(trainData_exceptID['SalePrice'],plot=plt)
plt.grid()
plt.show()

# 6.特征工程------------------------------------------------------------------------------
# 6.1 单个特征的处理（主要针对离散型变量）
cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond',
        'ExterQual', 'ExterCond','HeatingQC', 'KitchenQual', 'BsmtFinType1',
        'BsmtFinType2', 'Functional', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'CentralAir', 'MSSubClass', 'OverallCond',
        'YrSold', 'MoSold')            # 使用Label Encoding处理这些特征（有序的离散特征）
for k in cols:
    LE1 = LabelEncoder()
    LE1.fit(list(all_data[k].values))  # 将每一列特征的值作为一个列表塞进编码字典中进行编码
    all_data[k] = LE1.transform(list(all_data[k].values))  # 将每一列特征的值转化为编码字典中的索引，对应fit就能知道每一个值编码后的结果
print("all_data's shape:{}".format(all_data.shape))
all_data = pd.get_dummies(all_data)    # get_dummies是pandas中独热编码的方法,处理剩下的离散特征
# 独热编码会增加特征维度
print(all_data.shape)
print(all_data.head())
# 6.2 特征抽取（新增特征）
# 新增“房屋总面积列”：地下室面积 + 1楼面积 + 2楼面积 = 房屋总面积
all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']
# 建造时间比较近的房子房价比较高，所以新创造一个01特征，如果房屋建造时间在1990年后，则为1，否则是0
all_data['YearBuilt_cut'] = all_data['YearBuilt'].apply(lambda x:1 if x>1990 else 0)
print(all_data.shape)
# 6.3 特征筛选
# 为避免多重共线性，剔除掉相关系数>0.9的特征（皮尔逊相关系数）
threshold = 0.9
# 相关系数矩阵
corr_matrix = all_data.corr().abs()
print(corr_matrix.head())
# 只选择矩阵上半部分(对称)
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape),k=1).astype(np.bool))
# where()第一种用法：where(conditions, x, y)，conditions为真，返回x,否则y
# where()第二种用法：where(conditions)，直接返回conditions为真的数组下标
# np.triu()获取上三角矩阵，k表示对角线起始位置
print(upper.head())
# 删除掉相关系数>0.9的特征
to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
print('There are %s columns to remove.'%(len(to_drop)))
all_data = all_data.drop(columns=to_drop)
print(all_data.shape)
# 6.4 又将训练集和测试集分开来
trainData = all_data[:n_trainData]
testData = all_data[n_trainData:]

# 7. 建模------------------------------------------------------------------------
from sklearn.linear_model import Ridge,RidgeCV,ElasticNet,LassoCV,LassoLarsCV
from sklearn.model_selection import cross_val_score
# 7.1 k折交叉验证，验证模型准确率或泛化误差
n_folds = 5 # 设置5折交叉
def rmsle_cv(model):
    rmse = np.sqrt(-cross_val_score(model, trainData, y_train, scoring='neg_mean_squared_error', cv=n_folds)) # scoring表示评分形式，这里使用均方误差
    return rmse  # 返回的是评分
# 7.2 导入岭回归模型(调节参数)
model_ridge = Ridge()
alphas = [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75] # 参数alphas，控制正则项的强弱（防止过拟合和欠拟合）
# 每调节一次alphas，计算一次模型误差的均值（5折交叉验证）
cv_ridge = [rmsle_cv(Ridge(alpha=alpha)).mean()
            for alpha in alphas]
# 可视化看看模型误差
cv_ridge = pd.Series(cv_ridge, index=alphas)  # 将原本是列表的cv_ridge转化为数组形式展现，索引是alphas
cv_ridge.plot(title='Validation Score')
plt.xlabel("alpha")
plt.ylabel("rmse")
plt.grid()
plt.show()
print(cv_ridge)
# 7.3 拟合模型
clf = Ridge(alpha=15)
clf.fit(trainData,y_train)
# 7.4 预测
predict = clf.predict(testData)
testData1 = testData
testData1['SalePrice_Predict'] = predict
print(testData1.head())