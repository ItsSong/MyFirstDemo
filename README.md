# MyFirstDemo
## 数据分析入门项目
***
 - **一、数据可视化探索**

1. 探索数据质量（缺失值、异常值等）
2. 探索特征与预测变量之间的相关性

### （1）先来导入数据，看一下数据的格式
```python
# 所需的工具包放在最前边
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm,skew
import numpy as np
from sklearn.preprocessing import LabelEncoder

# 1.先查看一下数据的格式
trainData = pd.read_csv('train.csv')
testData = pd.read_csv('test.csv')
print("Train data:")
print(trainData.head(5))
print("Test data:")
print(testData.head(5))
```
结果：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200514102943560.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0Ffemh1b18=,size_16,color_FFFFFF,t_70)

可以看到训练集：5行81列；测试集5行80列。训练集最后一列为房价。
***
### （2）探索缺失值
```python
# 2.查找缺失值
# 2.1 合并训练集和测试集一块处理数据，减少工作量
n_trainData = trainData.shape[0]  # 记录下训练集的行数
n_testData = testData.shape[0]    # 记录下测试集的行数
y_train = trainData.SalePrice.values  # 训练集最后一列（房价）
all_data = pd.concat([trainData,testData])  
# drop函数：剔除表中的列或者行。axis默认0表示删除行；inplace=true表示使用剔除掉列之后的数据替换原表，默认为false
all_data.drop(['SalePrice','Id'],axis=1,inplace=True) # 剔除掉Id列和房价列
print("all_data's shape：{}".format(all_data.shape))
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200514112836221.png)

```python
# 2.2 计算缺失率
all_data_na = (all_data.isnull().sum()/len(all_data))*100
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200514120746693.png)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200514120840726.png)
![在这里插入图片描述](https://img-blog.csdnimg.cn/2020051412092014.png)
![](https://img-blog.csdnimg.cn/20200514120948603.png)

```python
all_data_na = all_data_na.drop(all_data_na[all_data_na==0].index).sort_values(ascending=False) # 剔除掉缺失率为0的特征索引
missing_data = pd.DataFrame({'MissingData':all_data_na})
print(missing_data.head(10))
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200514121828606.png)

```python
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
```

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200514123534339.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0Ffemh1b18=,size_16,color_FFFFFF,t_70)

可以看出PoolQC、MiscFeature和Alley三个特征值缺失率超过80%，后续考虑可能需要剔除掉这三个特征。其余缺失值的处理后续再做。
***
### （3）探索异常值
    一般情况下，可以采用描述性统计查看是否存在异常值。第二，对于连续型变量可以采用散点图，查看特征与预测变量间的相关性的同时，发现异常值的存在；对于离散变量，可以采用画箱线图的方式，查看其特征分布，根据箱线图判断异常值。第三，可以采用3倍的标准差原则进行判断。
    这里，我们先查看特征与目标变量间的相关性，选择相关性大的特征探索异常值，这样才具有研究的价值吧（我是这样考虑的哈）。
```python
# 3.查找异常值
# 3.1 探索变量间的相关性（以热力图的形式展现）
trainData_exceptID = trainData.drop(['Id'],axis=1)# 剔除ID列
corr = trainData_exceptID.corr()# 计算变量间的相关系数
plt.figure(figsize=(12,9))
sns.heatmap(corr,square=True) # square设置图是否为正方形
plt.show()
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200514155610409.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0Ffemh1b18=,size_16,color_FFFFFF,t_70)

根据热力图可以看出：OverallQual（综合质量）、GrLivArea（居住面积）、GarageCars（车库能放多少量车）、GarageArea（车库面积）和SalePrice的相关性最强。其中，GarageCars（车库能放多少量车）和GarageArea（车库面积）两者之间相关性很强，考虑到多重共线性问题，因此可以去掉其中一个变量。
```python
# 3.2 可视化
# 3.2.1 SalePrice随OverallQual变化（箱线图更合适，先用一下散点图看看）
overallQual = trainData_exceptID['OverallQual']
salePrice = trainData_exceptID['SalePrice']
plt.scatter(overallQual,salePrice)
plt.xlabel('OverallQual')
plt.ylabel('SalePrice')
plt.grid()
plt.show()
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200514161722307.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0Ffemh1b18=,size_16,color_FFFFFF,t_70)
    可以用箱线图展示（见下文）
```python
# 3.2.2 SalePrice随GrLivArea（居住面积）变化的散点图
gArea = trainData_exceptID['GrLivArea']
plt.scatter(gArea,salePrice)
plt.xlabel('GrLivArea')
plt.ylabel('SalePrice')
plt.grid()
plt.show()
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200514162104816.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0Ffemh1b18=,size_16,color_FFFFFF,t_70)
    
随着居住面积越大，售价越高，很明显，图中存在两个异常点。
我们再看一个GarageArea（车库面积）和房价间关系：
```python
# 3.2.3 SalePrice随GarageArea（车库面积）变化的散点图
ggArea = trainData_exceptID['GarageArea']
plt.scatter(ggArea,salePrice)
plt.xlabel('GarageArea')
plt.ylabel('SalePrice')
plt.grid()
plt.show()
```
 ![在这里插入图片描述](https://img-blog.csdnimg.cn/20200514162456909.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0Ffemh1b18=,size_16,color_FFFFFF,t_70)
 
随着车库面积的增大，房价越高。同时似乎也存在异常值。   
接下来画离散变量的箱线图：
 

```python
# 3.2.4 SalePrice随OverallQual变化，每一个特征按照箱线图展示
price_qual = pd.concat([trainData_exceptID['SalePrice'],trainData_exceptID['OverallQual']],axis=1) # axis=1按照列维度拼接，行数不变
sns.boxplot(x='OverallQual',y='SalePrice',data=price_qual)
plt.grid()
plt.show()
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/202005141801289.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0Ffemh1b18=,size_16,color_FFFFFF,t_70)

同散点图趋势一样，随着综合质量的上升，房价也随之上升。
***
### （4）数据处理（异常值、缺失值）

```python
# 4.数据处理
# 4.1 居住面积存在异常值
trainData_exceptID = trainData_exceptID.drop(trainData_exceptID[(gArea>4000) & (salePrice<300000)].index)
# 重新画一下图
plt.scatter(trainData_exceptID['GrLivArea'],trainData_exceptID['SalePrice'])
plt.xlabel('GrLivArea')
plt.ylabel('SalePrice')
plt.grid()
plt.show()
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/2020051422201530.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0Ffemh1b18=,size_16,color_FFFFFF,t_70)

异常值已经剔除了。对于车库面积也可进行同样的操作。接下来对缺失值进行操作。
```python
# 4.2 缺失值
# 根据上边缺失率的计算，PoolQC、MiscFeature、Alley的缺失值都在90%以上，可以考虑直接删掉这些特征
all_data = all_data.drop(['PoolQC'],axis=1)
all_data = all_data.drop(['MiscFeature'],axis=1)
all_data = all_data.drop(['Alley'],axis=1)
print(all_data.shape)
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200514223106165.png)

```python
# 缺失率排名第4和第5的是Fence(栅栏)、FireplaceQu(壁炉):缺失可能代表没有，用none填充
all_data['Fence'] = all_data['Fence'].fillna('None')
all_data['FireplaceQu'] = all_data['FireplaceQu'].fillna('None')
```

```python
# 缺失率排名第6的是LotFrontage(房屋前街道的长度):考虑用均值填充缺失值
all_data['LotFrontage'] = all_data.groupby('Neighborhood')["LotFrontage"].transform(lambda x: x.fillna(x.median()))

```

```python
# 缺失率排名第7、8、9、10、11（缺失率完全相等）：GarageQual（车库质量）、GarageCond（车库条件）、GarageFinish（车库内部装修）、GarageType（车库位置）
# 考虑同时缺失的情况：房子应该没有车库，用None填充缺失值
for i in ('GarageQual','GarageCond','GarageFinish','GarageType'):
    all_data[i] = all_data[i].fillna('None')
```
```python
# 同理，接下来是车库相关的数值型/连续型变量，用0填充(如GarageYrBlt车库建成时间)
for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
    all_data[col] = all_data[col].fillna(0)
```

```python
# 地下室相关变量缺失率相同，可能因为没有地下室，用0填充
for j in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
    all_data[j] = all_data[j].fillna(0)
```

```python
# 地下室相关离散变量，同理用None填充
for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    all_data[col] = all_data[col].fillna('None')
```

```python
# Mas为砖石结构相关变量，缺失值我们同样认为是没有砖石结构，用0和none填补缺失值
all_data["MasVnrType"] = all_data["MasVnrType"].fillna("None")
all_data["MasVnrArea"] = all_data["MasVnrArea"].fillna(0)
```

```python
# 接下来是MSZoning(标识销售区域划分类别)，考虑使用众数填补缺失值
# print(all_data['MSZoning'].mode())    # mode函数求众数，显示第一个值为众数，第二个值为众数的数据类型
all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])
```

```python
# 继续往下走(I'm tired hahaha)
all_data = all_data.drop(['Utilities'],axis=1)
all_data['Functional'] = all_data['Functional'].fillna(all_data['Functional'].mode()[0])
all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])
all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])
all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])
all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])
all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])
```

现在来看看缺失情况是否好转了：<br>
```python
all_data_na1 = (all_data.isnull().sum()/len(all_data))*100
all_data_na1 = all_data_na1.drop(all_data_na1[all_data_na1==0].index).sort_values(ascending=False) # 剔除掉缺失率为0的特征索引
missing_data1 = pd.DataFrame({'MissingData1':all_data_na1})
print(missing_data1.head(10))
```

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200515112232841.png)

OK可以了！！！
***
 - **二、回归分析**
 ### 1.回归分析的假设
 （1）自变量X与因变量Y间关系：线性、可加性（Y=b+a1X1+a2X2+...+ϵ）<br> 
 线性：X每变动一个单位，Y相应的发生固定单位的变动，与X的绝对数值无关<br>
 可加性：X对Y的影响是独立的，如各个自变量如x1、x2...，x1对Y的影响独立于x2<br>
 线性与可加性：如果将线性模型拟合到非线性、非叠加的数据集上，回归算法将无法从数学上捕获数据集中的趋势，从而导致模型无效。在未知的数据集上进行预测，也将导致预测错误。<br>
 
 （2）各自变量间不相关，否则为多重性线性。如果出现多重共线性，那我们就很难得知自变量与因变量之间真正的关系了。当多重共线性性出现的时候，变量之间的联动关系会导致测得的标准差偏大，置信区间变宽。采用岭回归，Lasso回归可以一定程度上减少方差，解决多重共线性性问题。因为这些方法，在最小二乘法的基础上，加入了一个与回归系数的模有关的惩罚项，可以收缩模型的系数。<br>  
 
 （3）残差服从正态分布；残差项之间不相关（否则称之为自相关）；且方差恒定，即同方差性（否则称之为异方差性）
 
 检验方法：<br>
 
 （1）线性&可加性检验：观察残差（Residual）-估计值（Fitted Value，Y^）图<br>
 如果图中有任何模式，比如可能出现抛物线形状，需要考虑数据中的非线性迹象。这表明模型无法捕捉到非线性效应。解决方法：可以做一个非线性变换例如log(x)，√X 或X²变换<br>
 如果图中呈现漏斗形状，则是非常量方差的迹象，即异方差性。解决方法：变换响应变量，例如log(Y)或√Y。此外，还可以采用加权最小二乘法解决异方差性问题。<br>
 
 （2）自相关性检验：计算杜宾-瓦特森统计量（Durbin-Watson Statistic）<br>
 ![在这里插入图片描述](https://img-blog.csdnimg.cn/20200515121911787.png)<br>
 DW=2：没有自相关性；0<DW<2：残差间有正的相关性；2<DW<4：残差间有负的相关性；<br>
 
 （3）多重共线性检验：首先，可以通过观察自变量的散点图（Scatter Plot）来进行初步判断。然后，针对可能存在多重共线性性的变量，我们观察其方差膨胀系数（VIF–Variance Inflation Factor）。若VIF<3，说明该变量基本不存在多重共线性性问题，若VIF>10，说明问题比较严重。<br>
 
 （4）正态分布检验：正态QQ图（Quantile-Quantile Plot，即分位数-分位数图，本质是散点图，可以理解为样本和理论分位数的差异）。如果服从正态分布，则Q-Q图呈现出一条直线。若直线出现偏差时，误差不服从正态分布。如果误差不是正态分布的，那么变量（响应或预测变量）的非线性变换可以改善模型。<br>
 通常，概率图也可以用于确定一组数据是否服从任一已知分布，如二项分布或泊松分布。概率图展示的是样本的累积频率分布与理论正态分布的累积概率分布之间的关系。如果图中各点为直线或接近直线，则样本的正态分布假设可以接受。<br>
 同理,任意两个数据集都可以通过比较来判断是否服从同一分布。计算每个分布的分位数。一个数据集对应x轴，另一个对应y轴。做一条45度参考线，如果两个数据集数据来自同一分布，则这些点会落在参照线附近。<br>
 
 （5）异方差性检验：法1：比例位置图（残差的标准差/估计值图(Scale Location Plot)）。显示了残差如何沿着预测变量的范围传播。法2：残差（Residual）/估计值（Fitted Value，Y^）图。若该图呈现如上图所示的“漏斗形”，即随着Y^的变化，残差有规律的变大或变小，则说明存在明显的异方差性。<br>
 
 ### 2.检验
 先来检验一下房价是否符合正态分布：可视化更直观哦<br>
```python
# 5.回归分析
# 5.1 检验数据是否符合正态分布(可视化更直观哦)
sns.distplot(trainData_exceptID['SalePrice'],fit=norm)  # 拟合正态分布曲线
(mu,sigma) = norm.fit(trainData_exceptID['SalePrice']) # 计算均值，方差
print('\n mu={:.2f} and sigma={:.2f} \n'.format(mu,sigma))
```
结果：mu=180932.92 and sigma=79467.79 <br>
```python
plt.legend(['Normal dist. ($\mu=${:.2f} and $\sigma=${:.2f}'.format(mu,sigma)],loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice Distribution')
plt.grid()
plt.show()
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200515151926576.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0Ffemh1b18=,size_16,color_FFFFFF,t_70)

```python
fig2 = plt.figure()
result = stats.probplot(trainData_exceptID['SalePrice'],plot=plt) #概率图：概率以理论分布的比例（x轴）显示，y轴包含样本数据的未缩放分位数
plt.show()
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200515151952912.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0Ffemh1b18=,size_16,color_FFFFFF,t_70)

可以看出房价数据呈现右偏分布，需要进行一定的操作转化为正态分布（对数、倒数、平方根、平方根反正弦、Box-Cox等）。这里使用对数变换：
```python
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
plt.show()
```
 mu=12.02 and sigma=0.40<br>
 ![在这里插入图片描述](https://img-blog.csdnimg.cn/20200515162128766.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0Ffemh1b18=,size_16,color_FFFFFF,t_70)
 ```python
 # 再画个probplot概率图
fig3 = plt.figure()
result_afterT = stats.probplot(trainData_exceptID['SalePrice'],plot=plt)
plt.grid()
plt.show()
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200515162435158.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0Ffemh1b18=,size_16,color_FFFFFF,t_70)

看起来已经可以了哈，转化成功！
 ***
 ### 3.特征工程 
#### （1）先对离散型变量进行编码处理:Label Encoding、One-Hot Encoding
```python
# 6.特征工程
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
```
结果：all_data's shape:(2919, 75)<br>
之前删除了5列：Id、SalePrice、Utilities、PoolQC、MiscFeature、Alley
```python
all_data = pd.get_dummies(all_data)    # get_dummies是pandas中独热编码的方法,处理剩下的离散特征
# 独热编码会增加特征维度
print(all_data.shape)
print(all_data.head())
```
结果：(2919, 320)

 1stFlrSF  2ndFlrSF     ...       SaleType_Oth  SaleType_WD
0       856       854     ...                  0            1
1      1262         0     ...                  0            1
2       920       866     ...                  0            1
3       961       756     ...                  0            1
4      1145      1053     ...                  0            1
[5 rows x 320 columns]

暂且看不全，截点图来看看（确实全部变成数字了）：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200515194852725.png)
![在这里插入图片描述](https://img-blog.csdnimg.cn/202005151949134.png)

#### （2）特征抽取
```python
# 6.2 特征抽取（新增特征）
# 新增“房屋总面积列”：地下室面积 + 1楼面积 + 2楼面积 = 房屋总面积
all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']
# 建造时间比较近的房子房价比较高，所以新创造一个01特征，如果房屋建造时间在1990年后，则为1，否则是0
all_data['YearBuilt_cut'] = all_data['YearBuilt'].apply(lambda x:1 if x>1990 else 0)
print(all_data.shape)
```
结果：(2919, 322)<br>
看一下截图，确实增加了两列哈：<br>
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200515195903190.png)

#### （3）特征筛选
```python
# 6.3 特征筛选
# 为避免多重共线性，剔除掉相关系数>0.9的特征（皮尔逊相关系数）
threshold = 0.9
# 相关系数矩阵
corr_matrix = all_data.corr().abs()
print(corr_matrix.head())
```
结果：<br>
              1stFlrSF  2ndFlrSF      ...         TotalSF  YearBuilt_cut
1stFlrSF      1.000000  0.249823      ...        0.793379       0.237462
2ndFlrSF      0.249823  1.000000      ...        0.298512       0.203171
3SsnPorch     0.044086  0.032458      ...        0.024988       0.010986
BedroomAbvGr  0.108418  0.503506      ...        0.350625       0.039398
BsmtCond      0.040297  0.016495      ...        0.106404       0.116658
[5 rows x 322 columns]

```python
# 只选择矩阵上半部分(对称)
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape),k=1).astype(np.bool))
# where()第一种用法：where(conditions, x, y)，conditions为真，返回x,否则y
# where()第二种用法：where(conditions)，直接返回conditions为真的数组下标
# np.triu()获取上三角矩阵，k表示对角线起始位置
print(upper.head())
```
结果：<br>
              1stFlrSF  2ndFlrSF      ...         TotalSF  YearBuilt_cut
1stFlrSF           NaN  0.249823      ...        0.793379       0.237462
2ndFlrSF           NaN       NaN      ...        0.298512       0.203171
3SsnPorch          NaN       NaN      ...        0.024988       0.010986
BedroomAbvGr       NaN       NaN      ...        0.350625       0.039398
BsmtCond           NaN       NaN      ...        0.106404       0.116658
[5 rows x 322 columns]

```python
# 删除掉相关系数>0.9的特征
to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
print('There are %s columns to remove.'%(len(to_drop)))
all_data = all_data.drop(columns=to_drop)
print(all_data.shape)
```
结果：<br>
There are 6 columns to remove.<br>
(2919, 316)

到这里就处理完了！

之前将训练集和测试集合并到一块进行处理，现在将其分开：<br>
```python
# 6.4 又将训练集和测试集分开来
trainData = all_data[:n_trainData]
testData = all_data[n_trainData:]
```
OK，可以进行建模了！！！
 
