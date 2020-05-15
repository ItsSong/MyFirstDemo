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
 ![在这里插入图片描述](https://img-blog.csdnimg.cn/20200515121911787.png)
 DW=2：没有自相关性；0<DW<2：残差间有正的相关性；2<DW<4：残差间有负的相关性；<br>
 （3）多重共线性检验：首先，可以通过观察自变量的散点图（Scatter Plot）来进行初步判断。然后，针对可能存在多重共线性性的变量，我们观察其方差膨胀系数（VIF–Variance Inflation Factor）。若VIF<3，说明该变量基本不存在多重共线性性问题，若VIF>10，说明问题比较严重。<br>
 （4）正态分布检验：正态QQ图（分位数—分位数图，本质是一个散点图）。如果服从正态分布，则Q-Q图呈现出一条直线。若直线出现偏差时，误差不服从正态分布。如果误差不是正态分布的，那么变量（响应或预测变量）的非线性变换可以改善模型。<br>
 （5）异方差性检验：法1：比例位置图（残差的标准差/估计值图(Scale Location Plot)）。显示了残差如何沿着预测变量的范围传播。（<br>
 法2：残差（Residual）/估计值（Fitted Value，Y^）图。若该图呈现如上图所示的“漏斗形”，即随着Y^的变化，残差有规律的变大或变小，则说明存在明显的异方差性。<br>
 
 
 
