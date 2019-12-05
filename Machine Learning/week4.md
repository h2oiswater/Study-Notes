# 4 Nerual Networks: Representation

## Motivations
### Non-linear Hypotheses
当特征值非常多时，使用逻辑回归来构建模型会导致Hypotheses中充斥着高阶函数，并且计算过程的复杂度为n(O)^3，所以这里使用神经网络算法可以比较好处理这类情况。

***Question:***
- 已知特征向量的数量时，逻辑回归中矩阵计算的唯独纬度确定不是很清楚。


### Neurons and the Brain
从研究人类大脑中总结出了现在的神经网络算法。

### Model Representation I
![Model Representation I](https://xiaoyu-1253702963.cos.ap-chengdu.myqcloud.com/ModelRepresentationI.png)

### Model Representation II
![Model Representation II](https://xiaoyu-1253702963.cos.ap-chengdu.myqcloud.com/ModelRepresentationII-1.png)
![Model Representation II](https://xiaoyu-1253702963.cos.ap-chengdu.myqcloud.com/ModelRepresentationII-2.png)

## Applications
### Examples and Intuitions I
### Examples and Intuitions II
使用简单的神经网络算法来表示AND，OR，XNOR计算。
神经网络算法的每一层计算与逻辑回归类似。

### Multiclass Classification
类似于逻辑回归中，在结果需要分多类时，我们使用矩阵表示返回值，
例如 [0,0,1]表示一共分为3类，当前结果为第3类。
