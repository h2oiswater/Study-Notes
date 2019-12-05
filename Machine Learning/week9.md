9 Anomaly Detection

Density Education

### Problem Motivation
用于查找异常

### Gassian
μ和σ平方，μ代表了中心点，σ代表了中心点到两边的距离。

### Algorithm
![Algorithm](https://xiaoyu-1253702963.cos.ap-chengdu.myqcloud.com/AnomalyDetectionAlgorithm.png)

Building an Anomaly Detection System
===

### Developing and Evaluating an Anomaly Detection System
![AlgorithmEvaluation](https://xiaoyu-1253702963.cos.ap-chengdu.myqcloud.com/AlgorithmEvaluation.png)

### Anomaly Detection vs. Supervised Learning
- 使用异常检测的情况
	- 样本为倾斜类 y = 0
	- 特征值种类繁多，无法捕捉是由什么引起的异常
- 使用监督学习的情况
	- 样本情况比较均匀

### Choosing What Features to Use
把参数的直方图更高斯一点
Choose features that might take on unusually large or small values in the event of an anomaly.
