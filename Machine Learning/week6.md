6 Advice for Applying Machine Learning

Evaluating a Learning Algorithm
===

### Deciding What to Try Next

首先吴恩达给予了大家非常高的评价，说学到这个部分的同学都已经是experts，还是挺搞笑的。当然懂算法和用算法是两码事，因此学会评测算法就成了重要的技能。提高算法的效率，很好的办法有一个是收集更多的训练样本。也可以挑选几个特征，训练精准的模型。增加数据的特征也是一个好办法。最多人用的办法是，凭感觉，走玄学路线。

	Machine Learning Diagnostic：机器学习诊断法，利用这个测试我们可以了解到算法的局限性。
虽然诊断法需要用到比较多的时间，但是“磨刀不误砍柴工”。

### Evaluating a Hypothesis

假设有可能会导致过拟合，特征过多的时候很难找到假设的错误。首先将已有的数据分成两部分：训练集和测试集。首先用训练集训练出函数，之后用测试集计算出最后的函数中的误差。在逻辑回归中，还有一种值叫做误分率（或0/1错分率）也可以被用来检测误差。其本质就是通过逻辑回归最后输出的“非1既0”的特性来测算其中的误差，非常机智的二分法。

### Model Selection and Train/Validation/Test Sets

在模型选择的过程当中，需要考虑的问题其实非常复杂，因此将数据集分为训练集、验证集、和测试集后的结果，会帮我们选到更好的模型。在模型选择中，首先最重要的就是确定输入应该是多少次方，一般用d来表示。于是用训练集依次训练各个次方的模型，后比较假设参数。之后再使用验证集改良函数，从而给测试集寻找最终误差的机会。如果没有验证集，被选出来的函数被改良后，已经过拟合，为时已晚。

Bias vs. Variance
===

### Diagnosing Bias vs. Variance

偏差大说明欠拟合，大方差大说明过拟合。为了解决这些问题，我们需要同时计算出训练误差和交叉验证误差。训练误差会随着最高指数的升高而减少，但是这没有办法验证出过拟合的情况，只能解决欠拟合的问题。在算出交叉验证误差之后，我们可以明显看到过拟合的问题，从而将其解决。

### Regulation and Bias/Variance

正则化是解决过拟合问题的有效方法。首先我们只使用训练、验证、以及测试的平方差，而不添加正则化。之后通过穷举的方式找到表现最好的 \lambda ，从而放入验证函数，再放入测试函数，从而获得效果最好的参数。

### Learning Curves

	Learning Curve：学习曲线，用于判断所使用的算法是否存在偏差、方差、或者二者皆有的问题。
以样本数为横轴，误差数为纵轴，依次递增，画出曲线。我们会发现训练代价函数的误差会逐渐增加。不过验证代价函数会随着训练样本的增加而减少误差，因为函数被调试地越来越好。当最终结果表现为高偏差的时候，即使数据再多偏差还是会很大，同时验证误差和训练误差两条函数会几乎重合。当表现为高方差的时候，更多的数据会有帮助，虽然训练误差和验证误差之间会出现非常大的代沟，但随着样本增加算法的表现会越来越好。

### Deciding What to Do Next (Revisited)

重新回顾一下如何改良算法。更多的样本和更少的特征可以改善高方差。更多的特征和多项式可以改善高偏差。降低 \lambda ，改善高偏差；提高 \lambda ，改善高方差。

在神经网络中，小的神经网络易造成欠拟合，但是计算成本低。大的神经网络易造成过拟合，不过通过正则化改善，而且计算效果比小神经网络好很多，但问题是计算成本高。但现在不需要考虑这个问题，因为GPU越来越厉害了。

Building a Spam Data
===

### Prioritizing What to Work On

这两个视频中举了制作垃圾邮件分类器的例子，非常有用。垃圾邮件多数情况下会有一些错误的拼写，当然现在的垃圾邮件已经很强了。首先将垃圾邮件设为“1”，正常邮件设为“0”。然后，造一些特殊词汇作为特征，有些词的出现会更加偏向于垃圾邮件，有些词的出现会更偏向与普通邮件，都可以人为设定。之后将这些特征做成向量。在实际工作中，我们可以提取最常用的1万至5万个词汇，做成训练集，给模型加以训练。

在提升模型效能的部分有个搞笑的方法，工程师们有一个叫做“honeypot”的项目，仿造大量的电子邮件，以获取大量的垃圾邮件从而可以训练自己的模型。其他方法有可以通过邮件的外部信息来提升模型的效能。或者分析文本本身，这里使用现代自然语言处理效能肯定有飞一般地提升。

### Error Analysis

	Error Analysis：误差分析，通过实际效果和检验集的比较，从而发现系统级的可优化空间。
最开始搭建垃圾邮件分类器的时候，没必要一口气吃成一个胖子，可以从很小的模型开始。通过学习曲线可以了解到结果，从而改良模型。误差分析也对改良模型有非常大的帮助。

案例中吴恩达首先手动给100个错误分类的邮件重新进行分类，发现盗取密码的邮件识别率低。因此就有了改良的方向，主要研究密码盗取的邮件。正常邮件被误判也可以通过同样的办法进行分析。语法的改善，通过NLP可以得到很好的效果。当然，是否要使用NLP，可以在检验误差之后酌情考量。

Handling Skewed Data
===

### Error Metrics for Skewed Classes

	Error Metrics：误差度量值，判断算法好坏的值。
	Skewed Class：偏斜类，当一种数据量远远大于另一数据的量。
	Precision：查准率，真阳性/（真阳性+假阳性）*100%。
	Recall：召回率，真阳性/（真阳性+假阴性）*100%。

	True positive：真阳性，真实类为真，预测类为真。
	False positive：假阳性，真实类为假，预测类为真。
	True negative：真阴性，真实类为假，预测类为假。
	False negative：假阴性，真实类为真，预测类为假。
开篇是一个很搞笑的例子，一个预测癌症的模型，准确率高达99%，但是患病率其实只有0.5%。因此，如果做一个只会输出没有癌症的模型，准确率会更高，甚至高达99.5%。当我们学会使用召回率之后，就会发现这个算法的召回率只有0，因此这个算法不能用。即使我们的算法在偏斜类的问题中没有达到极好的表现，但只要有很好的查准率和召回率，这个算法必定是个好算法。

### Trading off Precision and Recall

依旧是癌症预测的例子，这次吴恩达使用的不再是搞笑函数，而是逻辑回归。为了照顾患者的感受，医生必须在有十足把握的情况下才可以告诉患者病情。假阳性也非常需要被避免，以此避免有患者错过最佳的治疗机会。然而这两种情况互为悖论。高召回率的情况下，查准率会变低。因此选取一个临界值成为必要。

为了选到一个非常好的临界值，我们需要测试不同算法的查准率和召回率。虽然选取二者平均值最高的算法看起来是个不错的选择，但是极端情况都会出现弊端。因此极高或者极低的两样数值都不应该成为最佳选择。而F值（或 F_1 值）是一个非常有用的算法来评估这两种数值。F值当然是高越好，最高为1。

![F Score](https://xiaoyu-1253702963.cos.ap-chengdu.myqcloud.com/FScore.png)


Using Large Data Sets
===

### Data for Machine Learning

经过试验研究人员发现，数据越多，算法的效果就越好。
