# **DBIS 新手入门（ML & NLP）**

欢迎来到数据库与信息系统研究室，有志于ML&NLP方向的新人可从以下几个方面开始学习。

* [<strong>DBIS 新手入门（ML &amp; NLP）</strong>](#dbis-新手入门ml--nlp)
  * [<strong>0. 建议学习时间</strong>](#0-建议学习时间)
  * [<strong>1. <a href="https://git-scm.com/" rel="nofollow">Git</a> 与 <a href="https://github.com/">GitHub</a></strong>](#1-git-与-github)
  * [<strong>2. 基础数学知识</strong>](#2-基础数学知识)
  * [<strong>3. 机器学习</strong>](#3-机器学习)
  * [<strong>4. 深度学习</strong>](#4-深度学习)
  * [<strong>5. 学术前沿</strong>](#4-学术前沿)

---

## **0. 建议学习时间**

*建议0：优先学习未标星号的内容，快速上手*
*建议1：多动手编程和做笔记，"talk is cheap, show me the code, paper, algorithm, etc"*

| 编号 | 学习内容     | 学习时间 |
| ---- | ------------ | -------- |
| 1    | git & github | 1周      |
| 2    | 基础数学知识 | 2周      |
| 3    | 机器学习     | 4周      |
| 4    | 深度学习     | 5周      |

## **1. [Git](https://git-scm.com/) 与 [GitHub](https://github.com/)**

+ [Git教程（廖雪峰）](https://www.liaoxuefeng.com/wiki/896043488029600)

***要求***：

* 学会使用Git与GitHub，创建自己的GitHub账户并
* 熟练掌握git的基本命令、 版本控制、分支控制、远端操作等
* 学会如何使用git参与团队协作*

---

## **2. 基础数学知识**

**掌握吴恩达机器学习课程（[Stanford CS229](http://cs229.stanford.edu/)）中的数学知识**

**（0）必备基础**

* [Linear Algebra Review and Reference](http://cs229.stanford.edu/section-spring2019/cs229-linalg.pdf)
* [Probability Theory Review](http://cs229.stanford.edu/section/cs229-prob.pdf)
* [Convex Optimization Overview, Part I](http://cs229.stanford.edu/section/cs229-cvxopt.pdf)
* [Convex Optimization Overview, Part II](http://cs229.stanford.edu/section/cs229-cvxopt2.pdf)

**（1）更多数学知识***

* [Hidden Markov Models](http://cs229.stanford.edu/section/cs229-hmm.pdf)
* [The Multivariate Gaussian Distribution](http://117.128.6.34/cache/cs229.stanford.edu/section/gaussians.pdf?ich_args2=470-19221120042387_b250646c60853f3a586bb1167beec8e2_10001002_9c89622cdec6f2d59f3b518939a83798_3c329201159bc30095bcfac4631f563d)
* [More on Gaussian Distribution](http://cs229.stanford.edu/section/more_on_gaussians.pdf)
* [Gaussian Processes](http://cs229.stanford.edu/section/cs229-gaussian_processes.pdf)

**（2）推荐书籍***

* Deep Learning, Goodfellow etc., Chapter 1-5
* Pattern Recognition and Machine Learning, Bishop, Chapter 1-3

---

## **3. 机器学习**

**（0）公开课**

*建议0：根据自身情况二选一，必须要完整的看过一门课程，第一遍部分内容看不懂没关系*
*建议1：多搜索，多与实验室师兄师姐交流*

+ 机器学习基石+机器学习技法，林轩田，台湾大学 （网上容易下载到视频）
  + 优点：循序渐进、深入浅出、全中文教学
  + 缺点：有的内容较难

+ Machine Learning, Stanford CS229, Andrew Ng
  + 优点：网上中英文资料多；难度合适，大多数人机器学习的入门课程
  + 缺点：主要讲怎么做，模型背后的原理和思想涉及较少

**（1）推荐书籍**

*建议：想系统学习机器学习，建议看完以下任一本书，推荐排名有先后*

+ [Pattern Recognition and Machine Learning, Bishop](http://users.isr.ist.utl.pt/~wurmd/Livros/school/Bishop%20-%20Pattern%20Recognition%20And%20Machine%20Learning%20-%20Springer%20%202006.pdf)* [虽然难，但是强烈推荐]
+ [机器学习，周志华著](https://www.zhihu.com/question/39945249) [公式推导跳跃性较大，全面的机器学习参考书]
+ [统计学习方法，李航著](http://www.dgt-factory.com/uploads/2018/07/0725/%E7%BB%9F%E8%AE%A1%E5%AD%A6%E4%B9%A0%E6%96%B9%E6%B3%95.pdf) [第二版出版了]

---

## **4. 深度学习**

**（0）深度学习基础**

*建议0：以下推荐排名有先后，尽量多参加实验室讨论班，多与实验室老师和师兄师姐交流*
*建议1：跳过基础篇直接看进阶篇也能看懂*

+ [吴恩达深度学习（网易云课堂）](https://mooc.study.163.com/university/deeplearning_ai#/c) + [吴恩达深度学习（课程笔记及资源）](https://github.com/fengdu78/deeplearning_ai_books) [非常简单和基础]

+ [《动手学深度学习》（英文版即伯克利“深度学习导论（STAT 157）”教材）](https://github.com/d2l-ai/d2l-zh) [重点在如何编程]

+ [深度学习（花书，Goodfellow，Bengio等著）](https://github.com/zsdonghao/deep-learning-book/blob/master/dlbook_cn_public.pdf)* [很好的综述]

***要求***：

+ 通过任一教程掌握神经网络的基本概念与结构，知道神经元、激活函数、反向传播等基本概念
+ 了解CNN、RNN等神经网络常见的结构

**（1）领域经典课程**

*建议：CS224d是学习DL for NLP的必修课*

+ Stanford CS224d, Deep Learning for Natural Language Processing
+ Stanford CS231n, Convolutional Neural Networks for Visual Recognition*

***要求***：
* 重点学习NLP相关的课程
* 理解并掌握深度学习NLP模型如何构建和实现

**（2）深度学习编程框架**

*建议0：编程方面要对python有一些了解，比如：Python的内置数据结构list, dict, tuple等，重要的包numpy, pandas等*
*建议1：人生苦短，我用pytorch :-)*
*建议2：pytorch和tensorflow在网上都有大量的入门资料，多查查github*

+ pytorch
    + [官方tutorials](https://pytorch.org/tutorials/)

+ tensorflow
  + [TensorFlow官方文档中文版](http://wiki.jikexueyuan.com/project/tensorflow-zh/)

+ [AiLearning](https://github.com/apachecn/AiLearning)*


***要求***：

  + 安装配置深度学习环境
  + 掌握TensorFlow或者PyTorch或者其他深度学习框架的使用，阅读框架的模型样例代码，对经典模型的实现有深度了解
  + 尝试自己复现领域经典模型

---

## **5. 学术前沿**

*建议：利用学术网站和学术会议，紧跟领域发展现状*

+ 熟练使用Google、Bing学术、DBLP、Arxiv等网站
+ ML、AI和DM方向的重要国际会议：
  + 方法论: ICML, NIPS
  + 泛AI: AAAI, IJCAI
  + 应用，偏NLP: ACL, EMNLP
  + 应用，偏CV: CVPR, ICCV
  + 应用，领域不限: ICDE, SIGKDD, SIGIR, CIKM

---
