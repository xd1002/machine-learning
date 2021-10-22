# 西瓜书 机器学习 程序实现
西瓜书学习历程，用Python实现算法  
不定期更新，随缘debug  
  
1.matrix_determinant：计算矩阵行列式  
通过矩阵初等行变换将矩阵化为上三角  
  
  
2.matrix_inversion:计算矩阵的逆  
用矩阵的伴随求逆  
  
  
3.linear_regression:多元线性回归  
未使用牛顿法等的凸优化方法，使用正规方程法（吴恩达课程中的叫法）通过矩阵一步到位（但需要注意x的转置与x的乘积所得矩阵的可逆性与对应逆矩阵的正定性问题，若乘积矩阵不可逆，删去矩阵的部分列向量使得列向量线性无关，但一般不会出现线性相关导致的不可逆问题。如果对应逆矩阵非正定，考虑加入正则项，可选择足够大的系数取消正定性问题（对角占优），程序中暂时未加入正则项功能）  
  
  
4.logistic_regression:逻辑回归  
用牛顿法进行优化，但未加入正则项，随迭代次数增加参数会不断变大  
  
  
5.QR_decomposition:QR分解  
基于施密特正交化（Gram Schmdit）的QR分解（QR算法求特征值的预备工作，用for循环，不知道能不能用矩阵进行计算）  
  
  
6.householder:householder变换（镜面变换）
QR算法求特征值的预备工作，在QR算法运行前将待求矩阵转换为upper hessenberg矩阵，加快QR算法迭代  
  
  
7.QR_algorithm:QR算法
用于求解矩阵的特征值，实和复特征值都能求。先通过householder变换将矩阵化为upper hessenburg矩阵，然后用移动QR求解特征值（过程中所有变换与分解所得矩阵可证明都与原矩阵相似，于是具有相同特征值），复数特征值与特征向量也能求  
  
  
8.solve_homogeneous_linear_equation:解齐次线性方程组
使用初等行变换将系数矩阵化为行阶梯矩阵进行求解。其余数值解法有严格收敛条件，不够一般  
  
  
9.LDA：LDA  
用于降维，求解有关广义瑞丽商的优化问题  
  
  
10.SVM(正式版)：SVM  
准确来说只有SVC（用于分类），自认为SVR也是类似的。软间隔最大化。其中smo算法中alpha1的选取是在违反KKT条件的alpha1中随便选取一个，alpha2是随机选一个和alpha1不一样的。尝试用《统计学习方法》中的选取方法，但不知道是我编程的问题还是别的问题，会导致结果十分不稳定（就算数据是完全线性可分的也会有准确率不高的结果，且支持向量都不对），且书中并未明说怎样衡量违背KKT条件的严重程度。用西瓜书中的方法（间隔最大的两个样本）编完也不是很靠谱。当然alpha1和alpha2全都随机选取也是一种解决方案，但实验表明第一个在违反KKT条件的alpha分量中随便选会减少迭代次数。仍存在的问题：使用alpha满足KKT条件作为迭代的终止条件时，永远不会满足，且总是在满足KKT条件前alpha的变化小于容忍度导致的迭代终止。问题可能还是出在smo算法中alpha的选取上。参考过https://github.com/Kaslanarian/libsvm-sc-reading 这位老哥的整理，符号和我目前接触的有点不一样，等以后有时间再去认真看一下（南京大学的本科生真吊啊）  
  
  
11.NB：朴素贝叶斯  
仅能实现二分类，第一次尝试用字典进行程序的主体设计。离散和连续特征都能处理，但需要预先知道每个离散特征的可能取值种类数，如果是连续的话可能取值种类数置0  
  
12.decision_tree：决策树  
仅可用于二分类，还未编剪枝相关的程序，基于ID3算法（C4.5也一样，懒得编了，反正用self.find_possible_num返回的indice_num_dict中的学习能算出特征A的熵）
