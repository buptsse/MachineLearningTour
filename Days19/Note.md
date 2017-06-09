# Day19
官网上对TensorFlow的介绍是，一个使用数据流图(data flow graphs)技术来进行数值计算的开源软件库。数据流图中的节点，代表数值运算；节点节点之间的边，代表多维数据(tensors)之间的某种联系。你可以在多种设备（含有CPU或GPU）上通过简单的API调用来使用该系统的功能。

有向图中的边表示节点之间的某种联系，它负责传输多维数据(Tensors)。图中这些tensors的flow也就是TensorFlow的命名来源。

图中的节点称为ops(operation的简称)

一个ops使用0个或以上的Tensors，通过执行某些运算，产生0个或以上的Tensors


TensorFlow程序中图的创建类似于一个 [施工阶段]，而在 [执行阶段] 则利用一个session来执行图中的节点。很常见的情况是，在 [施工阶段] 创建一个图来表示和训练神经网络，而在 [执行阶段] 在图中重复执行一系列的训练操作。

TensorFlow中使用tensor数据结构（实际上就是一个多维数据）表示所有的数据，并在图计算中的节点之间传递数据。

变量(Variables),变量在图执行的过程中，保持着自己的状态信息。

填充(Feeds):TensorFlow也提供这样的机制：先创建特定数据类型的占位符(placeholder)，之后再进行数据的填充。

[计算机求导的几种方法](http://www.10tiao.com/html/149/201607/2650470496/1.html)
[自动微分](http://blog.csdn.net/aws3217150/article/details/70214422)


[Huber Lost Function](http://blog.csdn.net/lanchunhui/article/details/50427055)




