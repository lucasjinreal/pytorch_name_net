# PyTorch 人工智能自动取名

本repo基于conditional rnn，如果对condition rnn不是非常了解也没有关系，大概原理就是根据给定的条件来生成不同的东西。我之前做过一个自动作诗机器人，近期打算用conditional RNN进行重构，这应该是个先行版本，最终目的就是指定不同年代创作不同的古诗，比如做宋词还是唐诗，还是汉律，还是现代诗，当然也欢迎大家直接把这个repo进行扩展。

![PicName](http://ofwzcunzi.bkt.clouddn.com/YEBpfA9QFHZpKNfC.png)

# Done

* 模型可以断点续训
* 数据集中包含了500个公司名字，和将近一千个人名，其中很多明星的名字
* Conditional RNN可以根据设定条件自动生成相应的名字，比如给定category是company就能生成有点像公司的名字


# Usage

使用非常简单：

```
python3 train.py
python3 generate.py company 国
```

# Future Work

这个项目可以作为RNN的练手，但是我隐约感觉到这里面有个很玄学的问题，那就是好像不同的姓生成的名字都差不多，大家有兴趣可以一起来debug了一下是哪里的问题，欢迎给我提PR！

# Copyright

```
this repo implement by Jin Fagang, owns the copyriht of datasets, please it user Apache License 2.0
```
