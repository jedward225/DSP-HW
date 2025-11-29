# DSP-HW

## Intro

This is the repository of the final project of the course DSP taught by Prof. Wenbing Huang.

Group Member:

Jiajun Liu, Haoxiang Sun, Yuan Tian, Xuyan Ye, Zijie Lin


## Task-description

![](task-describe.png)

> ## Requirements
> 
> 任务1：声音检索
> 
> 【代码实现】
> 需要自己实现FFT、STFT、MFCC等算法。
> 利用最后1个fold作为查询声音，前4个fold作为候选数据库，判断Top10、Top20中找到相同类别声音的精度。
> 需要比较不同的帧移、帧长等超参数下的精度。
> 
> 任务2：声音分类
> 
> 【代码实现】
> 需要自己实现FFT、STFT、MFCC等算法。
> 可以自由选择不同的神经网络模型，可以直接调用已有的模型和训练代码。
> 需要比较不同的帧移、帧长等超参数下模型的分类精度。
> 利用前4个fold进行训练，利用最后1个fold进行测试。
> 将该模型用于前面检索任务，对比有无机器学习的效果。
> 需要与大模型直接分类做对比。
> 
> 
> 【提交内容】
> 建议使用Pytorch。
> 报告1份：实现的流程、训练曲线、测试精度、不同setting的比较等。
> 代码1份：包括readme、requirement。
> 
> 分数：50分
> 时间：16周上课前一天提交所有材料，16周进行项目展示
> 组队要求：3~5人，报告需要明确每个人的角色和职责
> 