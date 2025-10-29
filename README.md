ROLO
=======
--------

Project Page: [http://guanghan.info/projects/ROLO/](http://guanghan.info/projects/ROLO/)

## 概览

ROLO是Recurrent YOLO 的简称[[1]]，旨在实现同步的目标检测与跟踪。

得益于长短期记忆网络（LSTM）在时空维度上的回归能力，ROLO能够将一系列高级视觉特征直接解析为被追踪目标的坐标。通过将高级视觉特征与YOLO检测结果进行融合拼接，该系统在空间维度上获得了监督信号，从而实现对特定目标的精准定位追踪。

其回归机制体现在两个层面:
(1) 单元内的回归，即视觉特征与拼接的区域表征之间的回归。当这些特征被拼接为一个单元时，LSTM能够从视觉特征中推断出区域位置。
(2) 序列单元间的回归，即在连续帧序列中拼接特征之间的回归。

这种监督机制在两方面发挥重要作用：:
(1) 当LSTM解析高级视觉特征时，初步的位置推断有助于将特征回归到特定视觉元素/线索的位置。这种空间监督的回归相当于一个在线外观模型。
(2) 在时间维度上，LSTM通过序列单元的学习将位置预测限制在特定空间范围内。

目前ROLO采用离线跟踪方式，若配以合适的在线模型更新机制，其性能有望进一步提升。该系统仍为单目标跟踪器，尚未探索用于多目标同步跟踪的数据关联技术。

----
## 环境
- Python 2.7 or Python 3.3+
- Tensorflow
- Scipy

----
## 开始

### 1. 下载数据和预训练模型

作为通用目标检测器，YOLO可通过训练识别任意目标。然而由于ROLO的性能依赖于YOLO结果，为确保比较的公平性，我们选择默认的YOLO小模型。若采用定制化训练的YOLO模型来衡量跟踪模块的性能不利于最后结果比对的公共。所以模型先在ImageNet数据集上进行预训练，然后在VOC数据集上微调，达到20个类别的目标的分类。因此，我们从基准测试 [OTB100](http://cvlab.hanyang.ac.kr/tracker_benchmark/datasets.html) 中选取了OTB100基准测试100个视频中的30个，这些视频的跟踪目标均属于上述类别，该子集被称为OTB30。

**数据**

- [DATA and Results for Demo](http://guanghan.info/projects/ROLO/DATA/DATA.zip)

**模型**

- [Model for demo](http://guanghan.info/projects/ROLO/demo/model_demo.ckpt)

- [Model for experiment 1](http://guanghan.info/projects/ROLO/experiment_1/model_step6_exp1.ckpt)

- Model for experiment 2: [step=1](http://guanghan.info/projects/ROLO/experiment_2/model_step1_exp2.ckpt), [step=3](http://guanghan.info/projects/ROLO/experiment_2/model_step3_exp2.ckpt), [step=6](http://guanghan.info/projects/ROLO/experiment_2/model_step6_exp2.ckpt), [step=9](http://guanghan.info/projects/ROLO/experiment_2/model_step9_exp2.ckpt)

- [Model for experiment 3](http://guanghan.info/projects/ROLO/experiment_3/model_step3_exp3.ckpt)

**环境**

- [Evaluation Results (including other trackers)](http://guanghan.info/projects/ROLO/output/evaluation.rar)

### 2. Demo运行

使用预训练模型复现结果:

	python ./experiments/testing/ROLO_network_test_all.py

Or download the results at [Results](http://).

运行视频 Demo:

	./python ROLO_demo_test.py


### 3. 训练与测试

随着深度学习应用日趋成熟，构建由正交模块组成的多功能网络将更为高效。在这种情况下，特征表征最好通过独立训练来提供共享特征。如YOLO论文已讨论的，我们跳过了ImageNet视觉特征的预训练环节，将重点放在LSTM模块的训练上。

**实验 1**:

离线跟踪的局限性在于模型训练需要海量数据支撑，而公开的目标跟踪基准数据集却难以满足这一需求。即便采用OTB100[[2]]全部100个视频，其数据规模仍比图像识别任务小几个数量级，因此跟踪器极易出现过拟合现象。

为验证ROLO的泛化能力，我们设计了实验1：使用OTB30中22个视频进行训练，并在其余8个视频上进行测试。结果表明，该模型的性能超越了基准测试[[2]]中所有传统跟踪器。

我们从未入选OTB30的视屏中额外测试了3个真实标注对象为人脸而非人体的视频。由于默认YOLO模型未包含人脸类别，YOLO会转而检测人体目标，而ROLO则在监督指导下对人体进行跟踪。演示视频可在此处查看。
[Video 1](https://www.youtube.com/watch?v=7dDsvVEt4ak),
[Video 2](https://www.youtube.com/watch?v=w7Bxf4guddg),
[Video 3](https://www.youtube.com/watch?v=qElDUVmYSpY).

[Video 4](https://www.youtube.com/embed/7dDsvVEt4ak)

实验1复现：

- Training: 

	```
	python ./experiments/training/ROLO_step6_train_20_exp1.py
	```

- Testing: 

	```
	python ./experiments/testing/ROLO_network_test_all.py
	```

**实验 2**:

若模型必须在有限数据下进行训练，可通过具有相似运动模式的数据进行弥补（此策略同样适用于采用在线模型更新的跟踪器）。使用OTB30数据集前1/3帧序列训练第二个LSTM模型，并在剩余帧上进行测试。结果表明该方法的性能得到了提升。研究发现，当在具有相似运动模式的辅助帧序列上训练后，ROLO在测试序列中表现更优。这一特性使得ROLO在监控场景中尤为实用，证明利用预采集数据离线训练模型是可行的。

实验2复现：

- Training:

	```
	python ./experiments/training/ROLO_step6_train_30_exp2.py
	```
- Testing:
	```
	python ./experiments/testing/ROLO_network_test_all.py
	```


**实验 3**:

基于实验2中观察到的这一特性，我们尝试增加训练帧数进行实验。
结果表明，使用全部视频帧进行训练（但仅使用其中1/3的真实标注数据）能够进一步提升模型性能。

实验3复现：

- Training:

	```
	python ./experiments/training/ROLO_step6_train_30_exp3.py
	```
- Testing:
	```
	python ./experiments/testing/ROLO_network_test_all.py
	```

**局限性分析**

实验2与实验3均使用了1/3的视频帧进行训练。在性能评估时，这些训练帧必须从测试集中排除。同时，即使在同一视频序列中分别选取训练帧和测试帧，仍可能存在数据泄露风险。这点在ROLO设计在线更新机制需特殊注意。

后续将采用定制化YOLO模型进行实验更新，以实现对任意目标的检测能力。这将支持在完整OTB100数据集上的测试，并能够通过在不同数据集上进行训练与测试来完成交叉验证。

**参数敏感性分析**

使用不同step: [1, 3, 6, 9]重复实验2

```
python ./experiments/testing/ROLO_step1_train_30_exp2.py
```

```
python ./experiments/testing/ROLO_step3_train_30_exp2.py
```

```
python ./experiments/testing/ROLO_step6_train_30_exp2.py
```

```
python ./experiments/testing/ROLO_step9_train_30_exp2.py
```

![](http://guanghan.info/projects/ROLO/fps_over_steps.png)
![](http://guanghan.info/projects/ROLO/IOU_over_steps.png)

### 4. 可视化

- Demo:
	```
	python ./ROLO_demo_heat.py
	```
- Training:
	```
	python ./heatmap/ROLO_heatmap_train.py
	```
- Testing:
	```
	python ./heatmap/ROLO_heatmap_test.py
	```

![](http://guanghan.info/projects/ROLO/heatmap_small1.png)
![](http://guanghan.info/projects/ROLO/heatmap_small2.png)
- Blue: YOLO结果
- Red: 真实标注

### 5. 性能评估

	python ./ROLO_evaluation.py


### 6. 结果

更多定性分析结果请参见项目页面，定量分析结果请参阅arXiv论文。

![](http://guanghan.info/projects/ROLO/occlusion.jpeg)
![](http://guanghan.info/projects/ROLO/occlusion2.jpeg)

- Blue: YOLO检测
- Green: ROLO跟踪
- Red: 真实标注


---
## License

ROLO依据Apache License Version 2.0版本发布（具体条款详见LICENSE文件）。

---
## 引用
The details are published as a technical report on arXiv. If you use the code and models, please cite the following paper:
[arXiv:1607.05781](http://arxiv.org/abs/1607.05781).

	@article{ning2016spatially,
	  title={Spatially Supervised Recurrent Convolutional Neural Networks for Visual Object Tracking},
	  author={Ning, Guanghan and Zhang, Zhi and Huang, Chen and He, Zhihai and Ren, Xiaobo and Wang, Haohong},
	  journal={arXiv preprint arXiv:1607.05781},
	  year={2016}
	}


---
## 参考
[[1]] Redmon, Joseph, et al. "You only look once: Unified, real-time object detection." CVPR (2016).

[1]: http://arxiv.org/pdf/1506.02640.pdf "YOLO"

[[2]] Wu, Yi, Jongwoo Lim, and Ming-Hsuan Yang. "Object tracking benchmark." IEEE Transactions on Pattern Analysis and Machine Intelligence 37.9 (2015): 1834-1848.

[2]: http://ieeexplore.ieee.org/xpl/login.jsp?tp=&arnumber=7001050&url=http%3A%2F%2Fieeexplore.ieee.org%2Fxpls%2Fabs_all.jsp%3Farnumber%3D7001050 "OTB100"
