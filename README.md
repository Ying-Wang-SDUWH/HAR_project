# HAR动作姿态识别项目说明

> 山东大学（威海）   18 数据科学 王萦



#### B站视频（展示+讲解）：https://www.bilibili.com/video/BV1u54y1S7LQ/

#### 知乎文章：https://zhuanlan.zhihu.com/p/161612873



**人类行为识别**，简称HAR，指利用智能手机记录的三/六轴时间序列数据对人所做的动作进行识别或分类的问题。本项目选取了徒手侧平举、前后交叉小跑、开合跳、半蹲四个动作，通过微信小程序对动作进行识别。实际检测时，测试者左手持手机，利用微信小程序的API实时采集手机的六轴数据，利用已经训练好的随机森林模型和波峰检测法，对测试者的动作进行实时识别与计数。



## 一、文件夹说明

#### 1、“动作姿态识别”小程序源码

+ 小程序已发布，二维码如下，欢迎扫码体验：

<img src='https://picb.zhimg.com/80/v2-d2b7ad8cc633d003ed12c83ae04b2f47_1440w.jpg' width='800px'>

#### 2、har_notebook

> 项目全流程，包含原始数据与处理后数据，均有详细注释

+ raw_json文件夹：从云数据库导出的原始数据

+ raw_csv文件夹：剔除无效数据，添加动作标签后导出的csv文件

+ extract_feature文件夹：特征提取代码以及特征提取后的csv文件

+ filter.py：滤波、提取重力/加速度核心函数

+ forest.pkl：训练好的模型，部署至云端

+ **data_process.ipynb：项目全流程说明**

  

#### 3、云端flask代码.py

* 服务器端代码



## 二、小程序使用介绍

### 1、首页及动作示范

<table>
    <tr>
        <td ><center><img src="https://github.com/Ying-Wang-SDUWH/HAR_project/blob/master/picture/1.jpg"  width="80%"> </center></td>
        <td ><center><img src="https://github.com/Ying-Wang-SDUWH/HAR_project/blob/master/picture/7.jpg"  width="80%"></center></td>
    </tr>
    <tr>
        <td><center>首页</center></td>
        <td><center>查看动作示范</center> </td>
    </tr>
</table>




### 2、动作采集

<table>
    <tr>
       <td ><center><img src="https://github.com/Ying-Wang-SDUWH/HAR_project/blob/master/picture/2.jpg"  width="80%"> </center></td>
        <td ><center><img src="https://github.com/Ying-Wang-SDUWH/HAR_project/blob/master/picture/3.jpg"  width="80%"></center></td>
    </tr>
    <tr>
        <td><center>动作采集首页</center></td>
        <td><center>单个动作采集</center> </td>
    </tr>
</table>




### 3、动作识别

<table>
    <tr>
        <td ><center><img src="https://github.com/Ying-Wang-SDUWH/HAR_project/blob/master/picture/6.jpg"  width="80%"> </center></td>
        <td ><center><img src="https://github.com/Ying-Wang-SDUWH/HAR_project/blob/master/picture/5.jpg"  width="80%"></center></td>
    </tr>
    <tr>
        <td><center>动作识别中（语音播报）</center></td>
        <td><center>识别结束，显示个数</center> </td>
    </tr>
</table>



