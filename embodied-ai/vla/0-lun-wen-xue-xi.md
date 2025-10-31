---
description: 'π0: A Vision-Language-Action Flow Model for  General Robot Control'
---

# Π0：论文学习



1. produces continuous actions via flow matching。这里的flow matching是什么\
   A：相当于diffusion从噪音到降噪的过程。获得从噪声到真实动作的流场。主动加噪声，作为数据喂给模型让他学会去噪过程
2. 训练具体任务也是先训练general任务再进行微调效果更好，为什么
3. general模型需要训练数据达到一定规模存在一个必须的阈值
4. training recipe的优化始终是最重要的一步
5. VLA训练时选择了不同具身的数据的融合
6. 传统方法：交叉熵离散的动作。采取flow matching可以获得连续动作。\
   借鉴以往的分权重（也就是分模块），VLM+expert\


<figure><img src="../../.gitbook/assets/image (1) (1).png" alt=""><figcaption></figcaption></figure>

7. 他们并非简单地照搬 Transfusion 的混合训练模式，而是通过创建一个专门负责处理机器人本体状态和动作的“专家网络”（拥有独立的参数），对原有思想进行了优化，并用实验证明了这种“专业化分工”的设计能够带来实实在在的性能好处。
8.  模型学习根据【状态，摄像头，语义】学习给出的动作A，但是

    * 动作块 (Action Chunk)：模型一次性预测出未来 H 个时间步的完整动作序列 。
    * H = 50：在这篇论文的任务中，模型会一次性生成接下来连续50步的动作 。这使得机器人的动作非常连贯和流畅，而不是一系列断断续续的、卡顿的动作

    “翻译”过程：模型会用不同的“编码器 (encoders)”把这三种不同格式的数据，“翻译”成统一的数学语言（即“same embedding space”，相同的嵌入空间）
