---
license: Apache License 2.0


text:
  conversational:
    size_scale:
      - 258k
    type:
      - task-qa
  text-generation:
    language:
      - zh
  question-answering:
    language:
      - zh

configs:
- config_name: default
  data_files:
  - split: train
    path: "SoulChatCorpus-sft-multi-Turn.json"
---

# SoulChatCorpus
该项目为项目[https://github.com/scutcyr/SoulChat](https://github.com/scutcyr/SoulChat)的数据集部分。

## 最近更新
- 👏🏻  2024.06.06：SoulChatCorpus数据集的开源版本发布，详情见[https://www.modelscope.cn/datasets/YIRONGCHEN/SoulChatCorpus](https://www.modelscope.cn/datasets/YIRONGCHEN/SoulChatCorpus)，特别地，我们过滤掉了约9万个对话样本（由于隐私风险、安全问题、政治风险、低质量样本等，该部分样本尚在人工优化阶段，人工审核完毕后，会更新到开源版本数据集当中），最终保留258354个多轮对话，总共1517344轮。新版本模型将会在近期发布，预计适配多个开源模型，以及多个参数量级，以方便使用者使用或进行对比实验研究等。
- 👏🏻  2023.12.07：我们的论文，收录在Findings of EMNLP 2023，详见[SoulChat: Improving LLMs’ Empathy, Listening, and Comfort Abilities through Fine-tuning with Multi-turn Empathy Conversations](https://aclanthology.org/2023.findings-emnlp.83/)


## 数据样例
```
{
    "id": 0,
    "topic": "成长",
    "messages": [
        {
            "role": "user",
            "content": "最近感觉很焦虑，不知道如何缓解。"
        },
        {
            "role": "assistant",
            "content": "我明白你的感受。首先，要知道焦虑是一种正常的情绪。但是，当它影响到你的生活质量时，就需要采取行动了。"
        },
        {
            "role": "user",
            "content": "可是我总觉得自己做什么都不够好，很害怕失败。"
        },
        {
            "role": "assistant",
            "content": "这可能是完美主义导致的。完美主义者总是把难度设置得很高，并在失败时很难接受自己。其实，你是一个有自尊和价值的人，无论成败。"
        },
        {
            "role": "user",
            "content": "但是我总是无法集中注意力，会经常拖延。"
        },
        {
            "role": "assistant",
            "content": "拖延是一种很常见的自我保护机制，但长期下来对心理健康负面影响很大。你可以尝试分段任务，每段任务完成后休息片刻，然后继续下一个任务。"
        },
        {
            "role": "user",
            "content": "但是每天都很疲惫，很难找到动力去改变。"
        },
        {
            "role": "assistant",
            "content": "情绪低落，疲惫和动力不足是相互作用的。你可以尝试建立自己的日常规划，合理分配时间，保证充足的睡眠和营养，以此建立更加健康的自己。"
        },
        {
            "role": "user",
            "content": "其实我很害怕自己无法排解焦虑情绪。"
        },
        {
            "role": "assistant",
            "content": "我会一直在这里支持你，并分享一些我的应对技巧。你可以与朋友交流，寻求专业帮助，做一些自己感兴趣的事情，放松身心。"
        },
        {
            "role": "user",
            "content": "我会尝试这些方法，谢谢你的支持和理解。"
        },
        {
            "role": "assistant",
            "content": "不要忘记，焦虑并不是终点，而是一个可以克服的过程。你是勇敢的，你可以做到。任何时候，都欢迎你来和我交流。"
        }
    ]
}
```



## 下载方法 
:modelscope-code[]{type="sdk"}
:modelscope-code[]{type="git"}




## 声明
* 本项目使用了ChatGLM-6B 模型的权重，需要遵循其[MODEL_LICENSE](https://github.com/THUDM/ChatGLM-6B/blob/main/MODEL_LICENSE)，因此，**本项目仅可用于您的非商业研究目的**。
* 本项目提供的SoulChat模型致力于提升大模型的共情对话与倾听能力，然而，模型的输出文本具有一定的随机性，当其作为一个倾听者的时候，是合适的，但是不建议将SoulChat模型的输出文本替代心理医生等的诊断、建议。本项目不保证模型输出的文本完全适合于用户，用户在使用本模型时需要承担其带来的所有风险！
* 您不得出于任何商业、军事或非法目的使用、复制、修改、合并、发布、分发、复制或创建SoulChat模型的全部或部分衍生作品。
* 您不得利用SoulChat模型从事任何危害国家安全和国家统一、危害社会公共利益、侵犯人身权益的行为。
* 您在使用SoulChat模型时应知悉，其不能替代医生、心理医生等专业人士，不应过度依赖、服从、相信模型的输出，不能长期沉迷于与SoulChat模型聊天。

## 致谢
本项目由[华南理工大学未来技术学院](https://www2.scut.edu.cn/ft/main.htm) 广东省数字孪生人重点实验室发起，得到了华南理工大学信息网络工程研究中心、电子与信息学院等学院部门的支撑，同时致谢广东省妇幼保健院、广州市妇女儿童医疗中心、中山大学附属第三医院、合肥综合性国家科学中心人工智能研究院等合作单位。

同时，我们感谢以下媒体或公众号对本项目的报道（排名不分先后）：
* 媒体报道
  [人民日报](https://wap.peopleapp.com/article/rmh36174922/rmh36174922)、[中国网](https://hs.china.com.cn/gd/83980.html)、[光明网](https://health.gmw.cn/2023-06/13/content_36628062.htm)、[TOM科技](https://tech.tom.com/202306/4526869977.html)、[未来网](http://www.zzfuture.cn/news/956.html)、[大众网](http://linyi.dzwww.com.3xw.site/xinwen/202306/t20230613_202306135667.htm)、[中国发展报道网](http://www.chinafzbdw.com/computer/13149.html?1686564408)、[中国日报网](http://energy.chinaduily.com.cn/c/2023/15205.html)、[新华资讯网](http://www.xinhuazxun.com/world/21762.html?1686564382)、[中华网](https://life.china.com/2023-06/12/content_215815.html)、[今日头条](https://www.toutiao.com/article/7243412314223952418/)、[搜狐](https://www.sohu.com/a/684501109_120159010)、[腾讯新闻](https://page.om.qq.com/page/OhSXIMEUtDtdg0rTi6aAoTbg0)、[网易新闻](https://www.163.com/dy/article/I70BJ9U00552UJUX.html)、[中国资讯网](http://www.chinazxun.com/world/23252.html?1686564532)、[中国传播网](http://www.chinachbo.com/a/view/11697.html?1686564509)、[中国都市报道网](http://www.zgdsbdw.com/meida/11273.html?1686564485)、[中华城市网](http://www.zhcsww.com/hot/2023/0612/9609.html?1686564434)

* 公众号
  [广东实验室建设](https://mp.weixin.qq.com/s/gemlKfLg8c-AtjiV7uTUTQ)、[智能语音新青年](https://mp.weixin.qq.com/s/vBMKXUJoAIywkXY2nY60eA)、[深度学习与NLP](https://mp.weixin.qq.com/s/qSHLT8FbvohZESp-UCah6g)、[AINLP](https://mp.weixin.qq.com/s/EX3f9WblLKM8K_nSwhno_g)

## 引用
```bib
@inproceedings{chen-etal-2023-soulchat,
    title = "{S}oul{C}hat: Improving {LLM}s{'} Empathy, Listening, and Comfort Abilities through Fine-tuning with Multi-turn Empathy Conversations",
    author = "Chen, Yirong  and
      Xing, Xiaofen  and
      Lin, Jingkai  and
      Zheng, Huimin  and
      Wang, Zhenyu  and
      Liu, Qi  and
      Xu, Xiangmin",
    editor = "Bouamor, Houda  and
      Pino, Juan  and
      Bali, Kalika",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2023",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.findings-emnlp.83",
    pages = "1170--1183",
    abstract = "Large language models (LLMs) have been widely applied in various fields due to their excellent capability for memorizing knowledge and chain of thought (CoT). When these language models are applied in the field of psychological counseling, they often rush to provide universal advice. However, when users seek psychological support, they need to gain empathy, trust, understanding and comfort, rather than just reasonable advice. To this end, we constructed a multi-turn empathetic conversation dataset of more than 2 million samples, in which the input is the multi-turn conversation context, and the target is empathetic responses that cover expressions such as questioning, comfort, recognition, listening, trust, emotional support, etc. Experiments have shown that the empathy ability of LLMs can be significantly enhanced when finetuning by using multi-turn dialogue history and responses that are closer to the expression of a psychological consultant.",
}
}
```

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=scutcyr/SoulChat&type=Date)](https://star-history.com/#scutcyr/SoulChat&Date)

