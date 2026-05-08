# Harness Engineering 考核说明（2026年夏）

背景

![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-07/110c59b8-168a-4838-a912-b1621039fd7c/d5578c2ffcd2d45a47656c908d6a673385cac15b0d249ba78447959144f11786.jpg)


# Harness Engineering 考核

# 背景

Harness是⼀种围绕LLM构建的外部框架，通过控制LLM的输⼊（Prompt构造、记忆检索、上下⽂管理）和输出（解析、验证、多轮推理）来完成复杂任务。LLM本⾝只能⽣成⽂本⸺它⽆法读⽂件、跑命令、改代码；Harness的职责，是把模型输出中的“⼯具调⽤”截获、执⾏，再把结果回灌进对话历史，驱动模型继续推进，直到任务收敛。可以说，模型是⼤脑，Harness是⾝体。

当前最具影响⼒的 AI 产品，很多本质上都是 Harness：Claude Code（Anthropic）、Cursor、Codex CLI（OpenAI）、OpenCode，乃⾄ OpenClaw 这类更⾃主的 Agent 产品，⽆⼀例外。⼀个被反复验证的现象是：同⼀个底层模型在不同Harness中的表现可以相差数⼗个百分点以上⸺在许多 Agent 基准上，把 Claude Opus 从最⼩脚⼿架搬进 Claude Code 的完整 Harness，分数能从相差⼏⼗个百分点。⽽这正是HarnessEngineering的魅⼒所在：同⼀个模型，设计更好的Harness，能让它发挥出完全不同的能⼒⽔平。

⼀个成熟的Harness通常由这⼏层构成：分层组装的系统提⽰词、⼯具集（Tools）的定义与描述、上下⽂⼯程（压缩、检索、按需加载）、⼦Agent编排（Subagent/AgentTeams实现上下⽂隔离）、⽣命周期钩⼦（Hooks，把确定性逻辑下沉到代码层⽽⾮每次请求模型）、权限与沙箱（防注⼊、命令权限控制）、以及围绕“模型 ⼯具调⽤ 执⾏ 回灌”的主循环。看似只有60⾏的循环⻣架，复杂度全部沉淀在调优⾥⸺⼯具怎么切分、描述怎么写、上下⽂何时压缩、何时分发给⼦Agent。

随着 LLM 能⼒边界不断扩展，Harness Engineering 已成为 AI ⼯程领域的核⼼技能之⼀⸺ 模型层的差距正在收窄，Harness层的⼯程能⼒，正在决定⼀个AI产品最终是“能Demo”还是“能交付”。从 Claude Agent SDK 到 Codex SDK，⾏业正在从“调⽤ LLM API”转向“构建在 Harness 之上”，Harness Engineering 的重要性也在迅速提升。

# 任务：限制输⼊窗⼝的 LLM ⽂本分类任务

⼤语⾔模型（LLM）的出现使得“⽆需训练、直接推理”成为可能⸺通过在Prompt中提供少量带标签⽰例（few-shot），LLM可以在不更新任何参数的情况下完成新任务。如何利⽤LLM的语义理解性，在不改变权重的前提下，从少量带标签样本中快速“学习”并作出准确预测，⽤⽐较⼩参数的LLM在不损失很⼤性能的前提下替代传统的机器学习分类器，成为了⼀个重要的研究⽅向。

本次考核需要你设计⼀个基础的Harness⸺⼀个含有外部记忆管理、预算控制与推理Harness，使LLM在有限输⼊窗⼝（限制单轮输⼊Token数⼩于2048）的⽂本分类任务上达到尽可能⾼的准确率。

系统⾸先会依次将带标签的训练样本喂给你的Harness（ update ），你可以根据训练集更新Harness的记忆。训练流结束后，Harness对⽆标签测试⽂本进⾏预测（ predict ）。模型权重始终不变，所有“学习”发⽣在Harness维护的外部状态中。最终成绩将由分类任务正确率决定。

# 数据集说明

本地DEV集为客服意图分类，共77类。正式评测将在多个不同类型的任务上进⾏，涵盖不同领域的⽂本分类与⾃然语⾔理解任务，以考查Harness的泛化能⼒，因此请考⽣不要过拟合DEV集。

所有数据集均为JSONL格式，每⾏⼀条样本。训练集和测试集字段相同，测试集的 label 字段仅⽤于本地评测，正式评测时考⽣⽆法访问：

# 代码块

{"text": "I no longer have my phone.", "label": "lost_or_stolen_phone"}1

{"text": "My card is stuck in an ATM machine, how do I get it back quickly?","label": "card_swallowed"}

. text ：待分类的⾃然语⾔⽂本

• label ：类别标签字符串， predict() 的返回值须与其完全⼀致（exact match）

每个任务保证：测试集中出现的所有标签均在对应训练集中出现过。

# 模型说明

整个运⾏代码，均采⽤OpenAICompatible（OpenAI兼容）API的⻛格进⾏代码的调⽤。考试过程，LLMAPI需要考⽣⾃备（可以⽤公开平台的API服务，也可以⾃⾏⽤sglang/vllm进⾏部署）。为最⼩化考⽣成本，评分所⽤的模型为Qwen3-8B(Instruct)，且不开思考模式。

# ⽂件说明


代码块


```txt
1 solution.py 你唯一需要编辑的文件  
2 harness_base.py 一 Harness 基类（不可修改）  
3 llm_client.py 配置你的LLMAPI（修改顶部三行参数即可）  
4 run.py 本地调试脚本（默认4轮取均值）  
5 data/  
6 train_dev.json 一 DEV训练集（231条，77类，每类3条）  
7 test_dev.json 一 DEV验证集（539条，DEV集以及最终任务集保证test集中出现的标签都在train集中出现过)  
8 tokenizer/ 本地tokenizer（用于精确token计数）
```

# 快速开始

# 1.安装依赖


代码块


```batch
1 pip install -r requirements.txt
```

# 2. 配置 LLM API

编辑 llm_client.py 顶部三⾏，填⼊你的 API 信息：


代码块


1 BASE_URL  $\equiv$  "http://your-endpoint/v1"   
2 API_KEY  $\equiv$  "your-api-key"   
3 MODEL  $\equiv$  "your-model-name"

# 3. 实现你的 Harness

编辑 solution.py 中的 MyHarness 类，实现 update 和 predict ⽅法。

# 4.本地测试


代码块


```txt
1 python run.py #默认设置，与最终评测参数一致  
2 python run.py --runs 1 #快速单轮测试（最终评测将取4轮均值）  
3 python run.py --workers 50 #调整LLM并发数，防止因超时等导致错误
```

# 接⼝说明

```python
1 class MyHarness(Harness):   
2 def __init__(self, call_lli, count_tokens, count/messages_tokens, max_prompt_tokens: int):   
3 super().__init__(call_lli, count_tokens, count/messages_tokens, max_prompt_tokens)   
4   
5 def update(self, text: str, label: str) -> None:   
6 ""接收一条带标签的训练样本，更新内部记忆""   
7   
8 def predict(self, text: str) -> str:   
9 ""对文本预测标签，返回标签字符串""
```

# 基类提供以下注⼊接⼝：

<table><tr><td>属性</td><td>类型签名</td><td>说明</td></tr><tr><td>self.call_lli</td><td>(messages: list[dict]) -&gt; str</td><td>调用 LLM, 输入 OpenAI 格式 messages, 返回回复文本</td></tr><tr><td>self.count_tokens</td><td>(text: str) -&gt; int</td><td>计算单段文本的 token 数</td></tr><tr><td>self.count/messages_tokens</td><td>(messages: list[dict]) -&gt; int</td><td>计算 messages 列表的总 token 数 (只计算 content 总和, 不应用 chat_template, 与判题器判断是否需要截断一致)</td></tr><tr><td>self.max_prompt_tokens</td><td>int</td><td>每次调用的 prompt token 上限 (2048)</td></tr><tr><td>self.memory</td><td>list[tuple[str, str]]</td><td>存储 (text, label) 训练样本</td></tr></table>

token 管理：单次 call_llm 的 prompt 超过 max_prompt_tokens 时会被截断尾部并在stderr 打印警告。建议调⽤前⽤ count_messages_tokens 预先检查，主动控制 prompt ⻓度。

# 提交规则

1. Python⽂件：考⽣只需提交⼀个代码⽂件 solution.py，其中必须包含 MyHarness 类的完整实现。

◦ 只允许 import Python 标准库（re、math、random、collections 等）、numpy 和harness_base

◦ 禁⽌读写任何⽂件

◦ 禁⽌通过任何途径获取测试集标签（⼀经发现得分归零），禁⽌出现直接编码公开的相应测试集，并采⽤穷举法搜索官⽅正确答案（私有测试集也⽆法在开源数据中找到）；禁⽌任何情况的不正当的分与Hack⾏为。每个考⽣的代码均会经过内容复核，⼀经发现，该项考核按0分计算。

2. 探索报告：PDF⽂件，简易记录探索过程，包含不同Harness设计策略的尝试、效果和分析，作为主观分数的参考之⼀。

提交⽅式：（截⽌时间北京时间5⽉9号00:00，期间可以多次提交，会⾃动覆盖先前的提交⽂件）

1. 进⼊链接：https://send2me.cn/bLSuiHmE/StyTAqcKANgDLA

2. 精确填写个⼈信息，包括报名号、姓名（由填写错误导致的得分缺失，后果⾃负！）

3. FAQ⽹站：https://docs.qq.com/sheet/DUXRkd1BQcXJDWGp3?u=5965604e4f164981b50cfc104734afec，考⽣如果有需要向考官提问的问题，可以在该⻚⾯的“Question”列提出，考官会尽快给出解答。注意问题对所有⼈可⻅。

# 评分标准

本项考核的分数由两部分组成：

1. 客观得分（占总分 $8 0 \%$ ）

◦ 所有考⽣在私有集(每个任务的训练集和测试集格式与考⽣的DEV集完全相同，考⽣⽆法获取)上的加权平均准确率性能进⾏排名并赋分

◦ 私有测试集包含以下任务（每个任务的格式均和DEV集完全相同）：

▪ 与DEV集标签⼀致，⽂本不同的分类任务。（注：该部分测试集会含有较少⽐例的Prompt Injection 样本，请注意 Harness 的安全性设计）

OOD任务：若⼲个其他领域⽂本分类任务，内容、标签以及标签数量和DEV集完全不同。保证test集中出现的标签都在对应train集中出现过

▪ 复杂⾃然语⾔选择题任务：格式⼀致，但⽂本变为⾃然语⾔选择题，标签变为选项（如A/B/C/D），因此请考⽣不要针对⽂本分类任务设计过于特殊化的⽅案（如不使⽤LLM⽽设计了某种传统机器学习分类器），以免在该类任务上失去得分。

◦ 最终得分会在多个任务正确率上加权计算得到

◦ 测试统⼀使⽤Qwen3-8B (Instruct) ⾮思考模式模型，测试跑分的代码可以详⻅给定的 run.py

◦ 每个任务会进⾏多次采样（默认4次）取平均指标以保证结果稳定性

2. 提⽰词主观评价得分（占总分 $2 0 \%$ ）

◦ 由专家⽼师基于指定评价准则进⾏评分

◦ 评价内容包括Harness设计的创新性、合理性、可解释性等

注：正式评测时，会限制考⽣的任务执⾏时间（正常Harness设计不会超时），请考⽣不要进⾏恶意的⽆限轮调⽤LLM或⽤死循环卡住评测系统等⾏为，⼀经发现，该项考核按0分计算。