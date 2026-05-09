# Harness Engineering 文本分类方案

本仓库是 Harness Engineering 考核的本地实现与评测记录。任务形式是有限上下文窗口下的少样本文本分类：评测脚本先通过 `update(text, label)` 将带标签训练样本流式交给 Harness，再通过 `predict(text)` 对测试样本输出一个精确匹配的标签字符串。

根据考核说明，LLM 权重不更新，所有“学习”都发生在 Harness 维护的外部状态中。单次 `call_llm` 的 prompt token 上限为 2048，测试集标签只用于本地评测，正式评测时不可访问。

## 文件结构

| 路径 | 说明 |
| --- | --- |
| `solution.py` | 主要提交文件，包含 `MyHarness` 的完整实现 |
| `harness_base.py` | Harness 基类，定义 `update()` / `predict()` 接口 |
| `llm_client.py` | OpenAI-compatible LLM 客户端与 tokenizer 计数逻辑 |
| `run.py` | 官方本地单任务评测脚本 |
| `eval_all_benches.py` | 本仓库新增的全量评测脚本，遍历 `data` 与 `bench1`-`bench4` |
| `data/` | 官方 DEV 数据集 |
| `bench1/`-`bench4/` | 开源或补充 benchmark 数据集 |
| `results/` | 全量评测输出，包含 JSON 与 CSV 明细 |

## 数据集来源

- `data/`：官方提供的本地 DEV 数据集。
- `bench1/`：开源数据集，来源见 `bench1/source.txt`：<https://github.com/edgerunneres/2026-chuangzhi-academy-summer-camp-harness-engineering-mock-dataset>
- `bench2/`：开源数据集，来源见 `bench2/source.txt`：<https://github.com/leoguohr/open-harness-synthetic-benchmark/tree/main/open-harness-synthetic-benchmark>
- `bench3/`：补充数据集，仓库内未提供 GitHub 开源链接。
- `bench4/`：开源数据集，来源见 `bench4/source.txt`：<https://github.com/CoisiniStar/Harness_Dataset_SII2026Summer-Camp>

## 算法设计

`solution.py` 中的 `MyHarness` 是一个“检索增强 few-shot + 严格输出约束 + 确定性 fallback”的 Harness。核心目标是在 2048 prompt token 限制下，为每条测试样本挑选最有帮助的训练样例，并把 LLM 输出稳定映射回合法标签。

### 1. 外部记忆与延迟建索引

`update()` 只把 `(text, label)` 写入 `self.memory`，并标记索引失效。第一次 `predict()` 时才构建索引，避免每次新增样本后重复计算。索引构建使用线程锁保护，因此评测脚本并发调用 `predict()` 时不会重复建索引或读到半初始化状态。

### 2. 稀疏 TF-IDF 检索

索引同时维护两套特征：

- word 特征：用正则抽取英文、数字 token；
- char 特征：抽取 3-5 gram 字符 n-gram，并在文本两侧加入空格边界。

每个训练样本被编码为归一化稀疏 TF-IDF 向量。检索时，对 query 也构造同类向量，用 word/char 混合余弦相似度排序训练样本。字符 n-gram 能覆盖拼写变化、短文本和标签词变体；word 特征则保留更清晰的语义关键词。

### 3. 标签原型与标签名特征

除逐样本相似度外，算法还为每个 label 汇总训练样本向量，得到 label prototype。对于普通文本分类任务，还会把 label 字符串本身拆成词与字符 n-gram，计算 query 与 label name 的重合度。最终检索分数可由三部分组成：

- query 与单个训练样本的相似度；
- query 与该样本所属 label prototype 的相似度；
- query 与 label name 的词面相似度。

这些权重不是固定拍脑袋设定，而是在训练集上做轻量 leave-one-out 自调参。

### 4. 任务类型检测与 MCQ 模式

如果全部标签都是较短的字母数字串，且类别数不多，Harness 会自动进入 MCQ 模式。例如 `A/B/C/D` 或 `0/1/2/3` 这类选择题标签。MCQ 模式下：

- 不使用 label name 特征，因为 `A`、`B` 这类标签本身没有语义；
- 保留全部候选答案；
- few-shot 每类最多保留更多样例；
- prompt 强调只输出一个选项 token；
- 输出解析采用更严格的短标签解析逻辑。

### 5. 训练集自调参

构建索引时，`_auto_tune_retrieval()` 会在训练集上做小型网格搜索，比较不同 `word_weight`、`proto_mix`、`label_name_mix` 的 leave-one-out 效果。评分综合考虑：

- 最近邻 top-1 是否同类；
- top-5 是否包含正确类别；
- 被选入 few-shot 的样例中是否覆盖正确类别；
- 纯检索 fallback 是否能预测正确。

随后还会选择 fallback 聚合时使用的 top-N 范围，让检索失败兜底更稳。

### 6. 多样化 few-shot 选择与 token 预算

对每条 query，Harness 先按检索分数得到候选训练样本，再进行类别多样化选择：

- 普通任务默认最多选 24 条；
- 每个类别默认最多 2 条，避免一个高频近邻类别挤占全部上下文；
- MCQ 模式每类上限提高到 6 条。

随后 `_fit_to_budget()` 用 `count_messages_tokens()` 检查 prompt 长度。如果超过 `max_prompt_tokens - SAFETY_MARGIN`，就逐步减少示例数量，直到满足预算。单条 query 与示例文本也有字符级截断，降低长文本把 prompt 撑爆的风险。

### 7. Prompt 与安全约束

普通分类 prompt 会列出 allowed labels，并明确要求只输出一个完全一致的类别字符串。示例采用 user/assistant 对话块提供，输入文本和示例内容都被声明为 data，而不是 instructions，以降低 prompt injection 样本影响。

MCQ prompt 则更短更硬：只允许输出候选答案 token，不允许解释、Markdown、前缀或标点。

### 8. 易混标签提示

对于普通分类任务，索引构建时会扫描训练集中的 near-miss：如果某样本最近的异类邻居分数接近同类邻居，就把两个 label 记为易混对。预测时，只有当前 query 的 top-2 候选 label 正好构成易混对且分差很小时，才额外注入一条区分性关键词提示。这样只在高歧义样本上增加信息，不干扰大多数简单样本。

### 9. 输出解析与 fallback

LLM 输出会先去除 `<think>...</think>`、引号、Markdown 包裹和 `Answer:` / `Label:` 等前缀。普通任务按以下顺序解析：

1. 完全匹配合法 label；
2. 大小写、空格、连字符归一化后匹配；
3. 在输出文本中查找合法 label；
4. 若仍失败，回退到检索聚合预测。

MCQ 模式使用专门的短 token 解析器，尽量避免把解释文本中的无关字母误判为选项。LLM 调用失败时会重试，最终仍失败则使用确定性检索 fallback。

## 运行方式

安装依赖：

```powershell
pip install -r requirements.txt
```

单任务官方评测：

```powershell
py -3.11 run.py --workers 25 --runs 2
```

全量评测：

```powershell
py -3.11 eval_all_benches.py
```

只查看会评测哪些任务：

```powershell
py -3.11 eval_all_benches.py --list-only
```

## 评测设置

以下结果来自 `results/eval_all_20260509_003512.*`：

- 评测时间：2026-05-09 00:11:01 至 00:35:12，Asia/Shanghai；
- `workers=25`；
- `runs=2`；
- `max_prompt_tokens=2048`；
- 模型配置来自 `llm_client.py`，使用 OpenAI-compatible 接口调用 `qwen3-8b`，并关闭 thinking；
- 指标为 accuracy，任务表中的平均值是 2 次 run 的算术平均。

### data

| 任务 | Train | Test | Run 1 | Run 2 | 平均 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `data/dev` | 231 | 539 | 85.34% | 85.16% | 85.25% |

组内结果：micro 85.25%，macro 85.25%。

### bench1

| 任务 | Train | Test | Run 1 | Run 2 | 平均 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `bench1/task1` | 300 | 685 | 77.23% | 77.08% | 77.15% |
| `bench1/task2_education` | 116 | 249 | 97.59% | 97.59% | 97.59% |
| `bench1/task2_restaurant` | 116 | 249 | 97.99% | 97.99% | 97.99% |
| `bench1/task2_techsupport` | 140 | 299 | 81.61% | 81.61% | 81.61% |
| `bench1/task3` | 287 | 383 | 71.80% | 72.06% | 71.93% |

组内结果：micro 82.31%，macro 85.25%。

### bench2

| 任务 | Train | Test | Run 1 | Run 2 | 平均 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `bench2/task1_support_intent` | 36 | 72 | 98.61% | 98.61% | 98.61% |
| `bench2/task2_campus_routing` | 30 | 60 | 100.00% | 100.00% | 100.00% |
| `bench2/task3_choice_reasoning` | 80 | 80 | 66.25% | 66.25% | 66.25% |

组内结果：micro 86.79%，macro 88.29%。

### bench3

| 任务 | Train | Test | Run 1 | Run 2 | 平均 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `bench3/train_dev` | 231 | 539 | 85.34% | 85.53% | 85.44% |
| `bench3/train_mcq` | 96 | 384 | 80.73% | 80.47% | 80.60% |
| `bench3/train_ood` | 186 | 600 | 83.67% | 83.67% | 83.67% |

组内结果：micro 83.52%，macro 83.23%。

### bench4

| 任务 | Train | Test | Run 1 | Run 2 | 平均 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `bench4/ecommerce` | 150 | 300 | 97.67% | 97.67% | 97.67% |
| `bench4/finance` | 150 | 300 | 99.33% | 99.33% | 99.33% |
| `bench4/medical_triage` | 150 | 300 | 97.67% | 97.67% | 97.67% |
| `bench4/news_topic` | 150 | 300 | 95.33% | 95.33% | 95.33% |
| `bench4/tech_support` | 150 | 300 | 98.33% | 98.33% | 98.33% |

组内结果：micro 97.67%，macro 97.67%。

### 总体结果

| 范围 | 任务数 | Correct / Total | Micro Accuracy | Macro Task Accuracy |
| --- | ---: | ---: | ---: | ---: |
| `data` + `bench1`-`bench4` | 17 | 9831 / 11278 | 87.17% | 89.08% |

其中 micro accuracy 按所有样本汇总计算，macro task accuracy 先计算每个任务的 2-run 平均 accuracy，再对 17 个任务取平均。

