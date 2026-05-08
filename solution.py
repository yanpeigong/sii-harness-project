"""
solution.py — 考生唯一需要提交的文件

规则
----
1. 只能修改 MyHarness 类内部；其余部分不可改动。考生可以先行查看 harness_base.py 以了解可用接口和调用约定。
2. 只允许 import Python 标准库（re, math, random, json, collections 等）、numpy
   以及 harness_base（已提供）。
3. 禁止 import 其他第三方库（openai, sklearn, torch …）。
4. 禁止通过任何途径读写磁盘文件。
5. call_llm 每次调用的 prompt token 数若超过 max_prompt_tokens，
   会被自动截断至预算上限后再发送，
   可用 count_tokens（计算单条消息的 token 数） 和 count_messages_tokens（计算消息列表的总 token 数）预先控制 prompt 长度。
6. predict() 只接收 text，任何绕过接口获取 label 的行为将导致得分归零。
"""

from harness_base import Harness

# ============================================================
# 考生实现区（考生只能修改 MyHarness 类里的内容）
# ============================================================
class MyHarness(Harness):
    def __init__(self, call_llm, count_tokens, count_messages_tokens, max_prompt_tokens: int):
        super().__init__(call_llm, count_tokens, count_messages_tokens, max_prompt_tokens)
        # 若设计额外数据结构，请在下方初始化

    def update(self, text: str, label: str) -> None:
        # 替换为你设计的记忆更新逻辑
        super().update(text, label)

    def predict(self, text: str) -> str:
        # 可以在这里使用 call_llm 与 LLM 交互，在 test 集上做标签预测
        # 本方法要求返回的字符串即为预测的标签，请考生自行做好答案提取，避免因输出不规范导致的扣分
        # 示例：（请替换为你设计的方案）
        response = self.call_llm([
            {"role": "user",
             "content": f"Classify the following text. "
                         f"Respond with only the label, nothing else.\n\nText: {text}\nLabel:"}
        ])
        return response.strip()

    # 如需要，可以设计其他辅助方法
