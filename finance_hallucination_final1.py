# finance_hallucination_final.py
# 基准：Qwen + DeepSeek
# 裁判：火山引擎豆包
# 输出：幻觉率 + 拒绝率（单独柱状图） + 所有图表
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from openai import OpenAI

# ===================== 路径 =====================
BASE_DIR = r"D:\project"
DATA_PATH = r"D:\project\data\processed\finance_100.csv"
SAVE_DIR = r"D:\project\results"
os.makedirs(SAVE_DIR, exist_ok=True)

# ===================== 模型 =====================
qwen_client = OpenAI(
    api_key="sk-1840e3229c1849c3a2320c6331b9368e",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)
QWEN_MODEL = "qwen-turbo"

deepseek_client = OpenAI(
    base_url="https://ark.cn-beijing.volces.com/api/v3",
    api_key="61e8e9f2-c7df-4fd7-b975-3f502ef37256"
)
DEEPSEEK_MODEL = "deepseek-v3-2-251201"

judge_client = OpenAI(
    base_url="https://ark.cn-beijing.volces.com/api/v3",
    api_key="73dc561f-2113-44fc-8895-9044494a8cea",
)
JUDGE_MODEL = "doubao-seed-2-0-pro-260215"

# ===================== 加载数据 =====================
df = pd.read_csv(DATA_PATH)
df = df.head(10)
QUESTION_COL = "question"
settings = ["base", "detailed", "fewshot", "self_reflection"]
total_samples = len(df)

# ===================== 你指定的提示词 =====================
def get_prompts(question):
    return {
        "base": f"Question: {question}\nAnswer:",
        
        "detailed": f"""You are a professional financial analyst.
Answer the question ACCURATELY, based on real-world financial knowledge.
DO NOT make up fake data, facts, or numbers.
Question: {question}
Answer:""",
        
        "fewshot": f"""Example 1:
Question: What is the Federal Funds Rate?
Answer: The Federal Funds Rate is the interest rate at which depository institutions lend money to each other overnight.

Example 2:
Question: What is a bond?
Answer: A bond is a fixed-income instrument that represents a loan made by an investor to a borrower.

Now answer the following question accurately:
Question: {question}
Answer:""",
        
        "self_reflection": f"""Step 1: Answer the financial question.
Step 2: Check your answer for factual errors.
Step 3: Correct any hallucinations (fake facts, numbers, terms).
Question: {question}
Final Answer:"""
    }

# ===================== 生成函数 =====================
def generate(client, model, prompt):
    try:
        resp = client.chat.completions.create(
            model=model, messages=[{"role":"user","content":prompt}],
            temperature=0.1, max_tokens=300
        )
        return resp.choices[0].message.content.strip()
    except:
        return "ERROR"

# ===================== 生成回答 =====================
for s in settings:
    tqdm.pandas(desc=f"Qwen生成 {s}")
    df[f"qwen_resp_{s}"] = df[QUESTION_COL].progress_apply(
        lambda q: generate(qwen_client, QWEN_MODEL, get_prompts(q)[s])
    )

for s in settings:
    tqdm.pandas(desc=f"DeepSeek生成 {s}")
    df[f"deepseek_resp_{s}"] = df[QUESTION_COL].progress_apply(
        lambda q: generate(deepseek_client, DEEPSEEK_MODEL, get_prompts(q)[s])
    )

# ===================== 裁判：同时判断幻觉 + 拒绝 =====================
def judge_with_refusal(question, answer):
    prompt = f"""You need to judge two things:
1. Is this answer a hallucination?
2. Did the model REFUSE to answer (say "I don't know", "cannot answer", "no comment", etc.)

Output ONLY TWO numbers separated by space:
First number: 1=hallucination, 0=no hallucination
Second number: 1=refused to answer, 0=answered normally

Question: {question}
Answer: {answer}
Output two numbers only:"""

    try:
        resp = judge_client.responses.create(
            model=JUDGE_MODEL,
            input=[{"role":"user","content":[{"type":"input_text","text":prompt}]}]
        )
        out = resp.output_text.strip()
        parts = out.split()
        hal = int(parts[0]) if parts[0] in ["0","1"] else 1
        ref = int(parts[1]) if parts[1] in ["0","1"] else 0
        return hal, ref
    except:
        return 1, 0

# ===================== 评测 Qwen =====================
for s in settings:
    tqdm.pandas(desc=f"裁判 Qwen {s}")
    def judge_row(row):
        h, r = judge_with_refusal(row[QUESTION_COL], row[f"qwen_resp_{s}"])
        return pd.Series([h, r])
    df[["judge_qwen_"+s, "refuse_qwen_"+s]] = df.apply(judge_row, axis=1)

# ===================== 评测 DeepSeek =====================
for s in settings:
    tqdm.pandas(desc=f"裁判 DeepSeek {s}")
    def judge_row(row):
        h, r = judge_with_refusal(row[QUESTION_COL], row[f"deepseek_resp_{s}"])
        return pd.Series([h, r])
    df[["judge_deepseek_"+s, "refuse_deepseek_"+s]] = df.apply(judge_row, axis=1)

# ===================== 计算指标 =====================
def calc_metrics(df, prefix):
    rate = {}
    refuse = {}
    for s in settings:
        total = len(df)
        hal = df[f"judge_{prefix}_{s}"].sum()
        ref = df[f"refuse_{prefix}_{s}"].sum()
        rate[s] = round(hal/total*100, 2)
        refuse[s] = round(ref/total*100, 2)
    return rate, refuse

qwen_rate, qwen_refuse = calc_metrics(df, "qwen")
deepseek_rate, deepseek_refuse = calc_metrics(df, "deepseek")

# ===================== 控制台输出 =====================
print("\n" + "="*75)
print("             📊 实验结果（幻觉率 + 拒绝率）")
print("="*75)
print(f"{'Prompt':<20} {'Qwen幻觉率':<12} {'Qwen拒绝率':<12} {'DeepSeek幻觉率':<12} {'DeepSeek拒绝率':<12}")
print("-"*75)
for k in settings:
    print(f"{k:<20} {str(qwen_rate[k])+'%':<12} {str(qwen_refuse[k])+'%':<12} {str(deepseek_rate[k])+'%':<12} {str(deepseek_refuse[k])+'%':<12}")
print("-"*75)

# ===================== 1. 幻觉率表格图 =====================
# fig, ax = plt.subplots(figsize=(9, 3.5))
# ax.axis('tight')
# ax.axis('off')
# rows_hall = []
# for k in settings:
#     rows_hall.append([k, f"{qwen_rate[k]}%", f"{deepseek_rate[k]}%"])

# table1 = ax.table(
#     cellText=rows_hall,
#     colLabels=["Prompt", "Qwen Hallucination", "DeepSeek Hallucination"],
#     cellLoc="center", loc="center"
# )
# table1.auto_set_font_size(False)
# table1.set_fontsize(12)
# table1.scale(1, 2)
# plt.title("Hallucination Rate Comparison", pad=20, fontsize=14)
# plt.savefig(os.path.join(SAVE_DIR, "hallucination_table.png"), dpi=300, bbox_inches="tight")
# plt.close()

# ===================== 2. Qwen 幻觉率图 =====================
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.figure(figsize=(10,5))
bars = plt.bar(qwen_rate.keys(), qwen_rate.values(), color='#ff9999')
for b in bars:
    plt.text(b.get_x()+b.get_width()/2., b.get_height()+1, f'{b.get_height()}%', ha='center', fontsize=12)
plt.title("Qwen 幻觉率", fontsize=14)
plt.ylabel("幻觉率 (%)", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "qwen_result.png"), dpi=300)
plt.close()

# ===================== 3. DeepSeek 幻觉率图 =====================
plt.figure(figsize=(10,5))
bars = plt.bar(deepseek_rate.keys(), deepseek_rate.values(), color='#66b3ff')
for b in bars:
    plt.text(b.get_x()+b.get_width()/2., b.get_height()+1, f'{b.get_height()}%', ha='center', fontsize=12)
plt.title("DeepSeek 幻觉率", fontsize=14)
plt.ylabel("幻觉率 (%)", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "deepseek_result.png"), dpi=300)
plt.close()

# ===================== 4. 合并幻觉率对比图 =====================
plt.figure(figsize=(12,6))
x = np.arange(len(settings))
w = 0.35
plt.bar(x-w/2, qwen_rate.values(), w, label='Qwen', color='#ff9999')
plt.bar(x+w/2, deepseek_rate.values(), w, label='DeepSeek', color='#66b3ff')
plt.title('Qwen vs DeepSeek 幻觉率对比', fontsize=14)
plt.ylabel('幻觉率 (%)', fontsize=12)
plt.xticks(x, settings)
plt.legend()
for i, v in enumerate(qwen_rate.values()): plt.text(i-w/2, v+1, f'{v}%', ha='center')
for i, v in enumerate(deepseek_rate.values()): plt.text(i+w/2, v+1, f'{v}%', ha='center')
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "comparison_all.png"), dpi=300)
plt.close()

# ===================== 5. 幻觉频数图 =====================
qwen_cnt = {k: int(df[f"judge_qwen_{k}"].sum()) for k in settings}
deepseek_cnt = {k: int(df[f"judge_deepseek_{k}"].sum()) for k in settings}

plt.figure(figsize=(12,6))
plt.bar(x-w/2, qwen_cnt.values(), w, label="Qwen", color="#ff9999")
plt.bar(x+w/2, deepseek_cnt.values(), w, label="DeepSeek", color="#66b3ff")
for i, v in enumerate(qwen_cnt.values()): plt.text(i-w/2, v+0.3, f"{v}", ha="center")
for i, v in enumerate(deepseek_cnt.values()): plt.text(i+w/2, v+0.3, f"{v}", ha="center")
plt.title("幻觉样本数对比", fontsize=14)
plt.ylabel("数量", fontsize=12)
plt.xticks(x, settings)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "hallucination_count.png"), dpi=300)
plt.close()

# ===================== ✅ 6. 拒绝率单独柱状图（完全复刻你给的样式）=====================
plt.figure(figsize=(12, 6))
x = np.arange(len(settings))
w = 0.35

# 配色完全对齐你给的图
qwen_color = "#f08080"  # 珊瑚红（Qwen）
deepseek_color = "#73b6e6"  # 天蓝色（DeepSeek）

# 绘制柱状图
plt.bar(x - w/2, list(qwen_refuse.values()), w, label='Qwen', color=qwen_color)
plt.bar(x + w/2, list(deepseek_refuse.values()), w, label='DeepSeek', color=deepseek_color)

# 标注百分比
for i, v in enumerate(qwen_refuse.values()):
    plt.text(i - w/2, v + 0.2, f"{v}%", ha='center', fontsize=14)
for i, v in enumerate(deepseek_refuse.values()):
    plt.text(i + w/2, v + 0.2, f"{v}%", ha='center', fontsize=14)

# 标题、坐标轴、图例完全对齐
plt.title(f"Refusal Rate Comparison ({total_samples} samples, DouBao Judge)", fontsize=16, fontweight='bold', pad=20)
plt.ylabel("Refusal Rate (%)", fontsize=14)
plt.xticks(x, ["Base", "Detailed", "Few-shot", "Self-reflection"], fontsize=14)
plt.legend(fontsize=14)
plt.ylim(0, 10)  # 固定y轴0-10，和你给的图一致
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.tight_layout()

# 保存到results文件夹
plt.savefig(os.path.join(SAVE_DIR, "refusal_rate_bar.png"), dpi=300, bbox_inches="tight")
plt.close()

# ===================== 保存CSV =====================
df.to_csv(os.path.join(SAVE_DIR, "final_result1.csv"), index=False, encoding="utf-8-sig")

print("\n✅ 全部完成！")
print("✅ 拒绝率单独柱状图已生成：refusal_rate_bar.png")
print("✅ 所有图表保存在：D:\\project\\results")