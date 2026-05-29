# 基于拉丁方设计的 LangGraph 智能体评估与方差分析系统 (LLM-Agent-Evaluation-Latin-Square)

## 📌 项目简介

本项目是一个将**大模型智能体工程 (Agent Engineering)** 与 **严谨统计学实验设计 (Experimental Design)** 深度结合的自动化评估框架。

在评估大语言模型智能体（如 Tool-calling 或 ReAct 搜索助手）时，其最终表现往往受到多种交织因素的影响，例如：**智能体自身的配置（如温度 Temperature）**、**测试任务的难度** 以及 **评测的样本数量**。为了从复杂的混淆变量中精准剥离出“智能体配置”的真实主效应，本项目创新性地引入了统计学中的 **$3 \\\\times 3$ 拉丁方设计 (Latin Square Design)**，对基于 LangGraph 构建的智能体在 Berkeley Function Calling Leaderboard (BFCL) 评测集上的表现进行多因素方差分析 (ANOVA)。

---

## 🚀 核心特性

1. **LangGraph 智能体工作流**：构建具备状态管理与多节点控制的 ReAct 风格智能体，支持动态工具调用（Tool Calling），并集成 Tavily API 实现真实的互联网信息检索。
2. **BFCL 官方评测集成**：完美适配 Berkeley Function Calling Leaderboard (BFCL) 数据集与评估接口，覆盖 `simple_python`, `multiple_function`, `parallel_function` 等多种函数调用复杂度。
3. **自动化拉丁方实验矩阵**：通过代码自动调度 $3 \\\\times 3$ 实验矩阵（行：智能体配置，列：任务难度，单元格：样本数量），最大化利用实验样本并有效控制双向区组效应（无交互作用前提下）。
4. **统计学方差分析 (ANOVA)**：集成 Python `statsmodels` 库，实验结束后自动生成标准的方差分析表，输出平方和 (Sum of Squares)、自由度 (df)、F 检验统计量及 $p$ 值，为模型超参数调优提供具备统计显著性的量化依据。

---

## 📊 统计学原理与模型

本实验采用标准的 $3 \\\\times 3$ 拉丁方设计。假设各因子之间不存在交互作用，其理论统计学线性模型为：

$$Y_{ijk} = \\\\mu + \\\\alpha_i + \\\\beta_j + \\\\tau_k + \\\\epsilon_{ijk}$$

其中：
- $Y_{ijk}$：响应变量，即智能体在特定配置下的 BFCL 准确率 (Accuracy)。
- $\\\\mu$：总体均值。
- $\\\\alpha_i$：行效应，代表 **智能体配置 (Agent Configuration)** 分类因子（例如：低、中、高三种 Temperature 设定）。
- $\\\\beta_j$：列效应，代表 **任务难度 (Task Difficulty)** 分类因子（对应 `simple_python`, `multiple_function`, `parallel_function`）。
- $\\\\tau_k$：处理效应，代表 **样本数量 (Sample Size)** 分类因子（例如：5, 10, 15 个样本）。
- $\\\\epsilon_{ijk}$：随机误差项，假设服从独立同分布的正态分布 $N(0, \\\\sigma^2)$。

---

## 📂 项目结构

```text
agent-eval-latin-square/
│
├── src/
│   ├── __init__.py
│   ├── agent_workflow.py    # 基于 LangGraph 与 Tavily API 的智能体工作流定义
│   ├── eval_pipeline.py     # 适配 BFCL 接口的评估流水线与拉丁方矩阵控制
│   └── stats_analysis.py    # 基于 statsmodels 的多因素方差分析与数据可视化
│
├── .env.example             # 环境变量配置模板
├── .gitignore               # Git 忽略文件配置
├── requirements.txt         # 项目依赖项列表
└── README.md                # 项目说明文档
```


## 🛠️ 环境准备

### 1. 依赖安装

建议在 Python 3.10+ 环境下运行，执行以下命令安装核心依赖：

```bash
pip install langgraph==1.0.0a3 langchain_openai==0.3.33 python-dotenv tavily-python pandas statsmodels
pip install hello-agents[evaluation]==0.2.3

```

### 2. 克隆 BFCL 数据集仓库

首次运行前，请在项目同级目录下克隆官方的 Gorilla/BFCL 仓库作为数据源：

```bash
git clone [https://github.com/ShishirPatil/gorilla.git](https://github.com/ShishirPatil/gorilla.git) temp_gorilla

```

### 3. 环境变量配置

在项目根目录下创建 `.env` 文件，并填写您的 API 密钥：

```env
OPENAI_API_KEY="your_openai_api_key"
LLM_MODEL_ID="gpt-4o-mini"
LLM_BASE_URL="[https://api.openai.com/v1](https://api.openai.com/v1)"
TAVILY_API_KEY="your_tavily_api_key"
HF_TOKEN="your_huggingface_token"

```

---

## 💻 运行与使用

直接执行主流水线脚本，系统将依次运行 9 次拉丁方组合实验，收集数据并自动打印方差分析表：

```bash
python src/eval_pipeline.py

```

---

## 📈 实验输出示例

运行结束后，系统将为您输出如下形式的 **双向区组设计 (拉丁方) 方差分析表 (ANOVA Table)**：

```text
============================================================
🔬 拉丁方设计的方差分析表 (ANOVA):
============================================================
                    sum_sq     df          F    PR(>F)
C(Agent)          0.145800    2.0   4.253000  0.190342
C(Difficulty)     0.428450    2.0  12.497500  0.074088
C(SampleSize)     0.031250    2.0   0.911600  0.523114
Residual          0.034283    2.0        NaN       NaN
------------------------------------------------------------
💡 统计学结论提示：
- 若 PR(>F) < 0.05，说明该因子对智能体表现具有显著影响。
- 注：标准 3x3 拉丁方若无重复测量，残差自由度极低（df=2），建议在生产环境增加多轮正交重复试验以获得更稳健的检验效力。

```

---

## 🛠️ 技术栈与规范

* **LangGraph**：精细化控制大模型工具调用与推理状态。
* **Statsmodels**：提供严谨的普通最小二乘法 (OLS) 线性回归与 ANOVA 统计检验。
* **Git 规范**：已在 `.gitignore` 中严格配置隐私保护，避免任何 `.env` 敏感 API 密钥泄露。

