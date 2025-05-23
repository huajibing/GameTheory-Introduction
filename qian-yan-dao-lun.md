# 前言/导论

## 1. 什么是博弈论？

想象一下市场上的公司竞争定价、政治候选人制定选举策略、甚至是你和朋友决定晚上去哪里吃饭——这些场景的共同点是什么？它们都涉及多个决策者，每个人的选择都会影响到其他人，并且每个人都在试图做出对自己最有利的决定。**博弈论 (Game Theory)** 正是研究这种**理性决策者之间策略互动**的数学理论。

在博弈中，核心挑战在于你的最优决策往往取决于你**预期**其他人会怎么做，而其他人也在做同样的预期。这与传统的单人优化问题（比如如何最小化成本或最大化利润，而不直接考虑竞争对手的反应）有着本质的不同。博弈论关注的是这种相互依存的决策环境，分析在这种互动中可能出现的**均衡状态**以及如何达到这种状态。

博弈论的思维方式和分析工具已经渗透到众多领域：

* **经济学**：分析寡头垄断市场、拍卖机制设计、贸易谈判、契约理论等。
* **商业管理**：制定企业竞争战略、市场进入/退出决策、供应链管理、广告策略等。
* **政治学**：理解选举竞争、国际关系、联盟形成、立法博弈等。
* **生物学**：解释进化稳定策略（ESS）、动物的合作与冲突行为、种群动态等。
* **计算机科学**：设计多智能体系统、构建鲁棒的网络协议、算法博弈论、人工智能（如对抗生成网络 GANs）等。
* **社会学与哲学**：探讨社会规范的形成、信任与合作的机制、伦理决策等。

本教程旨在引导你深入理解博弈论中的几种**核心均衡概念**，从最著名的**纳什均衡**出发，逐步探索**子博弈精炼纳什均衡**、**相关均衡**、**贝叶斯纳什均衡**，并最终涉足**合作博弈**中的解概念，如**核心**和**夏普利值**。我们将不仅解释这些概念“是什么”，更会侧重于“如何求解”，并通过大量的案例分析，帮助你掌握将理论应用于实践的技能。

**学习建议：**

1. **概念优先：** 先透彻理解每个概念的定义和直觉含义，再深入学习其求解方法。
2. **动手实践：** 尝试独立完成每章的练习题，这是检验理解和巩固知识的最佳方式。
3. **联系现实：** 思考如何将学到的博弈模型和均衡概念应用于你所熟悉的生活、学习或工作场景。
4. **循序渐进，时常回顾：** 博弈论的概念是层层递进的，确保理解前一章内容是学习后续章节的基础。定期回顾有助于构建完整的知识体系。

## 2. 博弈的基本要素

要精确地描述和分析一个策略互动场景，我们需要定义博弈的几个基本构成要素：

**1. 参与人 (Players)**：\
参与博弈并做出决策的主体。可以是个人、公司、国家、团队，甚至是生物种群或算法。我们通常假设参与人是**理性的 (rational)**，即他们有明确的目标（通常是最大化自身的某种收益）并会选择最有助于实现该目标的行动。在模型中常用 P1, P2 或 A, B 等表示。

**2. 策略 (Strategies)**：\
参与人可以选择的**完整行动计划**。一个策略规定了参与人在博弈中可能遇到的每一种情况下应该如何行动。

* **纯策略 (Pure Strategy)**：明确选择一个特定的行动。
* **混合策略 (Mixed Strategy)**：以某种概率分布随机地选择不同的纯策略。  \
  策略的集合可以是有限的（如“合作”或“背叛”），也可以是无限的（如选择一个价格或投资额）。

**3. 支付/收益 (Payoffs)**：\
参与人从博弈的某个可能结局中获得的效用或回报。支付通常用**数值**表示，代表利润、满意度、市场份额、甚至是适应度（生物学）或负数值（如成本、刑期）。**支付函数 (Payoff Function)** 将所有参与人的策略组合映射到每个参与人获得的具体支付值。例如，`u_i(s_1, s_2, ..., s_n)` 表示在策略组合 `(s_1, ..., s_n)` 下，参与人 `i` 获得的支付。

**4. 信息 (Information)**：\
参与人在做决策时所拥有的知识。这包括对博弈规则、其他参与人的身份和数量、他们可能的策略、他们的支付函数以及（在动态博弈中）他们已经采取的行动的了解程度。

根据这些要素，特别是策略选择的时间和信息的结构，我们可以将博弈进行分类：

* **合作博弈 (Cooperative Games) vs 非合作博弈 (Non-cooperative Games)**：
  * **非合作博弈**：研究的核心是个体理性决策。参与人独立选择策略以最大化自身利益，不允许形成有约束力的协议（除非协议本身是自发强制执行的，即构成均衡）。**本教程主要关注非合作博弈。**
  * **合作博弈**：允许参与人形成具有约束力的联盟，并就如何分配联盟产生的总收益进行谈判。研究的核心是联盟的稳定性和收益的公平分配。
* **静态博弈 (Static Games) vs 动态博弈 (Dynamic Games)**：
  * **静态博弈**（又称同时行动博弈）：所有参与人同时做出决策，或者在不知道其他人选择的情况下做出决策。常用**策略式表述（矩阵）**。
  * **动态博弈**（又称序贯行动博弈）：参与人按照特定的顺序行动，后行动者可能观察到先行动者的选择。常用**扩展式表述（博弈树）**。
* **完全信息 (Complete Information) vs 不完全信息 (Incomplete Information)**：
  * **完全信息博弈**：每个参与人都**完全**了解所有其他参与人的**支付函数**（即知道别人的目标和偏好）。
  * **不完全信息博弈**（又称贝叶斯博弈）：至少有一个参与人不确定其他某个（或某些）参与人的支付函数。参与人可能只知道对方属于几种可能的“类型”(Type)之一，以及这些类型的概率分布。
* **完美信息 (Perfect Information) vs 不完美信息 (Imperfect Information)**：
  * **完美信息博弈**：在动态博弈中，每个参与人在轮到自己行动时，**完全**知道博弈到目前为止发生的所有历史行动。例如，国际象棋。
  * **不完美信息博弈**：在动态博弈中，至少有一个参与人在其决策时点，不完全清楚之前发生的某些行动（可能是其他参与人的，也可能是“自然”的随机选择）。例如，扑克牌（不知道对手的底牌）或同时出拳的石头剪刀布。

本教程将从最基础的**静态、完全信息、非合作博弈**入手，逐步引入动态性、不完全信息等复杂因素，最后再介绍合作博弈。
