# 双色球概率分布模型 V3

V3 把每期开奖明确写成一个**概率生成过程**，目标不是猜一个确定答案，而是学习“下一期每种结果出现的概率分布”以及参数的不确定性。

## 和 V1 的关系

想法部分一致，但并非同一个模型：

- V1 计算每个号码的平滑历史频率，再把这组点估计直接用于加权抽样；
- V3 为 33 个红球和 16 个蓝球分别维护 Dirichlet 后验，不只保留一个概率值，还保留概率参数本身的不确定性；
- 每次预测先从后验中抽一组潜在权重，再由该权重生成一期号码，因此输出来自完整的后验预测分布；
- V3 用组合的对数概率调参和回测，而不是只把各号码当成互相独立的二分类事件。

## 生成模型

蓝球服从 16 类分类分布，Dirichlet 先验与分类分布共轭。红球不能重复，因此使用 Plackett–Luce 逐次加权无放回分布：从 33 个正权重号码中连续抽取 6 个，抽中过的号码从候选池移除。程序用动态规划精确计算某个**无序红球集合**的概率。

历史观测使用指数衰减，较新的期次对后验参数影响更大。半衰期和先验强度只用测试期之前的验证区间选择，最后 20% 数据保持为独立时间测试集。模型还进行后验预测检查，对比模拟开奖与真实近期开奖的和值、跨度和奇数个数均值。

严格来说，红球的 Plackett–Luce 似然并不与 Dirichlet 共轭。V3 的“衰减入选次数 + Dirichlet”更新是一个可解释、计算量适中的幂后验近似；蓝球部分则是标准共轭更新。这项限制会明确保存在模型文件中。

## 当前回测结果

使用 `2003001`～`2026119` 共 3516 期数据，最后 704 期作为独立测试集：

- 红球集合平均负对数概率为 `13.917118`，均匀分布基线为 `13.917677`，优势极小；
- 后验均值 Top-6 平均命中 `1.135` 个红球，均匀随机理论期望为 `1.091` 个；
- 蓝球 Top-1 命中率为 `7.39%`，但平均负对数概率 `2.774649` 劣于均匀基线 `2.772589`；
- 后验预测模拟的和值、跨度、奇数个数均值分别为 `101.106`、`24.262`、`3.029`，最近 500 期实测值为 `100.078`、`24.508`、`3.026`。

这组结果说明生成分布能复现若干总体形态，但没有显示出足以支持稳定预测的概率优势。尤其蓝球的命中率看似较高，对数概率却更差，说明只看 Top-1 命中很容易产生误导。

## 使用方式

要求 Python 3.10 或更高版本，无第三方运行时依赖。在 `v3` 目录执行：

```powershell
python -m pip install -e .
python -m union_lotto_v3 pipeline
```

也可分步运行：

```powershell
python -m union_lotto_v3 fetch
python -m union_lotto_v3 validate
python -m union_lotto_v3 train
python -m union_lotto_v3 predict --tickets 5 --seed 2026 --simulations 10000
```

离线使用仓库内已有数据时，可直接跳过 `fetch`。输出包括：

- `artifacts/model.json`：后验参数、超参数、验证/测试指标与后验预测检查；
- `predictions/latest.json`：候选组合、红球后验入选概率、蓝球后验权重和 95% 区间；
- `data/ssq_history.csv`：标准化历史开奖数据。

运行测试：

```powershell
$env:PYTHONPATH = "src"
python -m unittest discover -s tests -v
```

## 参考

- [Saha & Gopalan (2020), PAC Battling Bandits in the Plackett-Luce Model](https://proceedings.mlr.press/v119/saha20b.html)：Plackett–Luce 的逐次加权无放回定义。
- [Dirichlet 分布与分类/多项分布共轭说明](https://www.ccs.neu.edu/home/vip/teach/DMcourse/5_topicmodel_summ/LDA_TM/dirichlet-conjugate-prior.pdf)。

> 彩票设计目标就是使各期近似独立随机。学到的后验分布主要反映有限历史样本与建模假设，不能证明开奖存在可利用规律，也不保证中奖。本项目仅供统计建模与编程研究。
