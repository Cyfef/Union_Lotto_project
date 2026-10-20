# 双色球预测模型 V2

V2 在 V1 的号码频率模型基础上重新构建为“两阶段统计特征模型”：

1. **号码倾向模型**：用正则化逻辑回归分别估计每个红球和蓝球的入选概率。输入包括近 5/15/50/150 期频率、遗漏期数、重号、号码位置、奇偶、大小、质数和除 3 余数。
2. **组合形态模型**：生成大量候选组合，再根据最近 500 期的和值、跨度、最小/最大号、奇偶比、大小比、质合比、三区比、012 路、连号、同尾和重号分布重新排序。

训练报告同时保留“形态权重为 0”的消融对照，但 V2 的正式模型只从非零形态权重中选择验证表现最佳者，防止自动调参悄悄退化成只看单号概率的 V1 模型。

所有特征只使用目标期开奖前的数据。最后 20% 的期次作为独立测试集，超参数只在更早的验证集选择，避免未来数据泄漏。

## 当前回测结果

使用 `2003001`～`2026119` 共 3516 期数据，最后 704 期作为独立测试集：

- 纯号码概率 Top-6 平均命中 `1.053` 个红球；
- 加入统计形态重排后平均命中 `1.060` 个红球；
- 均匀随机理论期望为 `1.091` 个红球；
- 蓝球 Top-1 命中率为 `6.11%`，均匀随机理论值为 `6.25%`。

统计形态重排改善了本模型自身的组合结果，但仍未超过随机基线，因此不能声称模型具有稳定预测能力。这一结论会随以后新增开奖数据继续接受时间回测。

## 快速开始

在 `v2` 目录执行：

```powershell
python -m pip install -e .
python -m union_lotto_v2 pipeline
```

分步执行：

```powershell
python -m union_lotto_v2 fetch
python -m union_lotto_v2 validate
python -m union_lotto_v2 train
python -m union_lotto_v2 predict --tickets 5 --seed 2026
```

运行测试：

```powershell
$env:PYTHONPATH = "src"
python -m unittest discover -s tests -v
```

输出文件：

- `data/ssq_history.csv`：历史开奖数据；
- `artifacts/model.json`：模型参数、验证指标和独立测试指标；
- `predictions/latest.json`：候选号码、概率排名及每组组合的统计特征。

## 统计指标来源

V2 采用的指标与上海福彩数据中心、中彩网双色球图表中展示的常用统计口径一致，包括跨度、和值、奇偶比、大小比、质合比、分区、余数、连号、同尾和遗漏等。

- [上海福彩双色球遗漏统计](https://appsh.swlc.net.cn/shfcoc_datachart/datachart/ssq/yltj/index.html)
- [上海福彩双色球直方图指标](https://appsh.swlc.net.cn/shfcoc_datachart/datachart/ssq/zft/)
- [中彩网双色球历史同期统计](https://tubiao.zhcw.com/tubiao/ssqNew/ssqJsp/ssq_LiShiTongQi_HongQiuFenBuTuAsc.jsp?issue=002)
- [双色球游戏规则](https://www.zhcw.com/upload/resources/file/2026/01/16/214094.pdf)

> 统计特征只能描述历史数据分布，不能证明开奖结果可预测。本项目仅供研究，不构成购彩建议。

