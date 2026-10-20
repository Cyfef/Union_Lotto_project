# Union Lotto Predictor

一个完整的双色球数据抓取、时序回测和下一期候选号码生成项目。

> 双色球开奖结果具有高度随机性。本项目只用于统计建模和编程研究，不保证中奖，也不构成购彩建议。

## 模型原理

模型分别估计 33 个红球和 16 个蓝球的入选概率：

- 使用经验贝叶斯平滑，将历史频率与理论均匀概率（红球 `6/33`、蓝球 `1/16`）结合；
- 历史记录采用指数衰减，越近的开奖权重越高；
- 用严格的时间顺序验证选择衰减半衰期和先验混合比例；
- 最后 20% 数据作为独立测试集，并与均匀随机基线比较；
- 按预测概率进行加权无放回抽样，生成 6 个红球和 1 个蓝球。

模型实现位于 `src/union_lotto/model.py`。复杂模型很容易在有限、随机性很强的彩票数据上过拟合，因此这里优先采用参数少、可解释且可以诚实回测的概率模型。

## 使用方式

要求 Python 3.10 或更高版本，无第三方运行时依赖。

```powershell
python -m pip install -e .
```

一键完成数据抓取、训练、回测和预测：

```powershell
python -m union_lotto pipeline
```

也可以分步运行：

```powershell
python -m union_lotto fetch --source auto
python -m union_lotto validate
python -m union_lotto train
python -m union_lotto predict --tickets 5 --seed 2026
```

主要输出：

- `data/ssq_history.csv`：标准化历史开奖数据；
- `artifacts/model.json`：训练参数和回测指标；
- `predictions/latest.json`：下一期候选号码及概率排名。

运行测试：

```powershell
$env:PYTHONPATH = "src"
python -m unittest discover -s tests -v
```

数据默认优先尝试中国福彩网官方接口，并使用公开历史镜像作为备用。开奖结果请以官方公告为准。
