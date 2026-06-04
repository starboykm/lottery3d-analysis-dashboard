# <span style="font-size: 32px;">彩票是随机概率事件，不可能被预测中！该项目只做娱乐用图，不能用于购买彩票！</span>

# 3D / 排列三概率分析看板

这是一个面向福彩 3D / 体彩排列三历史数据的本地分析项目。它可以录入开奖数据、训练概率模型、生成多套候选排序、记录预测结果，并在开奖后统计命中排名。

再次强调：本项目只做历史数据学习、概率排序、策略复盘和可视化展示。彩票开奖结果应视为随机事件，任何模型和策略都不能保证中奖，也不能稳定获利。

## 功能概览

- 历史开奖数据查询、导入、追加录入。
- 下一期号码概率预测。
- 百位、十位、个位分别输出 0-9 的概率排序。
- 生成预算内推荐票。
- 生成神经网络模型预测号。
- 生成策略模型预测号。
- 生成神经网络与策略模型交叉命中的相同预测号。
- 生成马尔可夫模型预测号。
- 生成马尔可夫模型前 5 个“不对位胆码”。
- 开奖后录入真实号码并结算命中情况。
- 统计推荐票、策略过滤有效组合、神经网络、策略模型、马尔可夫模型的历史排名表现。
- 展示近 100 期预测复盘表，包括开奖号、分位概率排名、有效组合数量、推荐排名、策略排名、马尔可夫排名等。

## 数据格式

默认历史文件为 `history.csv`，支持以下格式：

```csv
issue,number,trial
2026001,527,
2026002,049,138
2026003,718,
```

也支持无表头格式：

```csv
2026001,527
2026002,049
2026003,718
```

字段说明：

- `issue`：7 位期号，前 4 位为年份，后 3 位为当年第几期。
- `number`：3 位开奖号码，从 `000` 到 `999`。
- `trial`：可选试机号，3 位数字；没有可留空。

## 模型与策略

项目目前将预测分成几条独立线路，方便对比：

1. 基础概率 / 神经网络预测
   - 长期衰减频率。
   - 近 30 期频率。
   - 近 80 期频率。
   - 转移概率。
   - 遗漏间隔。
   - MLP 神经网络，环境安装 `scikit-learn` 且历史样本足够时自动启用。

2. 策略模型预测
   - 定位杀号。
   - 和值尾、跨度过滤。
   - 期号尾定胆。
   - 上期跨度定胆。
   - 上期开奖号位置对应胆码。
   - 多个可程序化的铁胆公式。
   - 可选试机号相关规则。

3. 马尔可夫模型预测
   - 按位置学习上一期到下一期的转移概率。
   - 学习整号状态转移。
   - 学习相邻两位状态转移。
   - 引入和值尾、跨度等弱状态转移信号。
   - 单独输出马尔可夫排序号。
   - 单独输出马尔可夫前 5 个不对位胆码。

## 安装依赖

建议使用项目已有的 Python 环境：

```powershell
C:\Users\starb\anaconda3\envs\Qbot\python.exe -m pip install -r requirements.txt
```

如果你的电脑有全局 Python，也可以使用：

```powershell
python -m pip install -r requirements.txt
```

## 启动看板

方式一：直接运行 Python。

```powershell
C:\Users\starb\anaconda3\envs\Qbot\python.exe dashboard.py --host 127.0.0.1 --port 8765
```

方式二：双击运行：

```text
start_dashboard.bat
```

启动后打开：

```text
http://127.0.0.1:8765
```

停止后台服务：

```powershell
.\stop_dashboard.ps1
```

## 命令行用法

生成示例数据：

```powershell
C:\Users\starb\anaconda3\envs\Qbot\python.exe lottery3d.py make-sample --output history.csv
```

训练并保存模型快照：

```powershell
C:\Users\starb\anaconda3\envs\Qbot\python.exe lottery3d.py train --data history.csv
```

预测下一期：

```powershell
C:\Users\starb\anaconda3\envs\Qbot\python.exe lottery3d.py predict --data history.csv --issue 2026125 --budget 20
```

带试机号预测：

```powershell
C:\Users\starb\anaconda3\envs\Qbot\python.exe lottery3d.py predict --data history.csv --issue 2026125 --trial 123 --budget 20
```

录入新开奖数据：

```powershell
C:\Users\starb\anaconda3\envs\Qbot\python.exe lottery3d.py add --data history.csv --issue 2026125 --number 527
```

开奖后结算预测：

```powershell
C:\Users\starb\anaconda3\envs\Qbot\python.exe lottery3d.py settle --issue 2026125 --actual 527
```

历史回测：

```powershell
C:\Users\starb\anaconda3\envs\Qbot\python.exe lottery3d.py backtest --data history.csv --start 35 --budget 20 --output backtest.json
```

从第 2 期开始滚动训练和统计：

```powershell
C:\Users\starb\anaconda3\envs\Qbot\python.exe lottery3d.py rolling-train --data history.csv --budget 20 --effective-limit 1000
```

查询某期开奖号码在预测列表中的排名：

```powershell
C:\Users\starb\anaconda3\envs\Qbot\python.exe lottery3d.py rank --issue 2026125 --number 527
```

## 目录说明

```text
.
├── dashboard.py          # 本地 Web 看板
├── lottery3d.py          # 核心模型、策略、预测、结算逻辑
├── history.csv           # 历史开奖数据
├── requirements.txt      # Python 依赖
├── start_dashboard.bat   # Windows 启动脚本
├── stop_dashboard.ps1    # Windows 停止脚本
└── records/              # 运行时预测、结算和统计记录，默认不提交到 Git
```

## 运行记录

程序运行后会生成以下类型的文件：

- `records/predictions.jsonl`：每次预测流水。
- `records/prediction_期号.json`：单期预测详情。
- `records/settlements.jsonl`：开奖后结算记录。
- `records/rolling_predictions.jsonl`：全历史滚动预测记录。
- `records/stats.json`：累计统计。
- `dashboard.log`、`dashboard.out.log`、`dashboard.err.log`：看板日志。
- `dashboard.pid`：后台服务进程号。

这些文件会随使用不断增长，默认通过 `.gitignore` 排除。

## 注意事项

- 本项目不会也不能预测真实彩票开奖。
- 所有“概率”“排名”“命中率”都只是基于历史样本的回测或排序结果。
- 历史命中表现不代表未来命中表现。
- 不要用本项目输出购买彩票。
- 如果继续扩展策略，可在 `lottery3d.py` 的 `apply_formula_strategy()` 中增加可程序化规则。

# <span style="font-size: 32px;">彩票是随机概率事件，不可能被预测中！该项目只做娱乐用图，不能用于购买彩票！</span>
