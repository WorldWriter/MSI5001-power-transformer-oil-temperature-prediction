# ETT数据集详细调研报告
# ETT Dataset Research Report

---

**报告日期 / Report Date**: 2025年10月 / October 2025
**报告目的 / Purpose**: 为MSI5001课程项目提供数据集背景和技术支持
**作者 / Author**: MSI5001 Group Project Team

---

## 目录 / Table of Contents

1. [数据集概述 / Dataset Overview](#1-数据集概述--dataset-overview)
2. [本地数据集对比分析 / Local Dataset Comparison](#2-本地数据集对比分析--local-dataset-comparison)
3. [变压器业务背景 / Transformer Business Context](#3-变压器业务背景--transformer-business-context)
4. [物理原理：负载-温度关系 / Physics: Load-Temperature Relationship](#4-物理原理负载-温度关系--physics-load-temperature-relationship)
5. [数据集详细信息 / Dataset Details](#5-数据集详细信息--dataset-details)
6. [技术方法演进 / Technical Methods Evolution](#6-技术方法演进--technical-methods-evolution)
7. [SOTA模型与基准结果 / SOTA Models & Benchmarks](#7-sota模型与基准结果--sota-models--benchmarks)
8. [评估指标 / Evaluation Metrics](#8-评估指标--evaluation-metrics)
9. [参考文献 / References](#9-参考文献--references)

---

## 1. 数据集概述 / Dataset Overview

### 1.1 什么是ETT数据集？

**ETT（Electricity Transformer Temperature，电力变压器温度）数据集**是用于支持长序列时间序列预测研究的标准基准数据集。该数据集由中国两个不同地区的变电站采集，涵盖2016年7月至2018年7月的2年数据。

ETT数据集最初由Zhou等人在**AAAI 2021会议**上发表的**Informer论文**中引入，该论文获得了**AAAI 2021最佳论文奖**。此后，ETT数据集已成为时间序列预测领域最广泛使用的基准之一，被超过**50篇研究论文**引用。

### 1.1 What is the ETT Dataset?

The **ETT (Electricity Transformer Temperature) dataset** is a standard benchmark dataset collected to support the investigation of long sequence time-series forecasting problems. The dataset was collected from transformer substations in two different regions of China, covering 2 years of data from July 2016 to July 2018.

The ETT dataset was first introduced in the **Informer paper** published at **AAAI 2021** by Zhou et al., which received the **AAAI 2021 Best Paper Award**. Since then, the ETT dataset has become one of the most widely used benchmarks in time series forecasting, being cited in over **50 research papers**.

### 1.2 数据集重要性

- **学术影响力**: Informer论文获得AAAI 2021最佳论文奖，引领长序列时间序列预测研究方向
- **工业应用**: 变压器油温预测对电力系统运维至关重要，可预防设备损坏、优化维护计划
- **研究价值**: 数据包含多种时间模式（日周期、周周期、长期趋势、不规则波动），非常适合测试模型的长期依赖捕获能力

### 1.2 Dataset Importance

- **Academic Impact**: The Informer paper won the AAAI 2021 Best Paper Award, leading research in long sequence time-series forecasting
- **Industrial Application**: Transformer oil temperature prediction is critical for power system operations, preventing equipment damage and optimizing maintenance schedules
- **Research Value**: The data contains multiple temporal patterns (daily cycles, weekly cycles, long-term trends, irregular fluctuations), making it ideal for testing models' long-term dependency capture capabilities

### 1.3 数据获取渠道

ETT数据集在多个平台公开可用：

| 平台 / Platform | 链接 / URL | 说明 / Description |
|----------------|------------|-------------------|
| **GitHub** | https://github.com/zhouhaoyi/ETDataset | 原始数据源，由作者维护 / Original source, maintained by authors |
| **Hugging Face** | https://huggingface.co/datasets/ett | 便于集成到ML流程 / Easy integration into ML pipelines |
| **Kaggle** | Search "Electricity Transformer Dataset" | 社区讨论和代码示例 / Community discussions and code examples |
| **Papers with Code** | https://paperswithcode.com/dataset/ett | 基准结果对比 / Benchmark results comparison |

---

## 2. 本地数据集对比分析 / Local Dataset Comparison

### 2.1 本地数据集统计

我们的项目使用了两个变压器的数据文件：

| 文件名 / File | 数据点数量 / Data Points | 起始日期 / Start Date | 结束日期 / End Date | 时间跨度 / Duration |
|--------------|--------------------------|---------------------|-------------------|-------------------|
| `trans_1.csv` | 69,680 | 2018-07-01 00:00 | 2020-06-26 19:45 | 约2年 / ~2 years |
| `trans_2.csv` | 69,680 | 2018-07-01 00:00 | 2020-06-26 19:45 | 约2年 / ~2 years |

**采样频率 / Sampling Frequency**: 15分钟间隔 / 15-minute intervals
**特征数量 / Features**: 6个功率负载特征 + 1个油温目标 / 6 power load features + 1 oil temperature target

### 2.1 Local Dataset Statistics

Our project uses data from two transformers:

The datasets contain **69,680 data points** each, spanning approximately 2 years with **15-minute sampling intervals**.

### 2.2 与ETT标准数据集的对比

#### 2.2.1 ETT数据集变体

ETT数据集有多个版本，主要区别在于采样频率：

| 变体 / Variant | 采样频率 / Sampling | 数据点数量 / Data Points | 说明 / Description |
|----------------|-------------------|------------------------|-------------------|
| **ETTh1, ETTh2** | 1小时 / 1 hour | 17,520 | h = hourly (小时级) |
| **ETTm1, ETTm2** | 15分钟 / 15 min | 70,080 | m = minute (分钟级) |

**计算验证 / Calculation Verification**:
- ETTm标准: 2年 × 365天 × 24小时 × 4（每小时4个15分钟） = **70,080个数据点**
- 本地数据: **69,680个数据点**
- 差异: 400个数据点（约占0.6%）

#### 2.2.1 ETT Dataset Variants

The ETT dataset comes in multiple versions based on sampling frequency. The standard ETTm variant contains 70,080 data points, while our local dataset has 69,680 points.

#### 2.2.2 对比结论

| 对比项 / Comparison Item | ETTm1/m2 标准 / Standard | 本地数据集 / Local Dataset | 匹配度 / Match |
|--------------------------|--------------------------|---------------------------|----------------|
| 采样频率 / Sampling | 15分钟 / 15 min | 15分钟 / 15 min | ✅ 完全匹配 / Perfect match |
| 时间跨度 / Duration | 2年 / 2 years | 约2年 / ~2 years | ✅ 匹配 / Match |
| 数据点数量 / Data Points | 70,080 | 69,680 | ⚠️ 接近但略少 / Close but slightly fewer |
| 特征结构 / Features | 6 + 1 (OT) | 6 + 1 (OT) | ✅ 完全匹配 / Perfect match |
| 特征名称 / Feature Names | HUFL, HULL, MUFL, MULL, LUFL, LULL, OT | 相同 / Same | ✅ 完全匹配 / Perfect match |

**结论 / Conclusion**: 本地数据集**类似于ETTm变体**（15分钟采样），而**不是ETTh变体**（1小时采样）。数据点数量接近标准ETTm数据集（差异<1%），可能是同源数据的变体版本或经过轻微预处理的版本。

**Conclusion**: The local dataset is **similar to the ETTm variant** (15-minute sampling), not the ETTh variant (1-hour sampling). The data point count is close to the standard ETTm dataset (difference <1%), suggesting it may be a variant or lightly preprocessed version of the same source data.

### 2.3 数据特征对比

通过对比两个变压器的数值范围，我们发现：

| 特征 / Feature | trans_1 范围 / Range | trans_2 范围 / Range | 说明 / Notes |
|----------------|---------------------|---------------------|--------------|
| **HUFL** | 4.3 ~ 13.0 | 32.8 ~ 43.2 | trans_2数值约为trans_1的3-4倍 |
| **OT (油温)** | 9.8 ~ 30.5°C | 27.2 ~ 45.3°C | trans_2运行温度明显更高 |

这种差异表明：
- **trans_1** 类似ETTm1，可能是较小型或轻载变压器
- **trans_2** 类似ETTm2，可能是大型或重载变压器

### 2.3 Data Feature Comparison

Comparing the value ranges of the two transformers:

The differences suggest **trans_1** is similar to ETTm1 (smaller or lightly loaded transformer), while **trans_2** resembles ETTm2 (larger or heavily loaded transformer).

---

## 3. 变压器业务背景 / Transformer Business Context

### 3.1 变压器在电力系统中的作用

**电力变压器（Power Transformer）**是电力系统的核心设备，负责在不同电压等级之间转换电能，实现电能的高效传输和分配。

### 3.1 Role of Transformers in Power Systems

**Power Transformers** are core components of electrical power systems, responsible for converting electrical energy between different voltage levels to enable efficient transmission and distribution.

### 3.2 电压等级体系

电力系统采用多级电压架构以优化长距离输电效率：

#### 3.2.1 高压（High Voltage, HV）

- **电压范围**: >36 kV（千伏）
- **应用场景**: 长距离电力传输（发电厂 → 变电站）
- **优点**: 高压传输可减少线路损耗（功率损耗 ∝ I²R，电压越高则电流越小）
- **数据集对应**: **HUFL**（高压有用负载），**HULL**（高压无用负载）

#### 3.2.2 中压（Medium Voltage, MV）

- **电压范围**: 5 kV ~ 35 kV
- **应用场景**: 城市区域配电网络
- **功能**: 从变电站向工业用户或区域配电站传输电力
- **数据集对应**: **MUFL**（中压有用负载），**MULL**（中压无用负载）

#### 3.2.3 低压（Low Voltage, LV）

- **电压范围**: <1 kV
- **应用场景**: 终端用户（家庭、商业建筑）
- **功能**: 将电压降至设备可用的安全水平（如220V或110V）
- **数据集对应**: **LUFL**（低压有用负载），**LULL**（低压无用负载）

### 3.2 Voltage Level Hierarchy

The power system uses a multi-level voltage architecture to optimize long-distance transmission efficiency:

| 电压等级 / Level | 范围 / Range | 用途 / Purpose | 示例 / Example |
|-----------------|--------------|---------------|----------------|
| **高压 / High Voltage** | >36 kV | 长距离输电 / Long-distance transmission | 500 kV, 220 kV transmission lines |
| **中压 / Medium Voltage** | 5-35 kV | 区域配电 / Regional distribution | 10 kV, 35 kV urban networks |
| **低压 / Low Voltage** | <1 kV | 终端用电 / End-user consumption | 220V, 380V household/commercial |

**电力流向 / Power Flow**: 发电厂（高压）→ 变电站（降压）→ 配电网（中压）→ 用户变压器（降至低压）→ 终端用户

### 3.3 功率负载类型：有功功率 vs 无功功率

#### 3.3.1 有用负载（Useful Load = Active Power = 有功功率）

**定义**: 实际转换为有用功的电功率（如机械功、光能、热能等）

**符号**: P，单位为瓦特（W）或兆瓦（MW）

**数据集中的特征**:
- **HUFL** (High Useful Load): 高压侧有功功率
- **MUFL** (Medium Useful Load): 中压侧有功功率
- **LUFL** (Low Useful Load): 低压侧有功功率

**物理意义**: 这部分功率真正做功，是用户实际消耗的电能。

#### 3.3.2 无用负载（Useless Load = Reactive Power = 无功功率）

**定义**: 用于建立和维持电磁场的功率，不转换为其他形式的能量，但在交流系统中必不可少。

**符号**: Q，单位为乏（VAR）或兆乏（MVAR）

**数据集中的特征**:
- **HULL** (High Useless Load): 高压侧无功功率
- **MULL** (Medium Useless Load): 中压侧无功功率
- **LULL** (Low Useless Load): 低压侧无功功率

**为什么叫"无用"但又必需？**

虽然无功功率本身不做功，但它对电力系统至关重要：

1. **维持磁场**: 变压器、电机等感性设备需要无功功率来建立磁场
2. **电压稳定**: 无功功率影响系统电压水平，缺乏无功会导致电压下降
3. **功率因数**: 过多无功功率会降低功率因数，增加线路损耗

**类比理解**:
- 有功功率 = 啤酒（真正喝到的部分）
- 无功功率 = 啤酒泡沫（看起来占空间但不喝，但啤酒没有泡沫不完整）

### 3.3 Power Load Types: Active Power vs Reactive Power

#### Active Power (Useful Load)

**Definition**: The electrical power actually converted into useful work (mechanical work, light, heat, etc.)

**Symbol**: P, unit: Watt (W) or Megawatt (MW)

**In Dataset**: HUFL, MUFL, LUFL represent active power at high, medium, and low voltage sides

**Physical Meaning**: This is the power that truly does work and is what users actually consume.

#### Reactive Power (Useless Load)

**Definition**: Power used to establish and maintain electromagnetic fields, not converted into other forms of energy, but essential in AC systems.

**Symbol**: Q, unit: VAR (Volt-Ampere Reactive) or MVAR

**In Dataset**: HULL, MULL, LULL represent reactive power at high, medium, and low voltage sides

**Why "Useless" but Necessary?**

Though reactive power doesn't do work, it's crucial for power systems:

1. **Maintain Magnetic Fields**: Inductive devices (transformers, motors) need reactive power to establish magnetic fields
2. **Voltage Stability**: Reactive power affects system voltage levels; lack of it causes voltage drops
3. **Power Factor**: Excessive reactive power lowers power factor and increases line losses

**Analogy**:
- Active Power = Beer (the part you actually drink)
- Reactive Power = Foam (seems to take space without drinking, but beer without foam is incomplete)

---

## 4. 物理原理：负载-温度关系 / Physics: Load-Temperature Relationship

### 4.1 为什么功率负载会影响油温？

变压器油温的升高主要源于变压器运行时产生的**内部损耗**，这些损耗转化为热量。负载越大，损耗越大，发热越多，油温越高。

### 4.1 Why Does Power Load Affect Oil Temperature?

Transformer oil temperature rise primarily stems from **internal losses** during transformer operation, which convert into heat. Greater load → greater losses → more heat → higher oil temperature.

### 4.2 变压器损耗的来源

变压器的主要损耗包括：

#### 4.2.1 铜损（Copper Losses / Winding Losses）

**公式 / Formula**: P<sub>copper</sub> = I² × R

- **I**: 绕组电流（与负载成正比）/ Winding current (proportional to load)
- **R**: 绕组电阻 / Winding resistance

**特点**:
- **负载相关**: 负载越大，电流越大，铜损按电流的平方增长
- **主要热源**: 在额定负载下，铜损通常占总损耗的60-70%
- **直接影响**: 绕组发热直接传导给周围的变压器油

#### 4.2.2 铁损（Iron Losses / Core Losses）

包括：
- **磁滞损耗（Hysteresis Loss）**: 铁芯磁化过程中的能量损失
- **涡流损耗（Eddy Current Loss）**: 铁芯中感应电流产生的损耗

**特点**:
- **负载无关**: 只要变压器通电，无论负载大小，铁损基本恒定
- **占比**: 约占总损耗的30-40%

#### 4.2.3 杂散损耗（Stray Losses）

- 绕组涡流损耗
- 结构部件（油箱壁、螺栓等）的涡流损耗
- 通常占总损耗的5-10%

### 4.2 Sources of Transformer Losses

#### Copper Losses (Winding Losses)

**Formula**: P<sub>copper</sub> = I² × R

**Characteristics**:
- **Load-dependent**: Higher load → higher current → copper loss increases quadratically
- **Main heat source**: At rated load, copper losses typically account for 60-70% of total losses
- **Direct impact**: Winding heat directly conducts to surrounding transformer oil

#### Iron Losses (Core Losses)

Includes:
- **Hysteresis Loss**: Energy loss during core magnetization
- **Eddy Current Loss**: Losses from induced currents in the core

**Characteristics**:
- **Load-independent**: Constant as long as transformer is energized
- **Proportion**: ~30-40% of total losses

#### Stray Losses

- Winding eddy current losses
- Structural component losses (tank walls, bolts, etc.)
- Typically 5-10% of total losses

### 4.3 热传导过程

变压器的热量传递遵循以下路径：

```
绕组发热 → 热量传递给油液 → 油液对流循环 → 散热器/油箱壁散热到环境
Winding Heat → Heat Transfer to Oil → Oil Convection → Radiator/Tank Wall Heat Dissipation
```

#### 4.3.1 热传导阶段

- **绕组 → 油液**: 绕组温度高于油温，热量通过热传导传递给周围油液
- **油液性质影响**: 油的热导率（Thermal Conductivity）决定传热效率

#### 4.3.2 对流循环阶段

- **自然对流**: 热油密度降低，向上流动；冷油下沉，形成循环
- **强制循环**: 大型变压器使用油泵强制循环以提高散热效率
- **油液性质影响**: 粘度（Viscosity）影响流动性，比热容（Specific Heat Capacity）影响吸热能力

#### 4.3.3 散热阶段

- **散热器**: 增大与空气接触面积
- **油箱壁**: 热量通过油箱壁辐射和对流散热到环境
- **环境影响**: 环境温度越高，散热越困难

### 4.3 Heat Transfer Process

Heat transfer in transformers follows this pathway:

```
Winding Heat → Heat Transfer to Oil → Oil Convection → Radiator/Tank Wall Dissipation to Environment
```

The process includes:
1. **Conduction**: Winding → Oil (thermal conductivity matters)
2. **Convection**: Hot oil rises, cold oil sinks (viscosity and specific heat capacity matter)
3. **Dissipation**: Radiator/tank wall → Environment (ambient temperature matters)

### 4.4 负载-温度数学关系

根据**IEEE C57.91**和**IEC 60076-7**标准，变压器油温可用微分方程建模：

**简化模型 / Simplified Model**:

```
ΔT_oil = ΔT_oil_rated × [(K² × R + 1) / (R + 1)]^n
```

其中 / Where:
- **ΔT_oil**: 油温升高 / Oil temperature rise
- **K**: 负载系数（实际负载/额定负载）/ Load factor (actual load / rated load)
- **R**: 负载损耗与空载损耗的比率 / Ratio of load loss to no-load loss
- **n**: 经验指数（通常为0.8-1.0）/ Empirical exponent (typically 0.8-1.0)

**关键洞察 / Key Insights**:

1. **非线性关系**: 油温不与负载线性相关，而是按指数增长
2. **负载平方项**: 由于铜损 ∝ I²，负载翻倍可能导致油温增加超过2倍
3. **时间延迟**: 油温变化具有热惯性（时间常数约为数小时），这使得预测更具挑战性

### 4.4 Load-Temperature Mathematical Relationship

According to **IEEE C57.91** and **IEC 60076-7** standards, transformer oil temperature can be modeled using differential equations.

**Key Insights**:

1. **Nonlinear Relationship**: Oil temperature doesn't scale linearly with load but grows exponentially
2. **Quadratic Load Term**: Due to copper losses ∝ I², doubling the load may increase oil temperature by more than 2×
3. **Time Delay**: Oil temperature changes have thermal inertia (time constant ~hours), making prediction challenging

### 4.5 为什么需要预测油温？

#### 4.5.1 设备保护

- **过热损坏**: 油温过高（>100°C）会加速绝缘老化，缩短变压器寿命
- **热点温度**: 油温是估算绕组热点温度的关键参数

#### 4.5.2 运维优化

- **负载优化**: 预测油温可指导调度员合理分配负载
- **维护计划**: 提前识别异常温升，安排预防性维护
- **经济效益**: 避免保守估计导致的容量浪费，同时防止过载损坏

### 4.5 Why Predict Oil Temperature?

#### Equipment Protection
- **Overheating Damage**: High oil temperature (>100°C) accelerates insulation aging, shortening transformer lifespan
- **Hot-spot Temperature**: Oil temperature is a key parameter for estimating winding hot-spot temperature

#### Operations Optimization
- **Load Optimization**: Predicting oil temperature guides dispatchers in load allocation
- **Maintenance Planning**: Early identification of abnormal temperature rises enables preventive maintenance
- **Economic Benefits**: Avoids capacity waste from conservative estimates while preventing overload damage

---

## 5. 数据集详细信息 / Dataset Details

### 5.1 特征说明

每个数据点包含以下7个字段：

| 特征名 / Feature | 全称 / Full Name | 中文含义 / Chinese Meaning | 单位 / Unit | 类型 / Type |
|-----------------|------------------|---------------------------|------------|------------|
| **date** | Date and Time | 时间戳 | YYYY-MM-DD HH:MM:SS | 索引 / Index |
| **HUFL** | High Useful Load | 高压侧有功功率 | kW or MW | 输入 / Input |
| **HULL** | High Useless Load | 高压侧无功功率 | kVAR or MVAR | 输入 / Input |
| **MUFL** | Medium Useful Load | 中压侧有功功率 | kW or MW | 输入 / Input |
| **MULL** | Medium Useless Load | 中压侧无功功率 | kVAR or MVAR | 输入 / Input |
| **LUFL** | Low Useful Load | 低压侧有功功率 | kW or MW | 输入 / Input |
| **LULL** | Low Useless Load | 低压侧无功功率 | kVAR or MVAR | 输入 / Input |
| **OT** | Oil Temperature | 油温（目标变量） | °C | 目标 / Target |

### 5.2 数据特点

#### 5.2.1 时间模式

ETT数据集包含多种时间模式：

- **短期周期性**: 每日周期（24小时），反映日常用电模式（白天高负载，夜间低负载）
- **中期周期性**: 每周周期（7天），工作日与周末的负载差异
- **长期趋势**: 季节性变化（夏季空调负载增加，冬季供暖负载增加）
- **不规则波动**: 突发事件、天气变化等导致的随机波动

#### 5.2.2 数据质量

- **完整性**: 数据几乎无缺失值
- **采样频率**: ETTm1/m2每15分钟采样一次，ETTh1/h2每小时采样一次
- **时间跨度**: 2年数据足以覆盖多个季节周期

### 5.2 Data Characteristics

#### Temporal Patterns

The ETT dataset contains multiple temporal patterns:

- **Short-term Periodicity**: Daily cycles (24 hours), reflecting daily electricity usage patterns
- **Medium-term Periodicity**: Weekly cycles (7 days), weekday vs. weekend load differences
- **Long-term Trends**: Seasonal variations (summer AC load increase, winter heating load increase)
- **Irregular Fluctuations**: Random variations due to sudden events, weather changes, etc.

#### Data Quality

- **Completeness**: Nearly no missing values
- **Sampling Frequency**: ETTm1/m2 sampled every 15 minutes, ETTh1/h2 every hour
- **Time Span**: 2 years of data sufficient to cover multiple seasonal cycles

### 5.3 数据预处理建议

根据课程要求和最佳实践：

1. **数据分割**:
   - 训练集/验证集/测试集 = 12/4/4个月（ETT标准）
   - 或 80%/20% 时间组分割（课程建议）
   - **重要**: 时间序列数据应按时间顺序分割，不能随机打乱

2. **归一化**:
   - 推荐使用标准化（Z-score normalization）: `(x - μ) / σ`
   - 每个特征单独归一化
   - **注意**: 使用训练集统计量归一化测试集

3. **序列生成**:
   - **滑动窗口**: 使用过去N个时间步预测未来1步或多步
   - **预测任务**:
     - 1小时预测: 使用前N步预测第4步后的油温
     - 1天预测: 预测第96步后的油温
     - 1周预测: 预测第672步后的油温

### 5.3 Data Preprocessing Recommendations

Based on course requirements and best practices:

1. **Data Splitting**:
   - Train/Val/Test = 12/4/4 months (ETT standard) or 80%/20% time-based split
   - **Important**: Time series data should be split chronologically, not randomly shuffled

2. **Normalization**:
   - Recommend Z-score normalization: `(x - μ) / σ`
   - Normalize each feature independently
   - **Note**: Use training set statistics to normalize test set

3. **Sequence Generation**:
   - **Sliding Window**: Use past N time steps to predict 1 or more future steps
   - **Prediction Tasks**:
     - 1-hour: Predict oil temperature 4 steps ahead
     - 1-day: Predict 96 steps ahead
     - 1-week: Predict 672 steps ahead

---

## 6. 技术方法演进 / Technical Methods Evolution

### 6.1 传统方法（物理模型）

#### 6.1.1 热回路模型（Thermal Circuit Model）

基于**IEEE C57.91**和**IEC 60076-7**标准，使用微分方程建模油温动态：

**优点**:
- 物理可解释性强
- 不需要大量历史数据
- 适用于实时监控

**缺点**:
- 需要准确的物理参数（热时间常数、绕组电阻等）
- 对非线性、不规则模式的拟合能力弱
- 难以捕捉复杂的环境影响

### 6.1 Traditional Methods (Physical Models)

#### Thermal Circuit Models

Based on **IEEE C57.91** and **IEC 60076-7** standards, using differential equations to model oil temperature dynamics.

**Pros**: Strong physical interpretability, no need for large historical data, suitable for real-time monitoring
**Cons**: Requires accurate physical parameters, weak fitting for nonlinear/irregular patterns, difficult to capture complex environmental influences

### 6.2 经典机器学习方法

#### 6.2.1 随机森林（Random Forest）

- **原理**: 集成多棵决策树，投票或平均预测结果
- **优点**: 对非线性关系建模能力强，鲁棒性好，可解释性较好
- **缺点**: 时间序列依赖建模较弱，需要手工提取时间特征

#### 6.2.2 支持向量机（SVM）

- **原理**: 寻找最优超平面进行回归
- **优点**: 小样本学习能力强
- **缺点**: 计算复杂度高，难以处理长序列

**性能**: 在ETT数据集上，Random Forest通常可达到R² ≈ 0.55-0.65（1小时预测任务）

### 6.2 Classical Machine Learning Methods

#### Random Forest
- **Principle**: Ensemble of multiple decision trees, voting or averaging predictions
- **Pros**: Strong nonlinear modeling, good robustness, decent interpretability
- **Cons**: Weak time series dependency modeling, requires manual time feature extraction

**Performance**: On ETT dataset, Random Forest typically achieves R² ≈ 0.55-0.65 (1-hour prediction task)

### 6.3 深度学习方法

#### 6.3.1 RNN系列（循环神经网络）

**LSTM (Long Short-Term Memory)**:
- **原理**: 通过门控机制（遗忘门、输入门、输出门）记忆长期依赖
- **优点**: 能有效捕捉时间序列的长期模式
- **缺点**: 训练较慢，参数量较大

**GRU (Gated Recurrent Unit)**:
- **原理**: LSTM的简化版，只有更新门和重置门
- **优点**: 参数更少，训练更快，性能接近LSTM
- **适用场景**: 计算资源有限时的首选

**BiLSTM/BiGRU (双向)**:
- **原理**: 同时考虑前向和后向时间依赖
- **优点**: 捕捉更丰富的上下文信息
- **注意**: 仅适用于离线预测（需要未来数据）

#### 6.3.2 Transformer系列

**Informer (AAAI 2021 Best Paper)**:
- **创新点**:
  1. **ProbSparse自注意力**: 降低计算复杂度从O(L²)到O(L log L)
  2. **自注意力蒸馏**: 逐层减少序列长度，聚焦关键信息
  3. **生成式解码器**: 一次性预测长序列，而非逐步预测
- **性能**: 在ETT数据集上显著优于RNN和传统Transformer
- **影响**: 开创了长序列时间序列预测研究方向

**PatchTST**:
- **创新**: 将时间序列分块（patch），类似ViT处理图像
- **优点**: 捕捉局部模式，减少计算量

**xPatch (AAAI 2025 - Current SOTA)**:
- **创新**: 双流架构 + 指数季节-趋势分解
- **性能**: 在60%的数据集上达到SOTA（MSE指标）

#### 6.3.3 混合架构

**RNN-ResNet (本项目实现)**:
- **架构**: RNN提取时间特征 → ResNet深度学习非线性映射
- **优点**: 结合RNN的时序建模能力和ResNet的深度表示能力
- **预期性能**: R² ≈ 0.62-0.72（1小时预测）

**CNN-BiGRU**:
- **架构**: CNN提取局部模式 → BiGRU建模长期依赖
- **优点**: CNN加速特征提取，BiGRU捕捉全局依赖

### 6.3 Deep Learning Methods

#### RNN Series

**LSTM (Long Short-Term Memory)**:
- **Principle**: Remembers long-term dependencies via gating mechanisms (forget, input, output gates)
- **Pros**: Effectively captures long-term patterns in time series
- **Cons**: Slower training, more parameters

**GRU (Gated Recurrent Unit)**:
- **Principle**: Simplified LSTM with only update and reset gates
- **Pros**: Fewer parameters, faster training, performance close to LSTM

**BiLSTM/BiGRU (Bidirectional)**:
- **Principle**: Considers both forward and backward time dependencies
- **Note**: Only suitable for offline prediction (requires future data)

#### Transformer Series

**Informer (AAAI 2021 Best Paper)**:
- **Innovations**:
  1. **ProbSparse Self-Attention**: Reduces complexity from O(L²) to O(L log L)
  2. **Self-Attention Distilling**: Layer-wise sequence length reduction, focusing on key information
  3. **Generative Decoder**: Predicts long sequences in one shot
- **Impact**: Pioneered long sequence time-series forecasting research

**xPatch (AAAI 2025 - Current SOTA)**:
- **Innovation**: Dual-stream architecture + exponential seasonal-trend decomposition
- **Performance**: Achieves SOTA on 60% of datasets (MSE metric)

#### Hybrid Architectures

**RNN-ResNet (Our Implementation)**:
- **Architecture**: RNN extracts temporal features → ResNet deep nonlinear mapping
- **Pros**: Combines RNN's temporal modeling with ResNet's deep representation
- **Expected Performance**: R² ≈ 0.62-0.72 (1-hour prediction)

---

## 7. SOTA模型与基准结果 / SOTA Models & Benchmarks

### 7.1 当前SOTA模型（2024-2025）

| 模型 / Model | 会议/期刊 / Venue | 年份 / Year | 主要创新 / Key Innovation |
|--------------|------------------|------------|--------------------------|
| **xPatch** | AAAI | 2025 | 双流架构 + 指数季节-趋势分解 / Dual-stream + exponential decomposition |
| **T3Time** | - | 2025 | 三模态时间序列预测 / Tri-modal time series forecasting |
| **TEMPO** | ICLR | 2024 | 改进的Patch机制 / Improved patching mechanism |
| **Informer** | AAAI (Best Paper) | 2021 | ProbSparse注意力 / ProbSparse attention |

### 7.2 ETT数据集基准结果

#### 7.2.1 ETTh1（1小时采样）- 预测窗口96步

| 模型 / Model | MSE | MAE | 说明 / Notes |
|--------------|-----|-----|-------------|
| **xPatch** | 0.378 | 0.394 | 当前最佳 / Current best |
| **LiNo** | 0.379 | 0.395 | 接近SOTA / Close to SOTA |
| **Informer** | 0.388 | 0.419 | 基准模型 / Baseline |
| **DLinear** | 0.386 | 0.400 | 简单但有效 / Simple but effective |

#### 7.2.2 ETTm1（15分钟采样）- 预测窗口96步

| 模型 / Model | MSE | MAE | 说明 / Notes |
|--------------|-----|-----|-------------|
| **TEMPO** | ~0.30 | ~0.35 | 相比前作提升19.1% / 19.1% improvement |
| **PatchTST** | ~0.33 | ~0.37 | Patch机制 / Patching mechanism |
| **Informer** | ~0.38 | ~0.42 | 开创性工作 / Pioneering work |

**注**: MSE和MAE的具体数值依赖于数据预处理和实验设置，上表为文献中典型结果的近似值。

### 7.2 ETT Dataset Benchmark Results

The table shows results for **ETTh1 (1-hour sampling)** with prediction window of 96 steps. Current SOTA is **xPatch** with MSE 0.378 and MAE 0.394.

**Note**: Specific MSE and MAE values depend on data preprocessing and experimental setup. The table shows approximate typical results from literature.

### 7.3 不同预测窗口的性能对比

预测窗口越长，任务越困难：

| 预测窗口 / Horizon | 时间跨度 / Duration (ETTh1) | 典型MSE范围 / Typical MSE Range |
|-------------------|---------------------------|-------------------------------|
| **96** | 4天 / 4 days | 0.38 - 0.50 |
| **192** | 8天 / 8 days | 0.42 - 0.60 |
| **336** | 14天 / 14 days | 0.45 - 0.70 |
| **720** | 30天 / 30 days | 0.48 - 0.85 |

**洞察 / Insight**:
- 长序列预测（720步）比短序列（96步）困难约50-70%（MSE增加）
- Transformer系列在长序列上优势更明显

### 7.3 Performance Comparison Across Prediction Horizons

Longer prediction horizons are more challenging:

**Insight**:
- Long sequence prediction (720 steps) is ~50-70% harder than short sequence (96 steps) in terms of MSE increase
- Transformer-based models show greater advantage on long sequences

### 7.4 本项目预期性能

根据实现的RNN-ResNet混合模型：

| 任务 / Task | 预期R² / Expected R² | 预期RMSE / Expected RMSE | 说明 / Notes |
|-------------|---------------------|--------------------------|-------------|
| **1小时预测** | 0.62 - 0.72 | 4.0 - 5.0°C | 优于Random Forest (0.60) / Better than RF |
| **1天预测** | 0.40 - 0.55 | 5.0 - 6.0°C | 中等难度 / Medium difficulty |
| **1周预测** | 0.25 - 0.40 | 5.5 - 6.5°C | 高难度 / High difficulty |

**对比基准**:
- Random Forest: R² ≈ 0.60 (1小时预测)
- Pure ResNet: R² ≈ 0.55-0.65 (1小时预测)
- **RNN-ResNet**: R² ≈ 0.62-0.72 (预期)

### 7.4 Expected Performance for Our Project

Based on the implemented RNN-ResNet hybrid model:

Our expected performance is **better than Random Forest (0.60)** and comparable to or better than pure ResNet, demonstrating the advantage of the hybrid architecture.

---

## 8. 评估指标 / Evaluation Metrics

### 8.1 常用指标定义

#### 8.1.1 均方误差（MSE - Mean Squared Error）

**公式 / Formula**:

```
MSE = (1/n) × Σ(y_true - y_pred)²
```

**特点 / Characteristics**:
- **优点**: 对大误差敏感（误差平方），数学上便于优化
- **缺点**: 单位是原始单位的平方（如°C²），不直观
- **适用**: 希望惩罚大误差时使用

#### 8.1.2 均方根误差（RMSE - Root Mean Squared Error）

**公式 / Formula**:

```
RMSE = √MSE = √[(1/n) × Σ(y_true - y_pred)²]
```

**特点 / Characteristics**:
- **优点**: 单位与原始数据相同（如°C），更易理解
- **缺点**: 仍对大误差敏感
- **适用**: 报告预测精度，方便与实际温度对比

#### 8.1.3 平均绝对误差（MAE - Mean Absolute Error）

**公式 / Formula**:

```
MAE = (1/n) × Σ|y_true - y_pred|
```

**特点 / Characteristics**:
- **优点**: 对异常值鲁棒，所有误差等权重
- **缺点**: 数学上不如MSE易于优化
- **适用**: 需要均衡评估所有样本时使用

#### 8.1.4 决定系数（R² - R-squared / Coefficient of Determination）

**公式 / Formula**:

```
R² = 1 - (SS_res / SS_tot)
  = 1 - [Σ(y_true - y_pred)² / Σ(y_true - ȳ)²]
```

**特点 / Characteristics**:
- **取值范围**: (-∞, 1]，其中1表示完美预测
- **优点**: 无量纲，便于跨数据集比较
- **解释**: 表示模型解释了多少目标变量的方差
- **适用**: 评估模型整体拟合度

### 8.1 Common Metrics Definitions

#### MSE (Mean Squared Error)
- **Pros**: Sensitive to large errors, mathematically convenient for optimization
- **Cons**: Unit is squared (e.g., °C²), not intuitive

#### RMSE (Root Mean Squared Error)
- **Pros**: Same unit as original data, easier to understand
- **Cons**: Still sensitive to large errors

#### MAE (Mean Absolute Error)
- **Pros**: Robust to outliers, equal weight to all errors
- **Cons**: Mathematically less convenient than MSE for optimization

#### R² (R-squared)
- **Range**: (-∞, 1], where 1 indicates perfect prediction
- **Pros**: Dimensionless, easy to compare across datasets
- **Interpretation**: Proportion of target variable variance explained by the model

### 8.2 指标对比与选择

| 指标 / Metric | 对异常值敏感度 / Outlier Sensitivity | 可解释性 / Interpretability | 优化难度 / Optimization Difficulty | 推荐场景 / Recommended Use Case |
|--------------|-------------------------------------|----------------------------|----------------------------------|-------------------------------|
| **MSE** | 高 / High | 低 / Low | 低 / Low | 训练时作为损失函数 / Loss function during training |
| **RMSE** | 高 / High | 高 / High | 低 / Low | 报告预测精度 / Reporting prediction accuracy |
| **MAE** | 低 / Low | 高 / High | 中 / Medium | 异常值较多时 / When outliers are present |
| **R²** | 中 / Medium | 高 / High | N/A | 评估整体拟合度 / Evaluating overall fit |

### 8.2 Metric Comparison and Selection

**Recommendation**: Use **multiple metrics** together for comprehensive evaluation:
- **MSE** for training (loss function)
- **RMSE** for reporting (same unit as temperature)
- **R²** for overall model quality assessment

### 8.3 为什么使用多个指标？

单一指标可能产生误导：

**案例 / Example**:
- 模型A: 99个预测误差为0.5°C，1个误差为10°C
  - MSE ≈ 1.23, RMSE ≈ 1.11°C, MAE ≈ 0.59°C
- 模型B: 所有100个预测误差均为1°C
  - MSE = 1.00, RMSE = 1.00°C, MAE = 1.00°C

**分析 / Analysis**:
- 仅看RMSE，模型B更好
- 但模型A在99%的情况下更准确
- **结论**: 需结合多个指标和实际应用场景判断

### 8.3 Why Use Multiple Metrics?

A single metric can be misleading. For example:
- Model A: 99 predictions with 0.5°C error, 1 with 10°C error → RMSE ≈ 1.11°C, MAE ≈ 0.59°C
- Model B: All 100 predictions with 1°C error → RMSE = 1.00°C, MAE = 1.00°C

**Conclusion**: Need to combine multiple metrics and consider real application scenarios.

---

## 9. 参考文献 / References

### 9.1 核心论文 / Core Papers

1. **Zhou, H., Zhang, S., Peng, J., Zhang, S., Li, J., Xiong, H., & Zhang, W. (2021)**.
   *Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting*.
   Proceedings of the AAAI Conference on Artificial Intelligence, 35(12), 11106-11115.
   **🏆 AAAI 2021 Best Paper Award**

2. **xPatch (2025)**.
   *xPatch: Dual-Stream Time Series Forecasting with Exponential Seasonal-Trend Decomposition*.
   Proceedings of the AAAI Conference on Artificial Intelligence, 2025.

3. **Nie, Y., et al. (2023)**.
   *A Time Series is Worth 64 Words: Long-term Forecasting with Transformers*.
   ICLR 2023. (PatchTST)

### 9.2 标准文档 / Standard Documents

4. **IEEE C57.91-2011**.
   *IEEE Guide for Loading Mineral-Oil-Immersed Transformers and Step-Voltage Regulators*.
   Institute of Electrical and Electronics Engineers.

5. **IEC 60076-7:2018**.
   *Power transformers - Part 7: Loading guide for mineral-oil-immersed power transformers*.
   International Electrotechnical Commission.

### 9.3 最新研究 / Recent Research (2024-2025)

6. **Boujamza, A., et al. (2025)**.
   *Predicting Oil Temperature in Electrical Transformers Using Neural Hierarchical Interpolation*.
   Journal of Engineering, 2025.

7. **Li, X., et al. (2024)**.
   *A New Deep Learning Architecture with Inductive Bias Balance for Transformer Oil Temperature Forecasting*.
   Journal of Big Data, 2023.

8. **Zhang, Y., et al. (2024)**.
   *Prediction of Transformer Oil Temperature Based on Feature Selection and Deep Neural Network*.
   IEEE Conference Publication, 2024.

### 9.4 数据集资源 / Dataset Resources

9. **ETT Dataset - GitHub**:
   https://github.com/zhouhaoyi/ETDataset

10. **ETT Dataset - Hugging Face**:
    https://huggingface.co/datasets/ett

11. **ETT Dataset - Papers with Code**:
    https://paperswithcode.com/dataset/ett

### 9.5 综述与教程 / Surveys and Tutorials

12. **Lim, B., & Zohren, S. (2021)**.
    *Time-series forecasting with deep learning: a survey*.
    Philosophical Transactions of the Royal Society A, 379(2194), 20200209.

13. **Torres, J. F., et al. (2021)**.
    *Deep learning for time series forecasting: a survey*.
    Big Data, 9(1), 3-21.

---

## 附录 / Appendix

### A. 术语对照表 / Glossary

| 中文 / Chinese | 英文 / English | 缩写 / Abbr. |
|---------------|----------------|--------------|
| 电力变压器 | Power Transformer | - |
| 油温 | Oil Temperature | OT |
| 有功功率 | Active Power / Real Power | P |
| 无功功率 | Reactive Power | Q |
| 视在功率 | Apparent Power | S |
| 功率因数 | Power Factor | PF |
| 高压 | High Voltage | HV |
| 中压 | Medium Voltage | MV |
| 低压 | Low Voltage | LV |
| 长序列时间序列预测 | Long Sequence Time-Series Forecasting | LSTF |
| 自注意力机制 | Self-Attention Mechanism | - |
| 循环神经网络 | Recurrent Neural Network | RNN |
| 长短期记忆网络 | Long Short-Term Memory | LSTM |
| 门控循环单元 | Gated Recurrent Unit | GRU |

### B. 常见问题 / FAQ

**Q1: ETTh和ETTm有什么区别？**
A: ETTh是1小时采样（17,520个数据点），ETTm是15分钟采样（70,080个数据点）。ETTm数据更细粒度，更适合捕捉短期波动。

**Q2: 为什么要预测油温而不是绕组温度？**
A: 油温更容易直接测量（通过温度传感器），而绕组温度通常需要通过油温间接估算。油温预测可作为绕组热点温度估算的基础。

**Q3: 本地数据集是ETTm1还是ETTm2？**
A: 从数值范围看，trans_1类似ETTm1（数值较小），trans_2类似ETTm2（数值较大）。但数据点数量略少于标准ETT（69,680 vs 70,080），可能是变体版本。

**Q4: 应该选择LSTM还是GRU？**
A: 对于油温预测任务，GRU通常是首选：训练速度快20-30%，参数少，性能接近LSTM。仅当需要捕捉非常长的依赖关系时才考虑LSTM。

---

**报告结束 / End of Report**

---

**声明 / Disclaimer**: 本报告基于公开文献和数据集信息编写，用于学术研究和课程项目。所有引用信息截至2025年10月，最新进展请参考相关论文和数据集官方页面。

**Disclaimer**: This report is based on publicly available literature and dataset information, intended for academic research and course projects. All cited information is current as of October 2025. For the latest developments, please refer to relevant papers and official dataset pages.
