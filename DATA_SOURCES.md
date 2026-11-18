# 数据来源与分析适用范围

本次核对使用 INDIflight 的 15b762626a35be4be4c9af92d138072ed36911f3 和 indiflightSupport 的 afebc0fa92e1f6a857e7dc83744e368bd1c28fa6。它们是核对时的上游版本，实验实际使用的提交及本地修改还需由实验记录确认。

## 当前字段选择

用 SITL 的 extPos/extVel/extAtt、REAL 的动捕 extPos/extVel/extAtt，以及 HIL 电脑仿真的位置、速度和姿态来评估轨迹、速度、姿态和过门表现，具有一致的物理含义。

ext 字段的可信程度由发送端决定。对当前实验，REAL 的发送端按实验说明为动捕系统，SITL 的标准 PyNDIflight 发送端来自仿真器状态。HIL 当前 CSV 的 x_ned/vx_ned/qw 等字段按实验说明来自电脑仿真，其具体导出脚本需要与实验代码一起保存，以便完整追溯。

## 上游代码对应关系

- Blackbox 的 extPos、extVel 分别记录 extPosNed.pos 和 extPosNed.vel，extAtt 记录 extPosNed.att。pos、vel 分别取 posEstNed、velEstNed。extAtt 实际乘以 1000 存储，单位为毫弧度，解码器再除以 1000 得到 rad。[Blackbox 状态记录](https://github.com/tudelft/indiflight/blob/15b762626a35be4be4c9af92d138072ed36911f3/src/main/blackbox/blackbox.c#L1831-L1856)
- 外部位置接收函数把发送端的 NED 位置、速度和四元数写入 extPosNed，四元数转为 roll、pitch、yaw。[外部状态接收](https://github.com/tudelft/indiflight/blob/15b762626a35be4be4c9af92d138072ed36911f3/src/main/io/external_pos.c#L92-L115)
- SITL 包装器的 sendMocap 直接发送 self.uav.xI、self.uav.vI、self.uav.q。固件的 setMocap 将其写入外部状态，所以这条标准链路中的 ext 字段是仿真状态经过传输、采样和日志量化后的记录。[仿真状态发送](https://github.com/tudelft/indiflightSupport/blob/afebc0fa92e1f6a857e7dc83744e368bd1c28fa6/Simulation/PyNDIflight/interfaces.py#L116-L128) · [固件接收](https://github.com/tudelft/indiflight/blob/15b762626a35be4be4c9af92d138072ed36911f3/src/main/mockup/main.c#L84-L103)
- PyNDIflight 的模拟动捕也直接读取 uav.xI、uav.vI、uav.q 后通过 UDP 发送，可供 HIL 外部定位链路使用。模型中的 q 初始值为 [1,0,0,0]，顺序为 w,x,y,z。[模拟动捕](https://github.com/tudelft/indiflightSupport/blob/afebc0fa92e1f6a857e7dc83744e368bd1c28fa6/Simulation/PyNDIflight/interfaces.py#L86-L111) · [模型状态](https://github.com/tudelft/indiflightSupport/blob/afebc0fa92e1f6a857e7dc83744e368bd1c28fa6/Simulation/PyNDIflight/crafts.py#L61-L72)
- Blackbox 的 quat 来自 getAttitudeQuaternion 读出的飞控内部姿态状态。当前仓库使用 extAtt，统一的是外部状态分析口径。[quat 记录](https://github.com/tudelft/indiflight/blob/15b762626a35be4be4c9af92d138072ed36911f3/src/main/blackbox/blackbox.c#L1800-L1804) · [内部姿态读取](https://github.com/tudelft/indiflight/blob/15b762626a35be4be4c9af92d138072ed36911f3/src/main/flight/imu.c#L672-L677)
- 官方导入器将位置除以 1000、速度除以 100、extAtt 除以 1000，得到 m、m/s、rad。当前程序接收已经完成这些换算的 CSV。[CSV 单位处理](https://github.com/tudelft/indiflightSupport/blob/afebc0fa92e1f6a857e7dc83744e368bd1c28fa6/LogAnalysis/indiflightLogTools/indiflight_log_importer.py#L309-L314)
- 固件使用 yaw-pitch-roll 的内禀 ZYX 旋转约定。当前 HIL 四元数转欧拉角的公式与该约定一致。[欧拉角公式](https://github.com/tudelft/indiflight/blob/15b762626a35be4be4c9af92d138072ed36911f3/src/main/common/maths.c#L135-L140)

## 采样与时间

标准仿真循环每 100 次迭代发送一次动捕状态，示例步长为 0.0005 s，对应 20 Hz。记录 CSV 的频率可以更高，期间外部状态保持上一帧。[仿真循环](https://github.com/tudelft/indiflightSupport/blob/afebc0fa92e1f6a857e7dc83744e368bd1c28fa6/Simulation/PyNDIflight/sim.py#L48-L52) · [示例步长](https://github.com/tudelft/indiflightSupport/blob/afebc0fa92e1f6a857e7dc83744e368bd1c28fa6/Simulation/exampleQuadSim.py#L154-L155)

当前 M23_kidney_speed6_1028.csv 约 500 Hz 记录，ext 状态变化间隔中位数为 0.050 s，与 20 Hz 一致。该 CSV 的 extTime 全为 0，状态本身有更新，更新率在此按状态变化估计。当前脚本的分析时钟使用 timeS。

实飞控制段的外部位置变化间隔中位数约 0.0211 s，P95 约 0.0406 s。当前 HIL 原生状态约 100 Hz 记录。不同采样率和样本保持会影响高频频谱、数值微分、峰值和过门时间精度。

组合报告用最短共同窗口，各组以各自分析段起点归零。该窗口适合比较整体分布，均速也包含启动阶段的影响。首门至末门的有序门段提供了更接近共同赛道区间的比较依据。当前首门至末门耗时分别为 SITL 4.299 s、HIL 4.315 s、REAL 4.566 s。

## 当前数据的核查结果

HIL 的 NN 段为 sim_s=7.0895–28.5945 s。四元数范数范围约为 0.99999991–1.00000014。相邻位置差分得到的区间平均速度，与两端记录速度均值的三轴平均绝对差约为 0.00159、0.00160、0.00079 m/s。这支持该 CSV 的位置、速度和时钟在数值上相互一致。

Real_Kidney_speed6_1028.csv 在原日志 timeS=13.636481–13.694284 s 有 28 个记录样本的 extAtt[0] 大于 90°，该时段处于所选控制段内。最大相邻 roll 跳变约为 125.0°，发生在 13.696653 s，新记录为 extAtt[0]=0.843 rad。需要结合动捕回放、刚体识别和机载传感器日志核查这段姿态变化。当前分析保留这些记录，分布、极值和标准差包含其影响。

## 结果解释

实飞动捕可作为高可信参考测量，实际精度仍受标定、刚体定位、传输延迟以及速度求导或滤波方式影响。仿真状态是仿真模型中的真值，Sim-to-Real 结论还取决于模型参数、执行器、传感器、时延、控制器和神经网络配置是否与实验对应。

当前方法适合评估轨迹、速度、姿态分布和有序过门表现。对逐点轨迹误差、控制延迟和高频动态的研究，还需要物理事件对齐、明确各通道时间戳及有效更新率。对电机的直接数值比较，还需要统一通道映射和转速单位。
