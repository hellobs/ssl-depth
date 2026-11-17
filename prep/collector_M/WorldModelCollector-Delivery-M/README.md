# WorldModelCollector Delivery M 采集改进版

## 1. 用途与分支来源

本目录用于采集面向配送任务的 AirSim World Model 数据，核心产物为有明确时间对应关系的转移序列

$$
(o_t,\ a_t,\ \Delta t_t,\ o_{t+1},\ m_t,\ e_t)
$$

$o_t$ 是观测，$a_t$ 是下发的速度设定值，$\Delta t_t$ 是实测仿真时长，$m_t$ 是可以重建奖励的测量量，$e_t$ 是碰撞、成功等事件。

基线位于

~~~text
D:\MetaEvo\UE4\AirSim-CityEnviron-UE4\WorldModelCollector
~~~

改进目录位于

~~~text
D:\MetaEvo\UE4\AirSim-CityEnviron-UE4\WorldModelCollector-Delivery-M
~~~

源目录在制作时没有 Git 仓库。本版采用独立目录分支，来源、基线 schema 和基线文件 SHA256 保存在 BRANCH.json。每次运行还会在 run.json 保存本版脚本 SHA256、环境、配置和随机种子。

本版 schema 为 **airsim-delivery-m.v1**，基线为 **airsim-world-model.v7-timed-episodes**。本版沿用控制进程、记录进程和原子文件通信结构，将整条路径动作改为逐步速度动作，加入配送任务参数与暂停步进采样。

采集参数速读见第 18 节，完整字段字典见第 8、9 节，控制脚本替换方法见第 17 节。

### 1.1 默认采集内容

| 类别 | 具体内容 |
| --- | --- |
| 画面 | 前视 RGB、下视 RGB |
| 几何画面 | 前视 DepthPlanar、下视 DepthPlanar，float32 米制数组与有效掩码 |
| 控制动作 | 前向、右向、向下速度设定值与偏航角速度，含裁剪、平滑前后值 |
| 状态 | 位置、四元数、偏航、线速度、角速度、线加速度、角加速度 |
| 配送目标 | 固定坐标、目标球半径、速度要求、连续停留要求 |
| 目标相对量 | 世界/航向/机体目标向量、距离、高度差、方位误差、接近速度 |
| 转移测量量 | 距离进展、位移、时间、静止持续时间、动作变化和控制量代理 |
| 结束信息 | 新碰撞、配送成功、超时、步数上限、terminated、truncated |
| 元数据 | 相机时间戳与位姿、FOV 与投影矩阵、设置、版本、种子和文件哈希 |

这里有两个相机方向和四个图像通道。深度提供几何信息，RGB 提供外观信息。

### 1.2 reward 的处理

采集文件保存构造 reward 所需的测量量和事件。observations.jsonl 与 transitions.jsonl 使用 reward-free 数据结构，校验器会检查其中出现的 reward 或 rewards 字段。

本设计可以表述为奖励延后重建

$$
r_t=f_\theta(m_t,e_t,g)
$$

$\theta$ 可以在训练时选择，也可以在另外的实验中学习。当前采集程序提供可重算的数据依据。第 12 节给出公式和离线计算示例。

## 2. 文件结构和维护位置

~~~text
WorldModelCollector-Delivery-M/
  RUN_WORLD_MODEL.cmd
  README.md
  BRANCH.json
  requirements.txt
  config/
    wm_settings.json
  scripts/
    collect_world_model.py
    world_model_path.py
    world_model_record.py
    world_model_airsim.py
    world_model_features.py
    world_model_session.py
    validate_dataset.py
  tests/
    test_delivery_features.py
  data/
    runs/
    validation/
~~~

| 文件 | 作用 |
| --- | --- |
| RUN_WORLD_MODEL.cmd | 查找 AirSim Python，检查导入，调用主入口，支持传入参数 |
| collect_world_model.py | 参数校验、模拟器连接、运行清单、创建两个进程、监测错误与收尾 |
| world_model_path.py | 重置、解锁、升高、初始朝向、目标设定、速度命令、推进仿真、最后悬停 |
| world_model_record.py | 快照、目标停留状态、测量量与事件、观测和转移写盘、采样确认 |
| world_model_airsim.py | 状态转换、相机信息、RGB/浮点深度写盘、取图时间检查 |
| world_model_features.py | 纯几何公式、目标采样、平滑控制策略、动作约束、奖励依据 |
| world_model_session.py | 原子 JSON 发布、逐步握手、停止信号、JSONL 写入 |
| validate_dataset.py | 对应关系、时间、公式、动作、结束边界和图像资产校验 |
| tests/test_delivery_features.py | 坐标符号、动作变换、停留、碰撞锁存和错误传播的回归测试 |

维护运动行为时主要查看 world_model_features.py 中的 BehaviorPolicy。维护采集字段时主要查看 world_model_airsim.py 和 world_model_record.py。修改数据定义后应同步更新校验器和 README。控制策略与整个控制进程的替换方法见第 17 节。

## 3. 运行环境与快速开始

### 3.1 已验证的环境

| 项目 | 版本或路径 |
| --- | --- |
| Python | 3.8.20 |
| Python 可执行文件 | C:\Users\31861\.conda\envs\ue4-airsim\python.exe |
| AirSim Python | 1.8.1 |
| NumPy | 1.24.3 |
| msgpack | 0.6.2 |
| msgpack-rpc-python | 0.4.1 |
| Tornado | 4.5.3 |
| 模拟器入口 | D:\MetaEvo\UE4\AirSim-CityEnviron-UE4\WindowsNoEditor\CityEnviron.exe |
| 默认 RPC | 127.0.0.1 的 41451 端口 |
| 默认车辆 | Drone1 |

启动器优先使用环境变量 AIRSIM_PYTHON 指定的可执行文件，其次使用用户目录下的 .conda\envs\ue4-airsim\python.exe。

需要重建环境时，可以在 Anaconda Prompt 中执行

~~~bat
conda create -n ue4-airsim python=3.8.20 -y
conda activate ue4-airsim
python -m pip install numpy==1.24.3 msgpack==0.6.2 msgpack-rpc-python==0.4.1 tornado==4.5.3
python -m pip install airsim==1.8.1
~~~

AirSim 安装元数据会导入依赖，因此上述顺序先安装 NumPy 和 RPC 依赖。现有 ue4-airsim 环境可以直接使用。

### 3.2 双击运行

双击本目录的 RUN_WORLD_MODEL.cmd。默认一个回合，最多 180 秒仿真时间，名义 0.2 秒一个动作区间，初始高度 30 米。

RPC 离线时，入口会使用本目录 config/wm_settings.json 启动 CityEnviron。RPC 已开启时会连接当前实例，并在 run.json 保存从服务端读取的运行时设置。需要改变相机设置时，应先关闭当前 CityEnviron，再通过启动器打开。

episode.json 的 settings_source 标明设置来源。服务端可以返回设置时使用 server_runtime，读取失败时使用 local_requested_unverified，并保存读取错误。图像本身的分辨率、时间戳和实际相机位姿仍由服务端返回值记录。

采集会重置场景和无人机。采集窗口内由控制进程拥有车辆控制权与暂停时钟，其他控制脚本或手动暂停操作会破坏时序检查。

### 3.3 PowerShell 命令

先设置工作目录与解释器

~~~powershell
Set-Location -LiteralPath 'D:\MetaEvo\UE4\AirSim-CityEnviron-UE4\WorldModelCollector-Delivery-M'
$airsimPython = 'C:\Users\31861\.conda\envs\ue4-airsim\python.exe'
~~~

短回合检查

~~~powershell
& $airsimPython '.\scripts\collect_world_model.py' --launch --steps 20 --seed 123
~~~

指定相对采集起点的目标。下面沿世界 NED 的 x 正方向飞 12 米，同时上升 2 米

~~~powershell
& $airsimPython '.\scripts\collect_world_model.py' --launch --goal-offset 12 0 -2 --goal-radius 3 --noise-scale 0 --episode-seconds 45 --seed 8
~~~

指定绝对目标坐标

~~~powershell
& $airsimPython '.\scripts\collect_world_model.py' --launch --goal-ned 60 30 -35 --seed 124
~~~

采集多个随机目标

~~~powershell
& $airsimPython '.\scripts\collect_world_model.py' --launch --episodes 10 --episode-seconds 180 --seed 125
~~~

只写参数和运行清单

~~~powershell
& $airsimPython '.\scripts\collect_world_model.py' --prepare-only --seed 126
~~~

prepare-only 在连接模拟器前返回，生成状态为 prepared 的 run.json，可用于审查配置。

参数也可以直接传给启动器

~~~powershell
& '.\RUN_WORLD_MODEL.cmd' --steps 30 --seed 127
~~~

完整输出目录会打印在终端中。每次运行使用时间与随机标识创建独立目录。

## 4. 两个进程怎样配合

控制进程和记录进程拥有各自的 AirSim RPC 客户端，通过 .ipc 中的原子 JSON 文件握手。

~~~mermaid
sequenceDiagram
    participant C as 控制进程
    participant S as AirSim
    participant R as 记录进程
    C->>S: 重置、升高、随机初始朝向
    C->>S: 暂停
    C->>R: 目标、相机信息、初始采样请求
    R->>S: 读取初始观测和四类图像
    R->>C: 初始采样确认
    loop 每个动作区间
        C->>S: 排队速度命令
        C->>S: 推进约 dt 秒并自动暂停
        C->>R: 命令参数、时间和下一次采样请求
        R->>S: 读取下一观测
        R->>R: 计算测量量与事件并写盘
        R->>C: 新观测、确认和结束状态
    end
    C->>S: 取消命令、恢复仿真、请求悬停
~~~

### 4.1 主入口的详细流程

1. 解析参数，检查时间、坐标、速度和高度范围，读取相机配置。
2. 选择根种子，创建 run 目录，保存环境、参数和代码哈希。
3. 检查 RPC，按 launch 参数启动模拟器或连接当前实例。
4. 检查车辆存在，尝试读取服务端实际 settings。
5. 每回合创建 episode 目录、episode.json 和 .ipc。
6. 先创建记录进程，再创建控制进程，记录进程等待初始化。
7. 检查进程退出码与错误信号，每约 30 秒显示已提交快照数。
8. 两个进程正常退出且 result.json 完整时，把回合标记为 complete。
9. 中断或异常写入 incomplete 和错误信息，并通知工作进程停止。
10. 多回合依次执行，每回合重新重置和产生目标及行为种子。

### 4.2 控制进程的详细流程

1. 连接 AirSim，恢复仿真并执行 reset()。
2. 开启 Drone1 的 API 控制，解锁车辆。
3. 使用基线已验证的 moveToZAsync()，直接升到指定 NED 高度。
4. 按种子旋转到随机初始偏航，调用悬停并短暂等待稳定。
5. 暂停仿真，读取初始状态，确定本回合固定目标。
6. 获取相机运行时 FOV、投影矩阵与位姿，发布初始化及初始采样请求。
7. 等待记录进程返回初始观测。
8. 根据目标误差产生期望速度，叠加相关噪声，执行边界与变化率约束。
9. 按当前偏航将前向、右向速度转换为世界 NED 的 vx、vy。
10. 在暂停状态排队 moveByVelocityAsync()，通过 RPC 往返刷新请求。
11. 用 simContinueForTime(dt) 推进物理，轮询到模拟器自动暂停。
12. 读取推进后时间，发布动作、时间和新采样请求。
13. 等待确认，使用确认中的新观测规划下一步。
14. 结束后取消命令、恢复仿真、请求悬停，记录清理结果。

初始化爬升与初始朝向调整位于采集窗口之前，累计时间从第 0 个观测开始。最后的清理悬停发生在采集窗口结束后。

### 4.3 记录进程的详细流程

1. 读取 episode 参数，等待目标和相机初始化信息。
2. 等待序号 0 的初始采样请求。
3. 确认模拟器暂停，读取取图前状态。
4. 在一次 simGetImages() 调用中请求四个图像通道。
5. 读取碰撞、环境和取图后状态。
6. 确认取图前后状态时间戳相同，并再次检查暂停状态。
7. RGB 写成 PNG，深度写成 float32 NPY，额外写入有效掩码。
8. 计算当前目标向量、距离、高度误差、方位和接近速度。
9. 从第二个观测起，用前后状态时间戳计算实测区间。
10. 更新目标停留时间、静止时间、位移和累计进展。
11. 判断新碰撞、配送成功、时间和步数限制。
12. 先写新观测，再写转移，刷新文件后发布确认。
13. 最后保留终点观测并写出 result.json。

记录完成后控制进程才推进下一步。两个进程独立维护，正常使用主入口统一启动与配对。

## 5. 动作究竟是什么

### 5.1 四维速度设定值

$$
a_t=[v_{\mathrm{forward}},v_{\mathrm{right}},v_{\mathrm{down}},\dot\psi]
$$

| 分量 | 单位 | 正方向 | 默认绝对上限 |
| --- | --- | --- | --- |
| forward | m/s | 区间起点的水平机头朝向 | 4 |
| right | m/s | 区间起点水平机头的右侧 | 3 |
| down | m/s | 世界 NED 向下 | 2 |
| yaw_rate | deg/s | NED 中绕向下轴的正向偏航，俯视顺时针 | 30 |

前向和右向使用只含偏航的水平坐标系。机体俯仰与横滚保存在状态中，offset_body_frd_m 另行提供完整姿态下的目标向量。

飞控根据速度误差产生姿态和电机输出，因此设定速度与实际速度会存在动态差异。动作标签采用实际下发的设定值，实际线速度属于状态和结果。

### 5.2 转换到 API

取区间起点偏航 $\psi_t$

$$
v_x=v_f\cos\psi_t-v_r\sin\psi_t,\quad
v_y=v_f\sin\psi_t+v_r\cos\psi_t,\quad v_z=v_d
$$

实际调用

~~~python
client.moveByVelocityAsync(
    vx, vy, vz, duration,
    drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
    yaw_mode=airsim.YawMode(True, yaw_rate_deg_s),
    vehicle_name="Drone1",
)
~~~

世界 vx、vy、vz 在区间内保持设定值。无人机可继续偏航，下一步使用新的偏航重新转换。action.api 保存实际世界速度和调用参数。

API 的 duration_s 默认是 dt + 1.0，用作任务存活余量。真正的采样动作窗口由前后观测间的实测时间定义。任务在暂停采样后由下一条命令替换，最后一步会取消。

### 5.3 action 字段

| 字段 | 语义 |
| --- | --- |
| names | 四个分量的名称与顺序 |
| frame | 航向水平 FRD 加世界 NED 向下分量 |
| requested | 策略与噪声产生的原始期望 |
| bounded | 分量裁剪及高度恢复后的期望 |
| applied | 进一步执行变化率限制后的下发动作 |
| normalized_applied | applied 分别除以 4、3、2、30 |
| normalization_scales | 对应除数和单位尺度 |
| previous_applied | 上一步 applied，第一步使用四个零 |
| slew_limits_per_s | 默认 2、2、1 m/s² 和 40 deg/s² |
| altitude_guards | 接近高度边界时采用的恢复标记 |
| exploration_noise | 本步内部相关噪声状态 |
| api | 世界速度、任务时长、偏航模式、车辆及 API 名称 |

M 的动作条件可使用 normalized_applied 或 applied。requested 适合分析策略意图和限幅。

slew 使用名义 dt 约束相邻设定值变化，实际时长另存在 timing。实际机体加速度由 state 记录。

## 6. 默认路线与目标产生方式

### 6.1 每回合固定目标

目标在初始快照前确定，在回合内固定。支持以下方式

- goal-ned 指定绝对世界 NED 坐标。
- goal-offset 以爬升、转向、稳定后的采集起点为基准，增加世界 NED 偏移。
- 默认按种子在起点水平距离 40～80 米的圆环采样，高度在初始高度附近变化 ±8 米，并裁剪到高度范围。

默认对半径和角度分别均匀采样。每回合随机化初始偏航，根种子默认每次运行随机产生。指定 seed 可以重现随机数，实际轨迹还会受到仿真调度影响。

episode.json 保存独立的目标、策略、初始朝向种子。第 $i$ 回合使用根种子加 $1009i$，再分别加 11、23、37。

### 6.2 平滑目标引导

默认 goal 策略按目标在航向坐标系下的偏移产生比例速度，比例系数为 0.4，水平巡航默认限制在 3 m/s。偏航角速度按方位误差产生，并执行限幅与变化率约束。

噪声具有时间相关性

$$
n_{t+1}=\alpha n_t+\sqrt{1-\alpha^2}\sigma\epsilon_t,\quad
\alpha=\exp(-\Delta t_{\mathrm{nominal}}/\tau)
$$

默认 $\tau=2$ 秒，四维内部噪声尺度为 $[0.7,0.7,0.25,5]$。goal 策略的 noise_scale 默认 0.35，并在目标附近随距离减弱噪声，便于稳定停留。

noise-scale 设为 0 可以观察纯目标引导。explore 策略使用相关随机速度探索，保留高度恢复约束。此模式仍记录目标关系，noise_scale 只作用于 goal 模式。

目标高度与当前位置不同会产生上升、下降或斜向飞行。

### 6.3 导航与任务能力边界

本版行为由比例引导、相关扰动、速度平滑和高度恢复产生。目标的 reachability_checked 为 false。建筑物可达性搜索、绕楼路径规划和深度避障是后续策略扩展点，碰撞会作为结束样本记录。

高度范围以 AirSim 原点为基准。地形和屋顶净空需要结合实际深度或地图处理。高度恢复是带变化率约束的软恢复，轨迹可能短时超过边界。

配送成功定义为空中目标区域低速停驻。取货、载荷、落地接触和卸货属于可扩展的任务阶段。

## 7. 数据目录和 N+1 对应关系

~~~text
data/runs/run_年月日_时分秒_标识/
  run.json
  episode_0000/
    episode.json
    task.json
    result.json
    control.log
    record.log
    observations.jsonl
    transitions.jsonl
    observations/
      000000_front_rgb.png
      000000_down_rgb.png
      000000_front_depth.npy
      000000_front_depth_valid.npy
      000000_down_depth.npy
      000000_down_depth_valid.npy
      ...
    .ipc/
      initialized.json
      request_000000.json
      ack_000000.json
      request_000001.json
      ack_000001.json
      ...
      cleanup.json
~~~

$N$ 个动作区间对应 $N$ 条转移和 $N+1$ 个观测

~~~text
observation 0 -- action 0 --> observation 1
observation 1 -- action 1 --> observation 2
...
observation N-1 -- action N-1 --> observation N
~~~

转移通过 observation_id 与 next_observation_id 引用前后观测。资产路径相对于 episode 目录，前后转移复用公共观测。

.ipc 用于运行同步与诊断。训练主要读取两个 JSONL、task.json 和图像资产。校验器会拒绝 incomplete 回合。中断可能留下已写观测、未提交转移或图像，原文件保留用于排查。

## 8. 观测字段字典

### 8.1 顶层

| 字段 | 含义 |
| --- | --- |
| observation_id | 从 0 开始的快照序号 |
| elapsed_sim_s | 相对于初始快照的仿真时间 |
| state | 无人机运动状态 |
| goal | 当前状态相对于固定目标的条件量 |
| collision | 当前 AirSim 碰撞状态 |
| environment | 环境与地理位置 |
| images | 四个通道的路径、时间、姿态和几何信息 |
| timing | 取图前后状态时间及图像偏差 |

JSON 保留 API 数值精度，时间戳为整数纳秒。Python 整数可以直接保存这些时间戳，JavaScript 读取时需关注大整数表示能力。

### 8.2 state

| 字段 | 单位或表示 | 说明 |
| --- | --- | --- |
| timestamp_ns | ns | AirSim 状态时间，以相邻差值计算区间 |
| position_ned_m | [x,y,z]，m | 世界 NED 位置 |
| orientation_xyzw | [x,y,z,w] | 机体到世界的四元数 |
| yaw_rad | rad | 从四元数提取的偏航 |
| linear_velocity_ned_m_s | m/s | 世界 NED 实际线速度 |
| angular_velocity_api_rad_s | rad/s | AirSim 原生角速度 |
| linear_acceleration_ned_m_s2 | m/s² | AirSim kinematics 世界 NED 线加速度 |
| angular_acceleration_api_rad_s2 | rad/s² | AirSim 原生角加速度 |
| landed_state | 枚举整数 | 默认 API 中 Landed 为 0，Flying 为 1 |
| ready | 原生布尔值 | 客户端可能提供默认值，用于诊断 |

角运动量以 api 命名保留原始分量。联合计算时应根据实际 AirSim 服务端版本确认其坐标约定，run.json 也保留此坐标说明。

状态源为 getMultirotorState().kinematics_estimated，可作为模拟器辅助监督。接入真实无人机时，应依据定位、视觉里程计和惯性测量可用性选择模型输入。

### 8.3 goal

设当前位置为 $p_t$、固定目标为 $g$、目标偏移为 $\delta_t=g-p_t$。

| 字段 | 单位 | 说明 |
| --- | --- | --- |
| offset_ned_m | m | 世界 NED 目标偏移 |
| offset_heading_frd_m | m | 只按偏航旋转得到的前、右、下向量 |
| offset_body_frd_m | m | 按完整四元数逆旋转得到的机体目标向量 |
| distance_3d_m | m | $\|\delta_t\|_2$ |
| distance_horizontal_m | m | $\sqrt{\delta_x^2+\delta_y^2}$ |
| height_error_up_m | m | $-\delta_z$，正值表示目标更高 |
| bearing_error_rad | rad | 水平目标方位减偏航，归一到 $[-\pi,\pi]$ |
| bearing_error_sin、bearing_error_cos | 无量纲 | 方位误差的连续表示 |
| bearing_valid | bool | 水平距离大于 $10^{-6}$ 米 |
| closing_speed_m_s | m/s | $\delta_t^\top v_t/\|\delta_t\|$，正值为接近 |
| inside_goal | bool | 三维距离小于等于目标半径 |
| speed_3d_m_s | m/s | 实际速度模长 |
| speed_below_delivery_threshold | bool | 实际速度满足投递阈值 |
| dwell_sim_s | s | 已确认的连续目标低速停留时间 |

水平距离退化时 bearing_error 设为 0，bearing_valid 设为 false。距离近零时 closing_speed 设为 0。

航向变换

$$
\delta_f=\cos\psi\,\delta_x+\sin\psi\,\delta_y,\quad
\delta_r=-\sin\psi\,\delta_x+\cos\psi\,\delta_y,\quad \delta_d=\delta_z
$$

完整 body 偏移进一步考虑俯仰和横滚，三个版本都保留供不同表示使用。

### 8.4 task.json 中的固定目标

| 字段 | 含义 |
| --- | --- |
| goal.position_ned_m | 本回合固定目标 |
| goal.radius_m | 三维目标球半径，默认 6 米 |
| goal.dwell_required_sim_s | 停留要求，默认 2 秒仿真时间 |
| goal.max_delivery_speed_m_s | 最大投递停驻速度，默认 0.5 m/s |
| goal.sampling_method | 绝对、相对或随机圆环来源 |
| goal.seed | 目标种子 |
| goal.initial_distance_3d_m | 初始三维距离 |
| goal.reachability_checked | 当前几何采样为 false |
| initial_state | 初始完整状态 |
| initial_yaw_requested_deg | 随机初始朝向 |
| cameras | 运行时标定信息和名义安装配置 |

### 8.5 images 与相机几何

各通道包含 camera、image_type、timestamp_ns、width、height、camera_world_pose、intrinsics_K_estimated、path 和 encoding。

- front_rgb 和 down_rgb 为 Scene PNG。
- front_depth 和 down_depth 为 DepthPlanar float32 NPY。
- 深度额外包含 units、valid_mask_path 和 summary。
- summary 保存有效比例、最小值、中位数、10% 分位数及中央三分之一区域的 10% 分位数。空有效区域使用 null。

DepthPlanar 单位为米，描述相机光轴方向的平面深度。斜向射线长度可以结合内参换算。float32 保存服务端原值，超过 65.535 米的数据也会保留。

掩码定义为数值有限且大于零。天空或超远区域的有限远平面数值可能通过掩码，训练预处理应依据场景含义选择有效距离范围。

内参估计采用运行时水平 FOV 与正方形像素假设

$$
f_x=f_y=\frac{W}{2\tan(\mathrm{FOV}_x/2)},\qquad c_x=W/2,\quad c_y=H/2
$$

矩阵标为 estimated。task.json 另保留运行时投影矩阵，精确重投影应结合 AirSim/UE 相机轴与像素中心约定。

相机沿用基线云台稳定配置。每帧保存实际 camera_world_pose，用于处理动态相机姿态。task.json 保存名义安装、运行时 FOV、投影矩阵与初始位姿。

### 8.6 collision、environment 与 timing

collision 包含 has_collided、timestamp_ns、object_name、object_id、normal_api、impact_point_ned_m、position_ned_m、penetration_depth_m。碰撞布尔值可能保留历史状态，新事件规则见第 11 节。

environment 保存位置、重力向量、气压 Pa、温度 K、空气密度 kg/m³、经纬度和地理高度。地理高度与本地 NED 使用不同原点，目标误差统一由本地 NED 计算。

run.json 中 weather_parameters 和 wind_ned_m_s 为 null，表示入口没有取得这些设置。环境 API 测量值随观测记录。

| timing 字段 | 含义 |
| --- | --- |
| state_before_images_ns | 取图前状态时间 |
| state_after_images_ns | 取图后状态时间 |
| image_timestamp_spread_s | 同组图像最大时间差 |
| image_minus_state_s | 各图像时间减状态时间 |
| capture_and_save_wall_s | RPC 取图与写盘现实耗时 |

取图前后状态时间不同会中止采集并标记 incomplete。相机与状态偏差超过 20ms 会产生质量标记。

## 9. 转移字段和 reward 依据

### 9.1 顶层与 timing

| 字段 | 含义 |
| --- | --- |
| transition_id | 从 0 开始的区间序号 |
| observation_id、next_observation_id | 前后观测引用 |
| action | 原始期望、约束后动作和实际 API 参数 |
| timing | 名义/实测时间和下发时刻 |
| measurements | 几何、运动和控制量依据 |
| events | 碰撞、成功、时间和步数事件 |
| terminated、truncated、done | 结束类型 |
| end_reasons | 原因列表 |
| quality | 对齐与可用性标记 |

| timing 字段 | 含义 |
| --- | --- |
| nominal_step_sim_s | 请求推进的仿真时长 |
| actual_step_sim_s | 相邻状态时间戳差 |
| step_error_sim_s | 实测减名义时长 |
| command_queued_at_sim_ns | 初始观测对应的排队时刻 |
| state_after_advance_ns | 推进后暂停状态时间 |
| advance_wall_s | 推进阶段现实耗时 |
| action_end_semantics | 暂停采样后由下一条命令替换或取消 |

默认名义频率为 $1/0.2=5$ Hz，每个观测都有四个通道。物理帧量化会造成实际区间差异，读取时使用 actual_step_sim_s。

仿真在取图、计算和写盘期间暂停，因此 180 秒仿真采集可能需要显著更长的现实时间。操作系统时间只用于性能诊断、超时和进度，物理时间增量采用状态时间戳。

### 9.2 measurements 完整字典

| 字段 | 单位 | 计算或用途 |
| --- | --- | --- |
| time_alive_increment_sim_s | s | 当前区间实测仿真时长 |
| goal_progress_3d_m | m | $d_t-d_{t+1}$，正值为接近 |
| goal_progress_horizontal_m | m | 水平距离的前后差 |
| absolute_height_error_reduction_m | m | 绝对高度误差的前后差 |
| goal_progress_rate_m_s | m/s | 三维进展除以实测时长 |
| displacement_ned_m | m | $p_{t+1}-p_t$ |
| displacement_heading_frd_m | m | 按起点偏航旋转后的位移 |
| displacement_3d_m | m | 前后位置直线弦长 |
| displacement_horizontal_m | m | 水平位移弦长 |
| average_translation_speed_m_s | m/s | 三维弦长除以时长 |
| speed_before_m_s、speed_after_m_s | m/s | 边界实际速度模长 |
| vertical_speed_down_after_m_s | m/s | 终点向下速度 |
| goal_dwell_before_sim_s、goal_dwell_after_sim_s | s | 前后连续投递停留时间 |
| inside_goal_before、inside_goal_after | bool | 前后是否位于目标球 |
| stationary_duration_sim_s | s | 边界低速条件连续成立的时间 |
| action_delta | 原分量单位 | applied 减 previous_applied |
| action_delta_l2_normalized | 无量纲 | 按动作尺度归一后的变化模长 |
| normalized_command_squared_integral | 归一控制量平方 × s | $\|\bar a_t\|_2^2\Delta t_t$ |
| cumulative_sampled_path_length_m | m | 从初始快照累积位移弦长 |
| cumulative_goal_progress_m | m | 从初始快照累积三维距离进展 |

normalized_command_squared_integral 是控制强度代理，真实电耗需要电池模型或电机功率数据。

累计路径长度采用采样折线。区间内发生弯曲或回摆时，会低估高频真实弧长。

stationary_duration_sim_s 要求区间两个边界实际速度都低于 stationary_speed，默认 0.3 m/s。目标外停滞、无进展与原地绕圈可以结合窗口位移、路径长度和进展重新定义。

距离进展使用终点信息，位于 transition.measurements。模型当前输入应只使用当前观测和既有历史，未来进展可以作为监督标签或离线奖励依据。

## 10. 连续动作如何对齐采样

区间起点保存观测，排队一条速度设定值，推进短时间后暂停并采集终点。底层飞控在区间中持续调节姿态与电机。

数据中的 action 是该区间外部设定值，模型学习带飞控闭环的无人机动态

$$
p(o_{t+1}\mid o_{\le t},a_{\le t},\Delta t_t,g)
$$

可以给 M 输入 actual_step_sim_s，或按训练规范对数据做时间重采样。

读取一个样本

~~~python
import json
from pathlib import Path
import numpy as np

episode = Path(r"D:\MetaEvo\UE4\AirSim-CityEnviron-UE4\WorldModelCollector-Delivery-M\data\runs\替换成实际run目录\episode_0000")
observations = [
    json.loads(line)
    for line in (episode / "observations.jsonl").read_text(encoding="utf-8").splitlines()
]
transitions = [
    json.loads(line)
    for line in (episode / "transitions.jsonl").read_text(encoding="utf-8").splitlines()
]
transition = transitions[0]
current = observations[transition["observation_id"]]
next_observation = observations[transition["next_observation_id"]]
action = np.asarray(transition["action"]["normalized_applied"], dtype=np.float32)
dt = transition["timing"]["actual_step_sim_s"]
front_path = episode / current["images"]["front_rgb"]["path"]
next_front_path = episode / next_observation["images"]["front_rgb"]["path"]
depth = np.load(episode / current["images"]["front_depth"]["path"], allow_pickle=False)
valid = np.load(episode / current["images"]["front_depth"]["valid_mask_path"], allow_pickle=False)
~~~

## 11. 碰撞、成功和结束判定

### 11.1 新碰撞

当前 has_collided 为 true，并且满足以下任意条件时，events.collision 为 true

- 前一观测 has_collided 为 false。
- 当前碰撞时间戳与前一观测不同。

这样可以识别锁存状态中的新接触。区间内发生碰撞时保存终点快照，碰撞时间戳、接触位置和物体名提供额外依据。接触时刻的时间源兼容性应随服务端版本确认。

### 11.2 配送成功

同时满足三条

1. 三维距离小于等于 radius_m。
2. 实际速度小于等于 max_delivery_speed_m_s。
3. 已确认连续停留达到 dwell_required_sim_s。

两个采样边界都满足条件时才累加区间，首次进入区域从零开始，任一当前条件失效即清零。该定义按采样边界保守确认，区间内部短暂离开需要更高频观测才能识别。

### 11.3 结束类型

| 情况 | terminated | truncated |
| --- | --- | --- |
| 新碰撞 | true | false |
| 配送成功 | true | false |
| 时间或步数上限 | false | true |
| RPC 失败、图像异常或人为中断 | 回合状态为 incomplete | 由错误状态表示 |

done 等于 terminated 或 truncated。碰撞与成功事件分别保留，同一区间可以同时成立。物理终止优先决定 terminated，时间/步数达到的事实仍保留在 events。

最后一条转移保留终点观测。RNN 在回合边界重置隐藏状态。RL 对截断样本的 bootstrap 处理应依据算法区分 terminated 和 truncated。

## 12. 从测量量重建 reward

下面表示一种可实验的奖励族，系数在后续训练中选择。

设 $\Delta d_t$ 为进展，$C_t$ 为新碰撞，$S_t$ 为成功，$H_t$ 为目标外低速停滞指示，$\bar a_t$ 为归一动作

$$
r_t=
w_p\Delta d_t+w_a\Delta t_t-w_hH_t\Delta t_t
-w_cC_t+w_sS_t-w_u\|\bar a_t\|_2^2\Delta t_t
$$

生存时间可用 time_alive_increment_sim_s。$H_t$ 可以基于连续低速时间，也可以根据多个点的窗口位移、累计路径和目标进展构造。

目标区域内低速停留用于完成配送，停滞逻辑应结合 inside_goal。单纯奖励生存时间容易偏向拖延，进展、成功、任务时长与停滞原始量便于对比不同奖励组合。

下面只在内存中计算，适合放进训练预处理。weights 由实验指定

~~~python
def reconstruct_reward(transition, next_observation, weights):
    measurements = transition["measurements"]
    events = transition["events"]
    dt = transition["timing"]["actual_step_sim_s"]
    stalled_outside = (
        measurements["stationary_duration_sim_s"] >= 2.0
        and not next_observation["goal"]["inside_goal"]
    )
    return (
        weights["progress"] * measurements["goal_progress_3d_m"]
        + weights["alive"] * dt
        - weights["stall"] * float(stalled_outside) * dt
        - weights["collision"] * float(events["collision"])
        + weights["success"] * float(events["delivery_success"])
        - weights["command"] * measurements["normalized_command_squared_integral"]
    )
~~~

当前任务的低速阈值、目标区域和停留时间用于在线事件判定。原始位置、速度、目标坐标和时刻允许重新标注其他定义。

采集在当前定义的成功或碰撞处结束。改用更长停留要求时，需要评估已有轨迹在原终点之后的数据覆盖，必要时调整参数补采。

## 13. 用于 World Model 的 M

一种数据流程

1. 用前视/下视 RGB 训练编码器得到 $z_t$。深度可加入编码器或作辅助预测目标。
2. 从当前观测取速度、姿态和目标关系等条件。
3. 读取 applied 或 normalized_applied，带上 actual_step_sim_s。
4. 用当前编码、历史和动作预测 $z_{t+1}$，可同时预测速度或接触概率。
5. 按回合切序列，在结束边界重置循环隐藏状态。
6. 后续控制优化或价值学习按保存的测量量重建 reward，也可以训练奖励预测头。

输入示意

$$
x_t=[z_t,v_t,\omega_t,\delta_t^{heading},d_t,\sin\beta_t,\cos\beta_t]
$$

$$
(z_{t+1},h_{t+1})\sim M_\phi(x_t,a_t,\Delta t_t,h_t)
$$

M 的核心监督是实际前后观测动态，固定目标可以作为回合条件。选择状态输入时应依据后续部署可获得的传感信息。

## 14. 默认参数

| 参数 | 默认 | 含义 |
| --- | --- | --- |
| --episodes | 1 | 回合数 |
| --episode-seconds | 180 | 从初始快照开始的最长仿真时间 |
| --steps | 0 | 动作上限，0 表示仅用时间限制 |
| --dt | 0.2 | 名义区间，支持 0.05～1 秒 |
| --altitude | 30 | 初始原点相对高度，m |
| --min-altitude、--max-altitude | 10、60 | 高度恢复和目标采样范围 |
| --goal-min-distance、--goal-max-distance | 40、80 | 随机目标水平距离，m |
| --goal-height-jitter | 8 | 目标高度扰动，m |
| --goal-radius | 6 | 三维目标球半径，m |
| --goal-dwell | 2 | 投递停留，仿真 s |
| --delivery-speed | 0.5 | 投递低速阈值，m/s |
| --stationary-speed | 0.3 | 静止持续时间阈值，m/s |
| --policy | goal | goal 或 explore |
| --cruise-speed | 3 | goal 水平巡航上限，m/s |
| --noise-scale | 0.35 | goal 的探索扰动强度 |
| --noise-tau | 2 | 噪声相关时间常数，s |
| --seed | 每次随机 | 根种子，运行时打印并保存 |
| --out | 本目录 data/runs | 输出父目录 |
| --ip、--port | 127.0.0.1、41451 | RPC |
| --vehicle | Drone1 | 无人机名称 |
| --launch | 关闭 | 离线时启动本地 CityEnviron，CMD 默认打开 |
| --exe | 相邻 WindowsNoEditor/CityEnviron.exe | 模拟器入口 |
| --settings | config/wm_settings.json | 相机与模拟器设置 |
| --prepare-only | 关闭 | 只写运行参数 |

成功、碰撞、时间和步数限制中首先成立的条件结束回合。完整选项可以用主入口 --help 查看。

## 15. 校验和实测

校验整个 run

~~~powershell
& $airsimPython '.\scripts\validate_dataset.py' 'D:\MetaEvo\UE4\AirSim-CityEnviron-UE4\WorldModelCollector-Delivery-M\data\runs\替换成实际run目录'
~~~

快速检查 JSON 对应关系和公式

~~~powershell
& $airsimPython '.\scripts\validate_dataset.py' '替换成实际run目录' --metadata-only
~~~

回归测试

~~~powershell
& $airsimPython -m unittest discover -s '.\tests' -v
~~~

校验包括 N+1 数量、序号、时间单调性、动作起止对齐、坐标变换、动作限幅与变化率、距离和进展、停留、新碰撞、结束边界、深度格式、有效掩码和 PNG 尺寸。

quality.step_error_exceeds_25pct 表示实测区间偏差超过名义值 25%。camera_state_skew_exceeds_20ms 检查前后两组观测的图像与状态偏差。valid_for_dynamics 在相机偏差通过时为 true，训练可据此过滤并进一步检查。

2026-11-16 在本机 CityEnviron 上完成

- 步数截断实采得到 5 条转移、6 组观测，约 1.002 秒仿真时间，完整校验通过。
- 配送实采采用偏移 [12,0,-2]、半径 3 米、噪声 0，得到 46 条转移、47 组观测，约 9.183 秒触发成功，最终距离约 0.090 米，完整校验通过。
- 通过 CMD 启动器连续完成两个随机目标回合，均按 0.45 秒仿真时限截断，各记录 3 条转移，完整校验通过。
- 关闭测试模拟器后，再由 CMD 自动启动 CityEnviron 并完成采样，验证了离线启动到写盘的完整入口。
- 纯特征与通信回归测试共 15 项，全部通过。

实采数据位于 data/validation，目录名与运行状态以实际文件为准，正式训练可使用 data/runs。

validation 中还保留了早期起飞接口兼容性调试产生的 incomplete 记录，以及 prepare-only 参数检查产生的 prepared 记录。训练与完整数据校验应选择 status 为 complete 的运行目录。初始化现已沿用基线的 moveToZAsync 直升流程，并通过后续实采验证。

## 16. 速度、恢复和扩展

默认每个快照保存两张 RGB、两张 float32 深度和两张掩码。仅深度与掩码约 0.625 MiB，RGB 和 JSON 另计。180 秒约 900 个动作区间，磁盘占用随图像压缩和结束时刻变化。

暂停取图将写盘耗时排除在运动时间之外，现实采集速度取决于 GPU、RPC 和磁盘。吞吐优化可从分辨率、深度存储和批量写入着手，并保留动作、前后观测与实测时间契约。

遇到异常时查看 run.json 与 episode.json 的 status 和错误，再查看 control.log、record.log 与 .ipc/error.json。重新运行会创建独立 run，原始中断回合保持其状态以便诊断。

清理结果保存在 .ipc/cleanup.json 并汇总到 result.json。正常结束恢复仿真并请求悬停，车辆保持 API 控制。系统强制结束进程后，应检查 CityEnviron 的暂停和控制状态再继续采集。

后续可以在 BehaviorPolicy 前加入地图或深度规划，在记录模块增加电机与 IMU 测量，并在训练侧尝试多种奖励或任务重标注。扩展时继续保存 applied 动作、实测时间、终点观测和明确的事件边界。

## 17. 控制脚本替换与采集接口

### 17.1 当前版本的接入方式

当前控制进程为 scripts/world_model_path.py，记录进程为 scripts/world_model_record.py。记录进程通过 initialized.json 获取回合初始化信息，按 request/ack 协议等待每次采样请求。控制进程同时负责车辆命令和仿真暂停、推进。

因此，当前替换入口需要保留逐步握手与动作记录协议。任意外部脚本直接调用 AirSim 飞行时，记录进程仍会等待对应请求，最终可能报等待超时。

| 替换范围 | 接入位置 | 需要维护的契约 |
| --- | --- | --- |
| 路线、目标引导、随机探索或学习策略 | BehaviorPolicy 或兼容策略类 | 四维动作返回结构、单位、限幅与坐标变换 |
| 整个控制进程 | world_model_path.py 或主入口选用的新文件 | 初始化、request/ack、暂停推进、动作参数、结束与清理 |
| 外部持续飞行脚本、人工控制或多频率命令 | 需要扩展执行与记录接口 | 命令事件日志、仿真时间对齐、区间动作片段和生命周期 |
| 电机、姿态或整条路径动作 | 需要扩展数据 schema | 动作类型、参数、归一化、执行 API、测量量与校验规则 |

对后续频繁更换路线算法的需求，可以先采用兼容策略类接入，复用现有逐步执行和采集流程。

### 17.2 更换规划或决策策略

当前调用关系为

~~~text
world_model_path.run()
    → BehaviorPolicy(config, policy_seed)
    → policy.command(observation, previous_applied)
    → moveByVelocityAsync()
    → 推进仿真并等待记录确认
~~~

推荐的接入步骤

1. 在 scripts 中放置自己的策略模块，例如 my_policy.py。
2. 提供一个策略类，接收配置与随机种子，并实现 command(observation, previous_applied)。
3. 让 command 返回第 5.3 节的完整 action 字典。
4. 在 world_model_path.py 中切换策略导入与实例化位置。
5. 通过原主入口进行短回合验证，再使用 validate_dataset.py 检查结果。

假设自定义模块已经实现兼容的 MyPolicy，接入位置如下

~~~python
from my_policy import MyPolicy

# 在 run() 中，读取初始 ack 后创建策略。
policy = MyPolicy(config, metadata["seeds"]["policy"])

# 后续每步沿用现有调用位置。
action = policy.command(observation, previous_action)
~~~

上例展示接入位置，MyPolicy 的规划逻辑由使用者提供。策略实例可以保存航点序号、局部地图、循环网络隐藏状态等回合内状态，每回合重新创建实例。

也可以直接修改现有 BehaviorPolicy.command()。其中 requested 生成部分决定路线行为，后续 bounded、applied、世界速度变换与动作字典负责约束和记录。修改时应让 requested 表达新策略输出，并让实际执行与记录继续使用最终 applied。

command 调用期间仿真处于暂停状态。规划计算可以读取当前状态和历史数据，车辆执行与仿真推进由外层控制流程统一调度。当前观测中的 images 保存资产路径和元数据，读取图像时以 episode_dir 为基准解析路径；使用视觉规划器时，可在策略构造函数中额外传入 episode_dir。

### 17.3 策略返回值的约定

兼容当前 schema 的策略应返回完整动作字典，字段详见第 5.3 节。以下关系会影响执行、测量或数据校验

| 内容 | 约定 |
| --- | --- |
| applied | 长度为 4，顺序为前向速度、右向速度、向下速度、偏航角速度 |
| 单位 | 前三项 m/s，最后一项 deg/s |
| 坐标 | 前向和右向采用区间起点的水平航向，向下采用世界 NED |
| normalization_scales | 当前版本为 [4,3,2,30] |
| normalized_applied | applied 按上述尺度逐分量归一化 |
| previous_applied | 上次实际下发动作，第一步为 [0,0,0,0] |
| 动作变化率 | 当前版本按名义 dt 与 [2,2,1,40] 约束 |
| api.velocity_ned_m_s | applied 前三项按当前偏航转换后的世界速度 |
| api.yaw_or_rate_deg_s | 与 applied 第四项一致 |
| api.duration_s | 默认 dt + 1.0，覆盖本次推进并留出任务存活余量 |
| 有限数值 | 所有 JSON 数值应为有限值，write_json 会拒绝 NaN 和无穷值 |

保留 names、frame、requested、bounded、normalization_scales、slew_limits_per_s、altitude_guards、exploration_noise 和完整 api 元数据，使行为来源可追溯。新策略的探索元数据应描述其实际行为，确定性策略可使用四维零噪声记录。

世界速度与归一化公式可复用 world_model_features.py 中的 heading_to_world、ACTION_SCALES 和 SLEW_LIMITS。

当前执行层固定调用 moveByVelocityAsync。更改 action.api.method 字符串时，也需要同步实现对应执行分支和数据定义。航点规划器可以先把航点转成四维速度命令，再交给现有执行层，沿用当前 M 的动作空间。

### 17.4 更换整个控制进程时的协议

主入口 collect_world_model.py 的 run_episode() 通过进程映射选择文件

~~~python
for name, script in (
    ("record", "world_model_record.py"),
    ("control", "world_model_path.py"),
):
    ...
~~~

采用新控制文件时，可以修改 control 对应文件名，并让新文件支持 --episode-dir 参数。主入口会把它与记录进程绑定到同一 episode 目录，收集日志并监控退出状态。

新控制进程应维持以下顺序。

#### A. 读取参数和完成初始化

读取 episode.json 中的 config、settings、seeds。完成所需的 reset、API 控制、解锁、初始位置和朝向准备，然后暂停仿真。

发布以下文件，使用 world_model_session.write_json() 原子写入。

| 文件 | 内容 |
| --- | --- |
| .ipc/initialized.json | goal、cameras、initial_timestamp_ns，以及初始状态和朝向元数据 |
| task.json | 固定目标、相机信息、初始状态与朝向，供训练和校验读取 |
| .ipc/request_000000.json | 初始请求，内容为 {"kind":"initial"} |

可以复用 sample_goal()、calibrations() 和 state_dict()。当前记录进程在回合开始读取一次固定目标，因此回合内动态更换目标需要额外扩展目标版本及生效时间记录。

#### B. 等待初始观测

通过 wait_json() 等待 .ipc/ack_000000.json。ack 包含 observation、stop 和 result。初始 observation 对应编号 0，上一动作初始化为四个零。

#### C. 每步执行与采样

对于请求序号 i，i 从 1 开始

1. 使用上次 ack 中的 observation 计算动作。
2. 确认仿真暂停，命令排队时的状态时间与该观测一致。
3. 下发与 action.api 参数一致的速度命令，刷新 RPC 请求。
4. 推进名义 dt 仿真时间，等待重新暂停。
5. 读取推进后状态时间，发布 request_i。
6. 等待 ack_i，更新 previous_applied。
7. ack.stop 为 true 时结束回合。

请求文件名采用六位补零序号，例如 request_000001.json。对应载荷为

~~~python
request = {
    "kind": "transition",
    "action": action,
    "command_queued_at_sim_ns": queued_at,
    "state_after_advance_ns": int(state_after.timestamp),
    "advance_wall_s": advance_wall_s,
}
~~~

请求 i 产生 transition_id 为 i-1 的转移，其起点观测为 i-1，终点观测为 i。记录进程完成写盘后发布 ack_i，保证 N 个区间对应 N+1 个观测。

推进前等待异步速度任务完成会阻止物理执行。现有流程采用排队命令、推进仿真、暂停采样的顺序，任务随后由下一条命令替换。

#### D. 错误、退出与清理

循环中使用 wait_json() 和现有 advance()，或实现等价的错误、STOP 与超时检查。当前 wait_json 默认等待上限为 120 秒，advance 的推进等待上限为 30 秒，更慢的规划或执行流程需要统一调整这些超时。

记录进程负责配送成功、碰撞、时间及步数结束判定。控制进程依据 ack.stop 收尾，写盘完成的 result.json 由主入口检查。

控制异常应写入 .ipc/error.json 并以失败状态退出。清理阶段取消当前任务、恢复仿真、请求悬停，并写入 .ipc/cleanup.json。新回合使用新的 episode 目录，避免旧握手文件混入当前序列。

### 17.5 外部脚本与后续独立采集架构

后续若希望外部控制脚本按自己的频率持续运行，可以增加统一命令发布层和独立仿真协调器

~~~text
任意路线或决策脚本
    → 统一命令执行与发布层
        → AirSim
        → 动作事件日志

独立采集器
    → 图像、状态、碰撞
    → 按仿真时间匹配动作事件
    → 生成观测与转移

仿真协调器
    → 回合生命周期、暂停推进或连续运行时钟
~~~

此处描述后续接入设计。当前可运行入口采用本节前述的 request/ack 暂停步进协议。

动作发布层至少应保存命令 ID、动作类型、实际参数、坐标与单位、下发时仿真时间、请求持续时间，以及替换、取消、失败事件。异步命令还需区分下发时刻与可确认的执行边界。

当一个采样区间出现多个命令时，应保留区间内的动作片段和各自覆盖时间。训练可以进一步切分转移或重采样，单个平均动作对非线性动态的表达需要单独评估。

采集器可以自主读取位置、图像、速度和碰撞。准确的命令动作需要在控制端下发时发布，实际速度与命令设定值都应保留。

电机、姿态、位置、路径和速度接口对应不同动作定义。扩展到其他类型时，应同步更新 schema、执行层、动作元数据、归一化、测量公式与校验器，并在数据中保留动作类型。

### 17.6 替换后的验证步骤

先在本目录使用现有入口做短回合检查

~~~powershell
$airsimPython = 'C:\Users\31861\.conda\envs\ue4-airsim\python.exe'
& $airsimPython '.\scripts\collect_world_model.py' --launch --steps 10 --seed 123
~~~

再对终端打印的实际 run 目录执行校验

~~~powershell
& $airsimPython '.\scripts\validate_dataset.py' '替换成实际run目录'
~~~

结合数据确认以下内容

- 实际飞行方向、速度和转向与 action.api 一致。
- applied、归一化值和上一动作记录符合第 5 节约定。
- command_queued_at_sim_ns 与起点观测时间一致。
- state_after_advance_ns 与终点观测时间一致。
- 10 个正常采集区间对应 11 个观测，提前成功或碰撞时按实际区间计数。
- 每组观测包含四个图像通道，控制与记录日志均正常退出。
- 完成、碰撞和上限事件保留各自语义，采集文件继续保存 reward 的重建依据。

当前校验器会校验既定的四维速度空间和变化率。更改动作定义时，应同步调整校验规则，并为新动作类型添加对应验证。

## 18. 采集参数简要讲解

本节整理采集参数讲解中的核心内容，便于第一次阅读数据时理解各字段的含义。完整字段字典见第 8、9 节，固定任务和配置见第 6、14 节。

### 18.1 一条训练样本怎样组成

数据描述无人机从当前观测执行动作后，到达下一观测的过程

$$
(o_t,\ a_t,\ \Delta t_t,\ o_{t+1},\ m_t,\ e_t)
$$

| 符号 | 含义 | 数据位置 |
| --- | --- | --- |
| $o_t$、$o_{t+1}$ | 前后时刻的图像、状态和目标关系 | observations.jsonl 与图像资产 |
| $a_t$ | 本区间实际下发的控制设定值 | transitions.jsonl 的 action |
| $\Delta t_t$ | 实际经过的仿真时间 | timing.actual_step_sim_s |
| $m_t$ | 位移、目标进展、低速持续时间等测量量 | measurements |
| $e_t$ | 碰撞、成功、达到上限等事件 | events |

执行 N 个动作区间，对应 N 条转移和 N+1 组观测

~~~text
观测 0 ──动作 0──→ 观测 1 ──动作 1──→ 观测 2
~~~

observations.jsonl 每行保存一组观测，transitions.jsonl 每行保存一个动作区间。后者通过 observation_id 和 next_observation_id 引用前后观测。

### 18.2 每个采样点记录什么

| 参数组 | 代表字段 | 通俗含义与用途 |
| --- | --- | --- |
| 采样身份 | observation_id、elapsed_sim_s | 当前是第几个采样点，距离采集起点经过多久 |
| 图像 | front_rgb、down_rgb | 无人机前方与下方看到的场景，可送入视觉编码器 |
| 深度 | front_depth、down_depth | 每个像素沿相机光轴方向的米制深度，可提供几何监督 |
| 相机几何 | camera_world_pose、intrinsics_K_estimated | 拍摄位置、朝向与估计内参，用于理解视角及重投影 |
| 位置与姿态 | position_ned_m、orientation_xyzw、yaw_rad | 无人机在哪里、朝向哪里、当前完整姿态 |
| 运动状态 | linear_velocity_ned_m_s、angular_velocity_api_rad_s，以及线/角加速度 | 当前实际怎样运动，可作为动态预测的辅助监督 |
| 目标关系 | offset_ned_m、offset_heading_frd_m、distance_3d_m、height_error_up_m | 目标在哪个方向、还有多远、需要升降多少 |
| 方位与接近速度 | bearing_error_rad、bearing_error_sin/cos、closing_speed_m_s | 目标在机头哪一侧，当前运动是否接近目标 |
| 到达状态 | inside_goal、speed_below_delivery_threshold、dwell_sim_s | 是否进入目标区域、是否足够慢、已连续停留多久 |
| 碰撞原始信息 | has_collided、timestamp_ns、object_name、impact_point_ned_m | 是否有接触记录、碰到什么、何时何地接触 |
| 环境 | gravity_ned_m_s2、air_pressure_pa、temperature_k、air_density_kg_m3、geo_point | 重力、气压、温度、密度与地理参考信息 |
| 时间诊断 | state_before_images_ns、state_after_images_ns、image_minus_state_s | 检查图像与运动状态的时间对应关系 |

角速度和角加速度以 API 原生分量保存，联合几何计算时应结合服务端坐标约定。四元数采用 xyzw 顺序，偏航角采用弧度。

深度使用 float32 NPY 保存，valid_mask_path 指向有效掩码。掩码为 1 表示数值有限且大于零，超远区域仍需根据训练距离范围筛选。summary 中的 min_m、median_m、p10_m 和 center_p10_m 提供整图或中央区域的深度摘要。

### 18.3 世界坐标与目标差怎么读

世界 NED 的 X、Y、Z 正方向分别为北、东、下，位置单位为米。位置 [10,5,-30] 表示相对原点北侧 10 米、东侧 5 米、上方 30 米，地形净空还需结合深度或地图计算。

目标偏移按目标位置减当前位置得到

$$
\delta_t=g-p_t
$$

例如 offset_ned_m 为 [30,10,-5]，表示目标在北侧 30 米、东侧 10 米、上方 5 米，height_error_up_m 对应 +5 米。

offset_heading_frd_m 把偏移转到当前水平机头的前、右、下方向。offset_body_frd_m 进一步考虑俯仰和横滚。

distance_3d_m 表示三维直线距离，distance_horizontal_m 表示水平距离。bearing_error 的正弦和余弦便于模型处理角度跨越边界，bearing_valid 标明水平目标方向是否有效。

### 18.4 动作参数怎么读

当前动作顺序为

$$
a_t=[v_{\mathrm{forward}},v_{\mathrm{right}},v_{\mathrm{down}},\dot\psi]
$$

前三项单位为 m/s，第四项为 deg/s。例如 [2,-0.5,-0.3,10] 表示前进 2 m/s、向左 0.5 m/s、上升 0.3 m/s，同时正向偏航 10°/s。

| 字段 | 含义 |
| --- | --- |
| requested | 策略产生的期望动作，包含该策略使用的扰动 |
| bounded | 分量限制和高度恢复后的动作 |
| applied | 进一步约束变化率后，实际下发的设定值 |
| normalized_applied | applied 分别除以 [4,3,2,30]，便于模型输入 |
| previous_applied | 上一个区间实际下发的动作 |
| slew_limits_per_s | 允许的设定值变化率 |
| altitude_guards | 本步采用的高度恢复标记 |
| exploration_noise | 策略内部相关噪声状态，实际使用幅度由策略配置决定 |
| api | 实际调用方法、世界速度、偏航模式、任务时长和车辆名 |

M 的动作输入可以选 applied 或 normalized_applied。state 中的速度描述实际运动结果，两者共同反映飞控闭环响应。

默认 API 任务时长为 dt+1.0，用于覆盖推进窗口并保留余量。当前采样窗口由 actual_step_sim_s 定义，旧任务在后续步骤被替换或在回合结束时取消。

### 18.5 reward 依据与结束事件怎么读

采集时保存测量量和事件，后续由训练程序构造 reward

$$
r_t=f_\theta(m_t,e_t,g)
$$

| 数据 | 代表字段 | 后续可表达的目标 |
| --- | --- | --- |
| 时间 | time_alive_increment_sim_s | 生存或任务时间项 |
| 接近目标 | goal_progress_3d_m、goal_progress_horizontal_m | 鼓励有效进展 |
| 高度改善 | absolute_height_error_reduction_m | 鼓励接近目标高度 |
| 移动与路径 | displacement_ned_m、displacement_3d_m、cumulative_sampled_path_length_m | 分析移动、绕行和路径效率 |
| 速度 | average_translation_speed_m_s、speed_before_m_s、speed_after_m_s | 速度约束与运动分析 |
| 低速持续 | stationary_duration_sim_s | 结合目标区域判断长期停滞 |
| 合法投递停留 | inside_goal_before/after、goal_dwell_before/after_sim_s | 识别任务要求的低速停驻 |
| 动作变化 | action_delta、action_delta_l2_normalized | 平滑控制项 |
| 控制强度代理 | normalized_command_squared_integral | 约束归一化命令强度 |
| 任务事件 | events.collision、events.delivery_success | 碰撞与成功项 |

目标进展为起点距离减终点距离，正值表示接近。normalized_command_squared_integral 等于归一化动作平方和乘以实际时长，真实电耗需要额外的功率依据。

默认目标停留要求为目标球内、实际速度不超过 0.5 m/s、连续约 2 秒。低速持续时间另用 stationary_speed 默认 0.3 m/s 计算。构造停滞惩罚时可以结合 inside_goal，保留完成配送所需的停留行为。

terminated 表示碰撞或成功造成的任务终止，truncated 表示时间或步数上限截断，done 表示当前回合结束。RPC 或取图异常通过 incomplete 状态表达。原始 has_collided 可能保留历史接触，events.collision 使用前后标记和碰撞时间戳识别新事件。

### 18.6 时间、配置与训练使用

默认名义间隔为 0.2 秒，约 5 Hz。实际间隔由前后状态时间戳计算，推进、计算和写盘的现实耗时分别记录。当前入口采用暂停步进，采样期间冻结仿真，恢复后继续演化已有运动状态。持续运行采集的接入设计见第 17 节。

quality 中的 step_error_exceeds_25pct 标记区间偏差，camera_state_skew_exceeds_20ms 标记前后图像与状态偏差。valid_for_dynamics 当前由相机偏差检查决定，训练还可结合其他字段评估样本质量。

| 文件 | 还需要关注的信息 |
| --- | --- |
| task.json | 固定目标、半径、停留要求、相机标定和初始状态 |
| episode.json | dt、控制策略、速度和高度范围、目标采样参数、随机种子与回合状态 |
| result.json | 成功、碰撞、结束原因、总步数、仿真时长与最终目标距离 |
| run.json | 设置、Python 与依赖版本、代码哈希、场景来源和全部回合结果 |

run.json 中天气和风速为 null 时表示入口没有取得对应设置，环境接口的逐帧测量值另行保存。

用于 M 时，可以把当前视觉编码、实际下发动作、实测时间和目标条件组成输入，预测下一视觉编码及运动状态

$$
x_t=[z_t,a_t,\Delta t_t,v_t,\delta_t^{heading}]
$$

$$
\text{预测目标}=[z_{t+1},v_{t+1},\text{碰撞事件}]
$$

当前目标距离和方向可作为条件输入。区间进展使用终点信息，适合作为监督标签或 reward 依据。循环模型在回合边界重置隐藏状态，后续价值学习按算法区分 terminated 与 truncated。



