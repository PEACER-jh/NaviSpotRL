from NaviSpotRL.robots.jetbot import JETBOT_CONFIG
from NaviSpotRL.robots.rover import PANORAMIC_ROVER_CFG

from isaaclab.sim import SimulationCfg
import isaaclab.sim as sim_utils
from isaaclab.utils import configclass
from isaaclab.sensors import CameraCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.scene import InteractiveSceneCfg

@configclass
class NavispotrlEnvCfg(DirectRLEnvCfg):
    decimation = 2          # 每隔多少 sim steps 采集一次数据（即环境步频）
    episode_length_s = 20.0 # 单环境生命周期（秒）
    action_space = 2        # 动作空间维度
    observation_space = 5   # policy: dot, cross, distance, fwd_vel, ang_vel (5D)
    # lidar_rays: (B, 2, n_rays//2) — separate obs group
    # waypoints: [dx1, dy1, dx2, dy2] (4D) — separate obs group
    state_space = 0         # critic 只使用 policy 相关的观测

    # ========= 模型参数 ==========
    # obs_groups defines which observation keys go to actor / critic
    # "policy"     = 1D  (5,)   → dot, cross, distance, fwd_vel, ang_vel
    # "waypoints"  = 1D  (4,)   → next 2 waypoints in robot frame
    obs_groups: dict[str, list[str]] = {
        "actor": ["policy", "waypoints", "lidar_rays"],
        "critic": ["policy", "waypoints", "lidar_rays"],
    }

    # ========= 日志参数 ==========
    tb_log_dir: str = "/tmp/navispotrl_logs"
    reward_scale: float = 0.3             # 提升正信号量, 障碍物惩罚上限已降
    is_camera_log: bool = False           # 是否启用摄像头日志（Rerun）
    is_ray_visualization: bool = False    # 是否可视化lidar射线（Isaac Sim）
    is_path_visualization: bool = False   # 是否可视化全局规划路径（Isaac Sim）

    # ========= 任务参数 ==========
    # dof_names = ["left_wheel_joint", "right_wheel_joint"]
    dof_names = ["left_wheel_joint", "left_wheel_back_joint", 
                 "right_wheel_joint", "right_wheel_back_joint"]
    target_distance_range = (8.0, 14.0) # 目标距离范围 [m]，在每个 episode 开始时随机采样
    target_reach_threshold = 0.3        # 距离小于 0.3m 即视为到达目标    

    # ========= 底盘运动学参数 ==========
    wheel_radius = 0.04         # 轮子半径 [m]
    half_track = 0.12           # 半轮距 [m] (左右轮 y = ±0.12)
    wheelbase = 0.16            # 轴距 [m] (前后轮 x = ±0.08)
    max_linear_vel = 2.0        # 线速度上限 [m/s]
    max_angular_vel = 4.0       # 角速度上限 [rad/s]

    # ========== 围墙参数 ==========
    arena_side_length = 10.0    # 围墙边长 [m]
    wall_height = 0.5           # 围墙高度 [m]
    wall_thickness = 0.1        # 围墙厚度 [m]
    wall_safe_margin = 0.3      # 与围墙的安全间距 [m]

    # ========== 障碍物参数 ==========
    num_obstacles = 0#20          # 每个围墙内的障碍物数量
    obstacle_safe_margin = 0.3  # 与障碍物的安全距离 [m]
    randomize_obstacles: bool = False  # True=每episode随机障碍物, False=固定布局重复训练
    
    box_min_size = 0.5          # 长方体最小边长 [m]
    box_max_size = 1.5          # 长方体最大边长 [m]
    box_min_height = 0.3        # 长方体最小高度 [m]
    box_max_height = 0.5        # 长方体最大高度 [m]

    # ========= 栅格地图参数 ==========
    grid_resolution = 0.05      # 占据栅格地图分辨率 [m/px]
    n_rays = 64                 # 深度图提取的射线数（前后各半）

    # ========= 规划器参数 ==========
    planner_inflate_margin = 0.2        # 障碍物膨胀距离 [m]（越大越保守）
    planner_contour_min_pts = 4         # 轮廓最小顶点数
    planner_max_edge_len = 5.0          # 拓扑图最大边长 [m]（过滤远距离连线）
    planner_use_astar: bool = False     # True=A*, False=Dijkstra
    
    # ========= 动态投影 & 重规划参数 ==========
    replan_threshold: float = 4.0          # cross-track 误差阈值 [m]（大幅放宽）
    replan_cooldown_steps: int = 200       # 同一机器人两次重规划的最少步数间隔
    waypoint_look_ahead_dist: float = 0.0  # 子目标 = 投影点所属waypoint（零前探距离）


    sim: SimulationCfg = SimulationCfg(
        dt=1/120, 
        render_interval=decimation,
        use_fabric=False,
    )

    # robot_cfg: ArticulationCfg = JETBOT_CONFIG.replace(
    #     prim_path="/World/envs/env_.*/Robot"
    # )
    robot_cfg: ArticulationCfg = PANORAMIC_ROVER_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        collision_group=-1,
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=32,                # 环境
        env_spacing=0.0,            # 所有机器人在同一原点
        replicate_physics=True,     # 物理引擎不复制
        filter_collisions=True,     # 禁止机器人之间碰撞
    )