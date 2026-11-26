"""A local flat scene without downloads of Nucleus textures or ground assets."""

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass


@configclass
class Gl165SceneCfg(InteractiveSceneCfg):
    def __init__(self, config, physics_dt, step_dt):
        super().__init__(num_envs=config.num_envs, env_spacing=config.env_spacing)
        self.ground = AssetBaseCfg(
            prim_path="/World/ground",
            collision_group=-1,
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.05)),
            spawn=sim_utils.CuboidCfg(
                size=(1000.0, 1000.0, 0.10),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    friction_combine_mode="multiply", restitution_combine_mode="multiply",
                    static_friction=1.0, dynamic_friction=1.0,
                ),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.22, 0.25, 0.28)),
            ),
        )
        self.robot = config.robot.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.contact_sensor = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True, update_period=physics_dt,
        )
        self.light = AssetBaseCfg(
            prim_path="/World/light", spawn=sim_utils.DomeLightCfg(intensity=1800.0, color=(0.85, 0.85, 0.85)),
        )
