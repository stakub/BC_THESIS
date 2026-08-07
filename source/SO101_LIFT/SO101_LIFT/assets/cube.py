import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.sim.schemas import MassPropertiesCfg

CUBE_COLOR = (158 / 255, 58 / 255, 56 / 255)
CUBE_MASS = 15 / 1000  # The weight in kilos
CUBE_CFG = RigidObjectCfg(spawn=sim_utils.CuboidCfg(size=(0.03, 0.03, 0.03), mass_props=MassPropertiesCfg(mass=CUBE_MASS),
                                                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                                                    collision_props=sim_utils.CollisionPropertiesCfg(),
                                                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=CUBE_COLOR,
                                                    metallic=0.0)), init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.3, 0.0, 1.0)))
