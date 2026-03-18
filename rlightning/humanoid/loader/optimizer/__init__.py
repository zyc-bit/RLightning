from rlightning.utils.config import Config

from .fit_actorcore_shape import ActorCoreShapeFitting
from .fit_bvh_shape import BVHShapeFitting
from .fit_mixamo_shape import MixamoShapeFitting
from .fit_motion_million_shape import MotionMillionShapeFitting
from .fit_parc_shape import ParcShapeFitting
from .fit_smplx_shape import SPMLXShapeFitting


OPTIMIZER_LIB = dict(
    actor_core=ActorCoreShapeFitting,
    bvh=BVHShapeFitting,
    mixamo=MixamoShapeFitting,
    motion_million=MotionMillionShapeFitting,
    parc=ParcShapeFitting,
    smplx=SPMLXShapeFitting,
)


def get_optimizer(optimizer_type: str, optimizer_cfg: Config):
    return OPTIMIZER_LIB[optimizer_type](optimizer_cfg)
