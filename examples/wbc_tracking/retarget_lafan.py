if __name__ == "__main__":
    import argparse
    from pathlib import Path

    from rlightning.humanoid.formatter import Formatter
    from rlightning.humanoid.loader.lafan_loader import LafanLoader, LoaderCfg
    from rlightning.humanoid.retarget import Retargeter
    from rlightning.utils.logger import get_logger

    logger = get_logger(__name__)

    parser = argparse.ArgumentParser("Test lafan loader")
    parser.add_argument("--f-path", help="Motion file/directory path.", required=True)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite or not")

    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    robot_xml_path = str(base_dir / "robots" / "g1_mocap_29dof.xml")
    cfg = LoaderCfg(
        data_format=".bvh",
        data_path=args.f_path,
        overwrite=args.overwrite,
        retargeter=Retargeter.RetargeterCfg(
            robot_xml_path=robot_xml_path,
            solver="daqp",
            damping=0.5,
            use_velocity_limit=False,
            ik_config_path=str(base_dir / "conf" / "ik" / "lafan_to_unitree_g1.yaml"),
        ),
        formatter=Formatter.FormatterCfg(
            robot_xml_path=robot_xml_path,
            height_adjust=False,
            root_offset=False,
            quat_order="xyzw",
            kinematic_model_device="cpu",
        ),
    )
    loader = LafanLoader(cfg)
    loader.prepare(args.f_path)
    motion = loader.sample()[0]

    logger.info(f"FPS: {motion.fps}")
    logger.info(f"root_pos shape: {motion.root_pos.shape}")
    logger.info(f"root_rot shape: {motion.root_rot.shape}")
    logger.info(f"dof_pos: {motion.dof_pos.shape}")
