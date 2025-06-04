import argparse
import logging
import sys

from sim import MjSim
from utils.helpers import Bool

_logger = logging.getLogger("mj_sim")


def setup_logging(loglevel):
    """Setup basic logging

    Args:
      loglevel (int): minimum loglevel for emitting messages
    """
    logformat = "[%(asctime)s] %(levelname)s: %(message)s"
    logging.basicConfig(
        level=loglevel, stream=sys.stdout, format=logformat, datefmt="%Y-%m-%d %H:%M:%S"
    )


def main():
    parser = argparse.ArgumentParser(
        description="MuJoCo simulation of manipulators and controllers."
    )

    parser.add_argument(
        "--scene_path",
        type=str,
        default="scenes/scene.xml",
        help="Path to the XML file defining the simulation scene.",
    )
    parser.add_argument(
        "--show_site_frames",
        type=Bool,
        default=True,
        help="Boolean flag to display the site frames in the simulation.",
    )
    parser.add_argument(
        "--gravity_comp",
        type=Bool,
        default=True,
        help="Boolean flag to enable gravity compensation.",
    )
    parser.add_argument(
        "--manual_ctrl",
        type=Bool,
        default=False,
        help="Boolean flag to enable manual control in the simulation.",
    )
    parser.add_argument(
        "--render_size",
        type=float,
        default=0.1,
        help="Size of the rendered site axes in the scene.",
    )
    parser.add_argument(
        "--state0", type=str, default="home", help="Initial state of the simulation."
    )
    parser.add_argument(
        "--cam_width",
        type=int,
        default=640,
        help="Width of the camera view in the simulation.",
    )
    parser.add_argument(
        "--cam_height",
        type=int,
        default=480,
        help="Height of the camera view in the simulation.",
    )
    parser.add_argument(
        "--headless",
        type=Bool,
        default=False,
        help="Run the simulation in headless mode. If set to True, the simulation will run without rendering a visual display. This is useful for running simulations on servers or environments without a graphical interface.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        dest="loglevel",
        help="set loglevel to INFO",
        action="store_const",
        const=logging.INFO,
    )

    args, _ = parser.parse_known_args()

    # set the default log level to INFO
    _logger.setLevel(logging.INFO)
    setup_logging(args.loglevel)

    _logger.info(" > Loaded configuration:")
    for key, value in vars(args).items():
        _logger.info(f"\t{key:30}{value}")

    sim = MjSim(args)
    sim.run()


if __name__ == "__main__":
    main()
