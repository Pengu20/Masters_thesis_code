from pathlib import Path

import glfw
import mujoco as mj
from dm_control import mjcf

from sims import BaseSim
from sims.base_sim import SimSync


class MjSim(BaseSim):
    def __init__(self):
        super().__init__()

        self._model, self._data = self.init()

        self.tasks = [self.spin]

    def init(self):
        # root
        _HERE = Path(__file__).parent.parent
        # scene path


        m = mj.MjModel.from_xml_path("learning/scenes/cloth_composite.xml")
        d = mj.MjData(m)
        

        mj.mj_saveLastXML("cloth_sim_defined.xml", m)


        # mjcf.export_with_assets(mjcf_model=scene, out_dir="/home/peter/OneDrive/Skrivebord/Uni stuff/Masters/masters_code/mj_sim/learning/scenes")

        # step once to compute the poses of objects
        mj.mj_step(m, d)

        return m, d

    def spin(self, ss: SimSync):
        while True:
            ss.step()

    @property
    def data(self) -> mj.MjData:
        return self._data

    @property
    def model(self) -> mj.MjModel:
        return self._model

    def keyboard_callback(self, key: int):
        if key is glfw.KEY_SPACE:
            print("You pressed space...")


if __name__ == "__main__":
    sim = MjSim()
    sim.run()
