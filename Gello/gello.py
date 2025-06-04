"""
from https://github.com/wuphilipp/gello_software/blob/main/gello/dynamixel/driver.py
"""

import subprocess
import time
from pathlib import Path
from typing import Sequence

from typing_extensions import Protocol

import numpy as np
from dynamixel_sdk.group_sync_read import GroupSyncRead
from dynamixel_sdk.group_sync_write import GroupSyncWrite
from dynamixel_sdk.packet_handler import PacketHandler
from dynamixel_sdk.port_handler import PortHandler
from dynamixel_sdk.robotis_def import (
    COMM_SUCCESS,
    DXL_HIBYTE,
    DXL_HIWORD,
    DXL_LOBYTE,
    DXL_LOWORD,
)

# Constants
ADDR_TORQUE_ENABLE = 64
ADDR_GOAL_POSITION = 116
LEN_GOAL_POSITION = 4
ADDR_PRESENT_POSITION = 132
LEN_PRESENT_POSITION = 4
TORQUE_ENABLE = 1
TORQUE_DISABLE = 0


class DynamixelDriverProtocol(Protocol):
    def set_joints(self, joint_angles: Sequence[float]):
        """Set the joint angles for the Dynamixel servos.

        Args:
            joint_angles (Sequence[float]): A list of joint angles.
        """
        ...

    def torque_enabled(self) -> bool:
        """Check if torque is enabled for the Dynamixel servos.

        Returns:
            bool: True if torque is enabled, False if it is disabled.
        """
        ...

    def set_torque_mode(self, enable: bool):
        """Set the torque mode for the Dynamixel servos.

        Args:
            enable (bool): True to enable torque, False to disable.
        """
        ...

    def get_joints(self) -> np.ndarray:
        """Get the current joint angles in radians.

        Returns:
            np.ndarray: An array of joint angles.
        """
        ...

    def close(self):
        """Close the driver."""


class FakeDynamixelDriver(DynamixelDriverProtocol):
    def __init__(self, ids: Sequence[int]):
        self._ids = ids
        self._joint_angles = np.zeros(len(ids), dtype=int)
        self._torque_enabled = False

    def set_joints(self, joint_angles: Sequence[float]):
        if len(joint_angles) != len(self._ids):
            raise ValueError(
                "The length of joint_angles must match the number of servos"
            )
        if not self._torque_enabled:
            raise RuntimeError("Torque must be enabled to set joint angles")
        self._joint_angles = np.array(joint_angles)

    def torque_enabled(self) -> bool:
        return self._torque_enabled

    def set_torque_mode(self, enable: bool):
        self._torque_enabled = enable

    def get_joints(self) -> np.ndarray:
        return self._joint_angles.copy()

    def close(self):
        pass


class DynamixelDriver(DynamixelDriverProtocol):
    def __init__(
        self,
        ids: Sequence[int],
        port: str = "/dev/ttyUSB0",
        baudrate: int = 2_000_000,
        port_latency_ms: int = 1,
    ):
        # set desired port latency
        port_name = Path(port).name
        port_latency_path = (
            Path("/sys/bus/usb-serial/devices/") / port_name / "latency_timer"
        )
        assert port_latency_path.exists(), port_latency_path
        with port_latency_path.open() as f:
            actual_port_latency = f.read().strip()
        if actual_port_latency != str(port_latency_ms):
            cmd = f"echo {port_latency_ms} > {str(port_latency_path)}"
            res = subprocess.call(f'sudo bash -c "{cmd}"', shell=True)
            assert res == 0

        """Initialize the DynamixelDriver class.

        Args:
            ids (Sequence[int]): A list of IDs for the Dynamixel servos.
            port (str): The USB port to connect to the arm.
            baudrate (int): The baudrate for communication.
        """
        self._ids = ids
        self._joint_angles = None

        # Initialize the port handler, packet handler, and group sync read/write
        self._portHandler = PortHandler(port)
        self._packetHandler = PacketHandler(2.0)
        self._groupSyncRead = GroupSyncRead(
            self._portHandler,
            self._packetHandler,
            ADDR_PRESENT_POSITION,
            LEN_PRESENT_POSITION,
        )
        self._groupSyncWrite = GroupSyncWrite(
            self._portHandler,
            self._packetHandler,
            ADDR_GOAL_POSITION,
            LEN_GOAL_POSITION,
        )

        # Open the port and set the baudrate
        if not self._portHandler.openPort():
            raise RuntimeError("Failed to open the port")

        if not self._portHandler.setBaudRate(baudrate):
            raise RuntimeError(f"Failed to change the baudrate, {baudrate}")

        # Add parameters for each Dynamixel servo to the group sync read
        for dxl_id in self._ids:
            if not self._groupSyncRead.addParam(dxl_id):
                raise RuntimeError(
                    f"Failed to add parameter for Dynamixel with ID {dxl_id}"
                )

        # Disable torque for each Dynamixel servo
        self._torque_enabled = False
        try:
            self.set_torque_mode(self._torque_enabled)
        except Exception as e:
            print(f"port: {port}, {e}")

        print("Connected successfully")



    def set_joints(self, joint_angles: Sequence[float]):
        if len(joint_angles) != len(self._ids):
            raise ValueError(
                "The length of joint_angles must match the number of servos"
            )
        if not self._torque_enabled:
            raise RuntimeError("Torque must be enabled to set joint angles")

        for dxl_id, angle in zip(self._ids, joint_angles):
            # Convert the angle to the appropriate value for the servo
            position_value = int(angle * 2048 / np.pi)

            # Allocate goal position value into byte array
            param_goal_position = [
                DXL_LOBYTE(DXL_LOWORD(position_value)),
                DXL_HIBYTE(DXL_LOWORD(position_value)),
                DXL_LOBYTE(DXL_HIWORD(position_value)),
                DXL_HIBYTE(DXL_HIWORD(position_value)),
            ]

            # Add goal position value to the Syncwrite parameter storage
            dxl_addparam_result = self._groupSyncWrite.addParam(
                dxl_id, param_goal_position
            )
            if not dxl_addparam_result:
                raise RuntimeError(
                    f"Failed to set joint angle for Dynamixel with ID {dxl_id}"
                )

        # Syncwrite goal position
        dxl_comm_result = self._groupSyncWrite.txPacket()
        if dxl_comm_result != COMM_SUCCESS:
            raise RuntimeError("Failed to syncwrite goal position")

        # Clear syncwrite parameter storage
        self._groupSyncWrite.clearParam()

    def torque_enabled(self) -> bool:
        return self._torque_enabled

    def set_torque_mode(self, enable: bool):
        torque_value = TORQUE_ENABLE if enable else TORQUE_DISABLE
        for dxl_id in self._ids:
            dxl_comm_result, dxl_error = self._packetHandler.write1ByteTxRx(
                self._portHandler, dxl_id, ADDR_TORQUE_ENABLE, torque_value
            )
            if dxl_comm_result != COMM_SUCCESS or dxl_error != 0:
                print(dxl_comm_result)
                print(dxl_error)
                raise RuntimeError(
                    f"Failed to set torque mode for Dynamixel with ID {dxl_id}"
                )
        self._torque_enabled = enable

    def get_joints(self) -> np.ndarray:
        dxl_comm_result = self._groupSyncRead.txRxPacket()
        if dxl_comm_result != COMM_SUCCESS:
            raise RuntimeError(f"comm failed: {dxl_comm_result}")

        joint_angles = np.zeros(len(self._ids), dtype=int)

        for i, dxl_id in enumerate(self._ids):
            if self._groupSyncRead.isAvailable(
                dxl_id, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION
            ):
                angle = self._groupSyncRead.getData(
                    dxl_id, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION
                )
                # TODO: what datatype is angle here?
                joint_angles[i] = np.int32(np.uint32(angle))
            else:
                raise RuntimeError(
                    f"Failed to get joint angles for Dynamixel with ID {dxl_id}"
                )
        return joint_angles * (2 * np.pi / 4096)  # convert to radians (4096 ticks/rev)

    def close(self):
        self._portHandler.closePort()



class GelloUR5:
    def __init__(self, usb_port: str):
        self.driver = DynamixelDriver(port=usb_port, ids=list(range(1, 8)), baudrate=57600)
        self.q_rest = np.array(
            [np.pi, -np.pi / 2, np.pi / 2, -np.pi / 2, -np.pi / 2, 0, 0]
        )
        self.q_gains = np.array([1, 1, -1, 1, 1, 1, -1.60])
        self.offset_path = "Gello/data/q_offset.npy"

        try:
            self.q_offset = np.load(self.offset_path)
        except:
            print("WARNING: offset calibration was not loaded - press ENTER to continue")
            input()
            self.q_offset = np.zeros(7)

            

    def get_q(self):
        q = self.driver.get_joints()
        q = q * self.q_gains + self.q_offset
        q[6] = np.clip(q[6], 0,1)
        return q

    def reset_offset(self):
        q = self.driver.get_joints()
        offset = self.q_rest - q * self.q_gains
        # offset_steps = np.round(q / (np.pi / 4))
        # offset = offset_steps * np.pi / 4
        self.q_offset = offset
        np.save(self.offset_path, offset)


def main():
    # Set the port, baudrate, and servo IDs
    ids = list(range(1, 8))

    # Create a DynamixelDriver instance
    driver = DynamixelDriver(ids, baudrate=57600)

    # Test setting torque mode
    driver.set_torque_mode(True)

    # Test reading the joint angles
    try:
        while True:
            start = time.monotonic()
            joint_angles = driver.get_joints()
            print(f"Joint angles for IDs {ids}: {joint_angles}")
            # print(f"Joint angles for IDs {ids[1]}: {joint_angles[1]}")
            duration = time.monotonic() - start
            print(f"{1/duration:.3f} hz")
            # time.sleep(max(0, 0.02 - duration))
    except KeyboardInterrupt:
        pass
    finally:
        driver.set_torque_mode(False)
        driver.close()


if __name__ == "__main__":
    main()  # Test the driver
