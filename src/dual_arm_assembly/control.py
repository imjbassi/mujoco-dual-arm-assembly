"""Damped least-squares pose IK on scratch data, never on the live physics state."""

import mujoco
import numpy as np


class PoseIK:
    def __init__(self, model: mujoco.MjModel, arm: str, site: str):
        self.model = model
        self.data = mujoco.MjData(model)
        self.site = model.site(site).id
        joints = [model.joint(f"{arm}_j{i}").id for i in range(7)]
        self.qids = model.jnt_qposadr[joints]
        self.dofs = model.jnt_dofadr[joints]
        self.limits = model.jnt_range[joints]
        self.jp = np.zeros((3, model.nv))
        self.jr = np.zeros((3, model.nv))

    def solve(self, qpos, target, iterations=12):
        self.data.qpos[:] = qpos
        for _ in range(iterations):
            mujoco.mj_kinematics(self.model, self.data)
            mujoco.mj_comPos(self.model, self.data)
            rotation = self.data.site_xmat[self.site].reshape(3, 3)
            # All task frames have identity orientation; use quaternion log error.
            quat = np.empty(4)
            mujoco.mju_mat2Quat(quat, rotation.ravel())
            inverse = quat.copy()
            inverse[1:] *= -1
            if inverse[0] < 0:
                inverse *= -1
            orientation_error = np.empty(3)
            mujoco.mju_quat2Vel(orientation_error, inverse, 1)
            error = np.r_[target - self.data.site_xpos[self.site], orientation_error * 0.35]
            if np.linalg.norm(error) < 1e-5:
                break
            mujoco.mj_jacSite(self.model, self.data, self.jp, self.jr, self.site)
            jac = np.vstack((self.jp[:, self.dofs], self.jr[:, self.dofs] * 0.35))
            delta = jac.T @ np.linalg.solve(jac @ jac.T + 0.0001 * np.eye(6), error)
            self.data.qpos[self.qids] = np.clip(
                self.data.qpos[self.qids] + np.clip(delta, -0.15, 0.15),
                self.limits[:, 0] + 0.01,
                self.limits[:, 1] - 0.01,
            )
        return self.data.qpos[self.qids].copy()


def smoothstep(start, end, fraction):
    t = np.clip(fraction, 0, 1)
    blend = t * t * t * (10 + t * (-15 + 6 * t))
    return np.asarray(start) + blend * (np.asarray(end) - start)
