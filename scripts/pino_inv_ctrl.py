#!/usr/bin/env python

"""_summary_

unit test : pinocchio inverse kinematic test
"""
import numpy as np
from numpy.linalg import norm, solve
import sys

import pinocchio
from copy import deepcopy
from pathlib import Path
import rospy 
import actionlib
from std_msgs.msg import Header
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from control_msgs.msg import JointTrajectoryControllerState  # 引入 JointControllerState 消息类型
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


class pinocchio_kinematics(object):

    def __init__(self, urdf_path: str, end_link: str, is_log=True):
        """Initializes the kinematics solver with a robot model."""
        self.urdf_path = urdf_path
        self.end_link = end_link #末端link
        self.is_log=is_log
        self.model = pinocchio.buildModelFromUrdf(self.urdf_path)
        self.model_data = self.model.createData()
        self.end_joint_id = self.model.getJointId(end_link)-1 # end joint id
        self.eps = 1e-4 # convergence threshold
        self.DT = 1e-1 # update step
        self.IT_MAX = 5000 # max iteration 
        self.damp = 1e-12 #damp factor to avoid Jacobian pseudo-inverse

    def qpos_to_limits(self, q: np.ndarray, upperPositionLimit: np.ndarray,
                       lowerPositionLimit: np.ndarray, joint_seed: np.ndarray,
                       ik_weight: np.ndarray):
        """Adjusts the joint positions (q) to be within specified limits and as 
        close as possible to the joint seed,  
        while minimizing the total weighted difference.  
    
        Args:  
            q (np.ndarray): The original joint positions.  
            upperPositionLimit (np.ndarray): The upper limits for the joint positions.  
            lowerPositionLimit (np.ndarray): The lower limits for the joint positions.  
            joint_seed (np.ndarray): The desired (seed) joint positions.  
            ik_weight (np.ndarray): The weights to apply for each joint in the total difference calculation.  
    
        Returns:  
            np.ndarray: The adjusted joint positions within the specified limits.  
        """
        qpos_limit = np.copy(q)
        best_qpos_limit = np.copy(q)
        best_total_q_diff = sys.float_info.max

        if ik_weight is None:
            ik_weight = np.ones_like(q)

        for i in range(len(q)):
            # Generate multiple candidates by adding or subtracting 2*pi multiples
            candidates = []
            for k in range(-5, 6 ):  # You can adjust the range of k to explore more possibilities
                candidate = q[i] + k * 2 * np.pi
                if lowerPositionLimit[i] <= candidate <= upperPositionLimit[i]:
                    candidates.append(candidate)

            # If no candidates are within the limits, just use the original value adjusted with 2*pi multiples
            if not candidates:
                candidate = (q[i] - joint_seed[i]) % (2 * np.pi) + joint_seed[i]
                while candidate < lowerPositionLimit[i]:
                    candidate += 2 * np.pi
                while candidate > upperPositionLimit[i]:
                    candidate -= 2 * np.pi
                candidates.append(candidate)

            # Find the candidate that gives the smallest total_q_diff
            best_candidate_diff = sys.float_info.max
            best_candidate = candidates[0]
            for candidate in candidates:
                qpos_limit[i] = candidate
                total_q_diff = np.sum(
                    np.abs(qpos_limit - joint_seed) * ik_weight)
                if total_q_diff < best_candidate_diff:
                    best_candidate_diff = total_q_diff
                    best_candidate = candidate

            qpos_limit[i] = best_candidate
            if best_candidate_diff < best_total_q_diff:
                best_total_q_diff = best_candidate_diff
                best_qpos_limit = np.copy(qpos_limit)

        return best_qpos_limit

    def getIk(self,
               target_pos: np.ndarray,
               target_orientation: np.ndarray,
               joint_seed: np.ndarray,
               ik_weight: np.ndarray = None):
        """Computes the inverse kinematics for a given target pose."""
        if not isinstance(target_pos,np.ndarray) \
                    or target_pos.shape != (3,):
            raise ValueError("target_pose must be a 1x3 numpy array")
        if not isinstance(target_orientation,np.ndarray) \
            or target_orientation.shape !=(4,):
            raise  ValueError("target_orientation must be a 1x4 numpy array")
                
        if not isinstance(joint_seed, np.ndarray):
            raise ValueError("joint_seed must be of type np.ndarray")
        target_pose_SE3 = pinocchio.SE3(pinocchio.Quaternion(target_orientation), target_pos)
        # q = deepcopy(joint_seed).astype(np.float64)
        q = np.copy(joint_seed).astype(np.float64)
        
        for i in range(self.IT_MAX):
            pinocchio.forwardKinematics(self.model,self.model_data,q)
            end_pose = self.model_data.oMi[self.end_joint_id]
            error_pose= target_pose_SE3.actInv(end_pose) # T_sd*T_sb^(-1)
            err = pinocchio.log(error_pose).vector # transform rotation to rotate vector by Rodrigues’ rotation formula

            if norm(err) < self.eps:
                
                q = self.qpos_to_limits(q,self.model.upperPositionLimit,self.model.lowerPositionLimit,
                                            joint_seed, ik_weight)
                if self.is_log:
                    print("Convergence iteration ")  
                    print("Pin:{} error = {}!".format(i, err.T))
                    self.getFk(q)
                return True,q
            J = pinocchio.computeJointJacobian(self.model, self.model_data, q,self.end_joint_id)
            v_e = -solve(J.dot(J.T)+self.damp*np.eye(6),err) #Levenberg-Marquardt（LM）
            v_j = J.T.dot(v_e) # map to joint space
            q = pinocchio.integrate(self.model, q, v_j*self.DT)
            
            if not i % 10 and self.is_log:
                print("Pin:{} error = {}!".format(i, err.T))

        print(
            "Pin:The iterative algorithm has not reached convergence to the desired precision"
        )
        return False, q
    def getFk(self,q:np.ndarray):
        pinocchio.forwardKinematics(self.model, self.model_data, q)
        
        print(f"{self.model_data.oMi[self.end_joint_id]=}")
    def get_joints(self):
        joints_name=[]
        for joint_id in range(self.model.njoints):
            joint = self.model.joints[joint_id]
            joint_type = self.model.joints[joint_id].shortname()
            # 检查关节类型，判断是否为可动关节
            if joint.nv > 0:
                joints_name.append(self.model.names[joint_id]) 
        print("可动关节列表{joint_name}")
        return joints_name

class ArmController:
    def __init__(self):
        rospy.init_node("space_mouse_controller")
        cur_dir = Path(__file__).parent.resolve()
        default_path = cur_dir / "rm_75.urdf"
        print(f"{cur_dir=}")
        self.is_sim = rospy.get_param("is_sim", default="True")
        self.urdf_path = rospy.get_param("urdf_path", default=str(default_path))
        self.ee_link = rospy.get_param("ee_link", default="Link7")
        rospy.Subscriber('/arm/arm_joint_controller/state', JointTrajectoryControllerState, self.stateCallback)
        # 创建一个action client连接到机械臂的动作接口
        self.client = actionlib.SimpleActionClient('/arm/arm_joint_controller/follow_joint_trajectory', FollowJointTrajectoryAction)
        self.client.wait_for_server() # rospy.Duration(2)
        print("get server")
        self.get_state_arm=False
        self.joint_actual_state=JointTrajectoryPoint()
        self.control_freq = 200 
        self.kin_solver = pinocchio_kinematics(str(self.urdf_path),self.ee_link) # note str
        self.joints_name = self.get_joint_names_from_urdf(self.urdf_path)
        self.target_position = np.array([0.2, 0.2, 0.4])  # 根据需求设置目标位置
        self.target_orientation = np.array([0.0, 0.0, 0.0, 1.0])  # 目标方向，以四元数表示
        self.rate = rospy.Rate(self.control_freq)
    

    def get_joint_names_from_urdf(self,urdf_path):
    # 导入URDF模型
        from urdf_parser_py.urdf import URDF
        urdf = URDF.from_xml_file(urdf_path)
        joint_names = [joint.name for joint in urdf.joints]
        return joint_names    
    def run(self):
        while not rospy.is_shutdown():
            if self.get_state_arm:
                joint_seed = np.array(self.joint_actual_pos)
                res, q_out = self.kin_solver.getIk(self.target_position,self.target_orientation, joint_seed)
                print(f"{q_out=}")
                self.trajectory_points = [
                        {
                            "positions": q_out,  # 弧度
                            "time_from_start": 1.0/self.control_freq,  # 移动到该点所需的时间（秒）
                        },  # 添加更多轨迹点...
                    ]
                self.setJointPositions(self.trajectory_points)
            self.rate.sleep()
    def stateCallback(self, msg):
        self.joint_actual_state = msg.actual
        self.joint_actual_pos = [
            joint_pos for joint_pos in self.joint_actual_state.positions
        ]
        self.get_state_arm = True
        # rospy.loginfo("Received joint state: position=%s\n, velocity=%s\n, effort=%s\n",
        #             msg.desired.positions, msg.desired.velocities, msg.desired.effort)
        # rospy.loginfo("Received joint state: position=%s\n, velocity=%s\n, effort=%s\n",
        #         msg.actual.positions, msg.actual.velocities, msg.actual.effort)
        

    def setJointPositions(self, trajectory_points):  # req
        # 创建目标关节姿态
        goal = FollowJointTrajectoryGoal()
        goal.trajectory.joint_names = self.joints_name

        # 创建JointTrajectoryPoint，设置目标位置
        hearder = Header()
        hearder.stamp = rospy.Time.now()
        hearder.frame_id = "base_link"
        point = FollowJointTrajectoryGoal()
        for point in trajectory_points:
            trajectory_point = JointTrajectoryPoint()
            trajectory_point.positions = point["positions"]
            trajectory_point.velocities = point.get(
                "velocities", [0.0] * len(self.joints_name)
            )  # 可选，设置速度
            trajectory_point.accelerations = point.get(
                "accelerations", [0.0] * len(self.joints_name)
            )  # 可选，设置加速度
            trajectory_point.time_from_start = rospy.Duration(point["time_from_start"])
            goal.trajectory.points.append(trajectory_point)  # 将点添加到轨迹中
        # 发送目标到机器人
        self.client.send_goal(goal)
        self.client.wait_for_result(rospy.Duration(5.0))
        print(" server send goal")


if __name__ == "__main__":
    try:
        ArmController=ArmController()
        ArmController.run()
    
    except rospy.ROSInterruptException:
        print("ERROR")
        pass
        

