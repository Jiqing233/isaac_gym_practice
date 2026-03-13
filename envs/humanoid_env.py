import os
import numpy as np
from isaacgym import gymapi, gymtorch, gymutil
import torch

class HumanoidEnv:

    def __init__(self, num_envs=1024, device="cuda"):

        self.num_envs = num_envs
        self.device = device

        self._create_sim()
        self._create_envs()
        self._prepare_tensors()
        self._load_motion()

        self.obs_dim = self.root_states.shape[1] + self.dof_states.view(self.num_envs, -1).shape[1] + 1 # Added a phase
        self.act_dim = self.num_dofs

    def _create_sim(self):
        self.gym = gymapi.acquire_gym()
        gymutil.parse_arguments()

        sim_params = gymapi.SimParams()
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0, 0, -9.81)

        sim_params.physx.use_gpu = True
        sim_params.use_gpu_pipeline = True

        self.sim = self.gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)

        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0, 0, 1)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self):
        project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        asset_root = os.path.join(project_dir, "assets")
        asset_file = "humanoid.xml"
        print("Loading from:", os.path.join(asset_root, asset_file))

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = False
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_EFFORT
        asset_options.replace_cylinder_with_capsule = True

        humanoid_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

        if humanoid_asset is None:
            raise RuntimeError("Failed to load humanoid asset. Check that Assets/humanoid.xml exists.")

        print("DOFs:", self.gym.get_asset_dof_count(humanoid_asset))
        print("Bodies:", self.gym.get_asset_rigid_body_count(humanoid_asset))

        self.num_dofs = self.gym.get_asset_dof_count(humanoid_asset)

        spacing = 2.0
        env_lower = gymapi.Vec3(-spacing, -spacing, 0)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)

        self.envs = []
        self.actors = []

        for i in range(self.num_envs):
            env = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))

            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(0, 0, 1.0)

            actor = self.gym.create_actor(env, humanoid_asset, pose, "humanoid", i, 1)

            self.envs.append(env)
            self.actors.append(actor)

        self.gym.prepare_sim(self.sim)

    def _prepare_tensors(self):

        rb_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        root_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_tensor = self.gym.acquire_dof_state_tensor(self.sim)

        self.rb_states = gymtorch.wrap_tensor(rb_tensor)
        self.root_states = gymtorch.wrap_tensor(root_tensor)
        self.dof_states = gymtorch.wrap_tensor(dof_tensor)

    def _load_motion(self):

        project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        motion_path = os.path.join(project_dir, "data","martial_arts", "amp_humanoid_walk.npy")

        motion_data = np.load(motion_path, allow_pickle=True)

        # AMP stores everything inside OrderedDict
        motion_dict = motion_data.item()

        root_pos = motion_dict["root_translation"]["arr"]
        rotation = motion_dict["rotation"]["arr"]
        lin_vel = motion_dict["global_velocity"]["arr"]
        ang_vel = motion_dict["global_angular_velocity"]["arr"]

        # Convert to torch
        self.root_pos_ref = torch.tensor(root_pos, dtype=torch.float32, device=self.device)
        self.rot_ref = torch.tensor(rotation, dtype=torch.float32, device=self.device)
        self.lin_vel_ref = torch.tensor(lin_vel, dtype=torch.float32, device=self.device)
        self.ang_vel_ref = torch.tensor(ang_vel, dtype=torch.float32, device=self.device)

        self.motion_length = self.root_pos_ref.shape[0]

        # Random phase per env
        self.motion_phase = torch.randint(
            0, self.motion_length, (self.num_envs,), device=self.device
        )

        print("Motion frames:", self.motion_length)

    def step(self, actions):

        self.gym.set_dof_actuation_force_tensor(
            self.sim,
            gymtorch.unwrap_tensor(actions)
        )

        # Update motion phase
        self.motion_phase += 1
        self.motion_phase %= self.motion_length

        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)

        obs = self.compute_observations()
        reward = self.compute_reward()
        done = self.compute_done()

        return obs, reward, done

    def compute_observations(self):

        dof_view = self.dof_states.view(self.num_envs, -1)

        phase = self.motion_phase.float() / self.motion_length
        phase = phase.unsqueeze(-1)

        obs = torch.cat([
            self.root_states,
            dof_view,
            phase
        ], dim=-1)

        return obs

    def compute_reward(self):
        rb = self.rb_states.view(self.num_envs, 15, 13)
        body_rot = rb[:, :, 3:7]   # quaternion
        ref_rot = self.rot_ref[self.motion_phase]
        rot_error = torch.mean((body_rot - ref_rot) ** 2, dim=(1, 2))
        reward = torch.exp(-5.0 * rot_error)
        return reward

    def compute_done(self):
        return self.root_states[:, 2] < 0.5