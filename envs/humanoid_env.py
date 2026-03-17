import os
import numpy as np
from isaacgym import gymapi, gymtorch, gymutil
import torch

class HumanoidEnv:

    def __init__(self, num_envs=1024, device="cuda", enable_viewer=False):

        self.num_envs = num_envs
        self.device = device

        self._create_sim()
        self._create_envs()
        self._prepare_tensors()
        self._load_motion()

        self.obs_dim = self.root_states.shape[1] + self.dof_states.view(self.num_envs, -1).shape[1] + 1 # Added a phase
        self.act_dim = self.num_dofs

        # Create a viewer
        self.enable_viewer = enable_viewer
        self.viewer = None

        if self.enable_viewer:
            self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
            if self.viewer is None:
                raise RuntimeError("Failed to create viewer")

            grid_size = int(self.num_envs ** 0.5)
            spacing = 2.0
            center = (grid_size - 1) * spacing / 2.0
            cam_pos = gymapi.Vec3(center + 15.0, center + 15.0, 10.0)
            cam_target = gymapi.Vec3(center, center, 1.0)

            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

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

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.initial_root_states = self.root_states.clone()
        self.initial_dof_states = self.dof_states.clone()

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

    
    def render(self):
        if self.viewer is None:
            return

        if self.gym.query_viewer_has_closed(self.viewer):
            raise SystemExit("Viewer closed")

        self.gym.step_graphics(self.sim)
        self.gym.draw_viewer(self.viewer, self.sim, True)
        self.gym.sync_frame_time(self.sim)
    
    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)

        if env_ids.numel() == 0:
            return self.compute_observations()

        # easier curriculum: always start from beginning of clip
        self.motion_phase[env_ids] = 0

        # restore saved initial simulator state
        self.root_states[env_ids] = self.initial_root_states[env_ids]
        self.dof_states.view(self.num_envs, self.num_dofs, 2)[env_ids] = \
            self.initial_dof_states.view(self.num_envs, self.num_dofs, 2)[env_ids]

        actor_indices = env_ids.to(dtype=torch.int32)

        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(actor_indices),
            actor_indices.numel()
        )

        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_states),
            gymtorch.unwrap_tensor(actor_indices),
            actor_indices.numel()
        )

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        return self.compute_observations()
        
    def step(self, actions):
        torque = 0.1 * actions
        self.gym.set_dof_actuation_force_tensor(
            self.sim, gymtorch.unwrap_tensor(torque)
        )

        # Update motion phase
        self.motion_phase += 1
        self.motion_phase %= self.motion_length

        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        obs = self.compute_observations()
        reward = self.compute_reward()
        done = self.compute_done()

        done_env_ids = torch.nonzero(done, as_tuple=False).squeeze(-1)
        num_done = done_env_ids.numel()
        if num_done > 0:
            root_h = self.root_states[:, 2]
            #print(
            #    "num_done:", num_done,
            #    "root_h_mean:", root_h.mean().item(),
            #    "root_h_min:", root_h.min().item()
            #)
            obs = self.reset(done_env_ids)

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

        body_pos = rb[:, :, 0:3]
        body_rot = rb[:, :, 3:7]
        body_lin_vel = rb[:, :, 7:10]
        body_ang_vel = rb[:, :, 10:13]

        ref_pos = self.root_pos_ref[self.motion_phase]          # [N, 3]
        ref_rot = self.rot_ref[self.motion_phase]               # [N, 15, 4]
        ref_lin_vel = self.lin_vel_ref[self.motion_phase]       # likely [N, 15, 3]
        ref_ang_vel = self.ang_vel_ref[self.motion_phase]       # likely [N, 15, 3]

        # ---------- rotation reward ----------
        # quaternion sign ambiguity: q and -q are same rotation
        rot_dot = torch.sum(body_rot * ref_rot, dim=-1).abs()
        rot_reward = rot_dot.mean(dim=1)   # [N], higher is better

        # ---------- root position reward ----------
        root_pos = self.root_states[:, 0:3]
        root_pos_error = torch.sum((root_pos - ref_pos) ** 2, dim=-1)
        root_pos_reward = torch.exp(-2.0 * root_pos_error)

        # ---------- root/body velocity reward ----------
        lin_vel_error = torch.mean(torch.sum((body_lin_vel - ref_lin_vel) ** 2, dim=-1), dim=1)
        ang_vel_error = torch.mean(torch.sum((body_ang_vel - ref_ang_vel) ** 2, dim=-1), dim=1)

        lin_vel_reward = torch.exp(-0.1 * lin_vel_error)
        ang_vel_reward = torch.exp(-0.1 * ang_vel_error)

        # ---------- alive / upright reward ----------
        root_h = self.root_states[:, 2]
        alive_reward = (root_h > 0.75).float()

        reward = (
            0.50 * rot_reward +
            0.20 * root_pos_reward +
            0.15 * lin_vel_reward +
            0.10 * ang_vel_reward +
            0.05 * alive_reward
        )

        return reward

    def compute_done(self):
        return self.root_states[:, 2] < 0.5
