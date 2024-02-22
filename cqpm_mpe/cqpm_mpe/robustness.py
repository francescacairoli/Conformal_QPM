# Author: Tom Kuipers, King's College London
import numpy as np
import os
from pcheck.semantics import stlRobustSemantics
from pcheck.series.TimeSeries import TimeSeries
from multiprocessing.pool import Pool
from multiprocessing import shared_memory
import multiprocessing as mp
import pickle
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

from cqpm_mpe.config import *


class STLRobustnessChecker:

    def __init__(self, time_robust=False):
        self.timeseries = None
        self.seeds = None
        self.robust_dist_shape = (CONFIG[CFG_DATA]['n_states'], CONFIG[CFG_DATA]['n_suffixes'])
        self.state_dimen = int((CONFIG[CFG_ENV]['n_total_agents'] * 4) + (CONFIG[CFG_ENV]['n_landmarks'] * 2))
        self.scaling_factor = 4
        self.time_robust = time_robust
        self.load_trajectories()

    def set_time_robust(self, time_robust):
        self.time_robust = time_robust

    def load_trajectories(self):
        try:
            self.seeds = sorted(list(np.load(os.path.join(CONFIG[CFG_PATH]['data'], "seeds.npy"))))
            self.output_dir = os.path.join(Config.DATA[CFG_PATH]["root"], "stl")
            if not os.path.exists(self.output_dir): os.makedirs(self.output_dir)
        except:
            print("[ERRO] Could not load seeds. Check data dir.")
            exit(1)

    def compute_distances(self, pos1, pos2):
        dists = np.zeros(shape=(pos1.shape[0]))
        dists[:] = np.linalg.norm((pos1 - pos2), axis=1)
        return dists

    def create_timeseries(self, trajectory):
        timesteps = np.arange(trajectory.shape[0])
        descriptors = ['COLLISION']
        sequences = []
        distances = []
        # Compute distances to all other agents
        for i in range(0, CONFIG[CFG_ENV]['n_total_agents'] * 4, 4):
            for j in range(i, CONFIG[CFG_ENV]['n_total_agents'] * 4, 4):
                if i != j:
                    distances.append(self.compute_distances(trajectory[:, i:(i+2)], trajectory[:, j:(j+2)]))
        # Stack distances across time
        dist_t = np.stack(distances, axis=-1)
        min_dists = np.min(dist_t, axis=1)
        sequences.append(min_dists)
        timeseries = TimeSeries(descriptors, timesteps, np.array(sequences))
        return timeseries
    
    def property_collision(self, t_start=0, t_len=10):
        safe_dist = (0.5 / self.scaling_factor)
        score = stlRobustSemantics(
            self.timeseries,
            t_start,
            f'(G_[{t_start},{t_len}] (COLLISION>={safe_dist}))'
        )
        return score
    
    def compute_time_robustness(self, robust_vals):
        # NOTE Treat 0 as positive or negative
        boolean_robust = robust_vals >= 0
        # Initial satisfaction value
        init_sat = boolean_robust[0]
        # Get the timestep at which the sign flip occurs
        # If there is no flip (fully satisfied), the below code will throw an exception
        # t_flip will be an empty array
        try: 
            t_flip = np.where(boolean_robust == (not init_sat))[0][0]
        # Return n_steps if satisfaction did not change
        # -1 as we include the final time step when evaluating robustness
        except IndexError:
            t_flip = boolean_robust.shape[0] - 1
        # If spec was initially violated, then negate it
        if not init_sat: t_flip *= -1
        return t_flip / CONFIG[CFG_DATA]['suffix_len']
    
    def evaluate_trajectory(self, trajectory, t_len):
        scaled_trajectory = trajectory / self.scaling_factor
        self.timeseries = self.create_timeseries(trajectory)
        if self.time_robust:
            robust_vals = np.array([ self.property_collision(t_len=t) for t in range(1, t_len + 1) ])
            return self.compute_time_robustness(robust_vals)
        else:
            robust_val = self.property_collision(t_len=t_len + 1)
            return robust_val / self.scaling_factor

    def test_stl(self, seed):
        traj = np.load(os.path.join(CONFIG[CFG_PATH]['data'], f"seed_{seed}", f"state_ground_traj_0.npy"))
        print(self.evaluate_trajectory(traj, t_len=CONFIG[CFG_DATA]['trajectory_len']))

    def compute_stl(self, seed, seed_idx):
        existing_shm = shared_memory.SharedMemory(name="stl_vals")
        stl_robust_dist = np.ndarray(shape=self.robust_dist_shape, dtype=np.float32, buffer=existing_shm.buf)
        seed_path = os.path.join(CONFIG[CFG_PATH]['data'], f"seed_{seed}")
        for i in range(self.robust_dist_shape[1]):
            traj = np.load(os.path.join(seed_path, f"state_ground_traj_{i}.npy"))
            stl_robust_dist[seed_idx, i] = self.evaluate_trajectory(traj, t_len=CONFIG[CFG_DATA]['trajectory_len'])
        existing_shm.close()

    def compute_stl_parallel(self):
        temp = np.zeros(shape=self.robust_dist_shape, dtype=np.float32)
        shm = shared_memory.SharedMemory(create=True, size=temp.nbytes, name="stl_vals")
        robust_vals = np.ndarray(temp.shape, dtype=temp.dtype, buffer=shm.buf)
        robust_vals[:] = temp[:]

#        mp.set_start_method('spawn')
        pool = Pool(processes=CONFIG[CFG_SIM]['generator']['n_threads'], maxtasksperchild=10)
        job_num = -1
        # Keep scheduling jobs so long as we have threads free and more seeds
        # to evaluate
        seeds = list(self.seeds)
        seeds.reverse()
        while(seeds):
            # if len(seeds) <= 995: break
            job_num += 1
            seed = seeds.pop()
            pool.apply_async(self.compute_stl, (seed, job_num,))
        
        pool.close()  # Done adding tasks.
        pool.join()  # Wait for all tasks to complete.
        fn_robust = "stl_time_robust.npy" if self.time_robust else "stl_spatial_robust.npy"
        np.save(os.path.join(self.output_dir , fn_robust), robust_vals)
        shm.close()
        shm.unlink()
        print("DONE!")

    def compute_stl_distribution(self):
        self.compute_stl_parallel()

    def load_prefix(self):
        seeds = list(self.seeds)
        temp = np.zeros(shape=(len(seeds), self.state_dimen), dtype=np.float64)
        for idx, seed in enumerate(seeds):
            temp[idx] = np.load(os.path.join(CONFIG[CFG_PATH]['data'], f"seed_{seed}", "state_ground_prefix.npy"))[0]
        x_scaled = temp / self.scaling_factor
        np.save(os.path.join(CONFIG[CFG_PATH]['root'], "stl", "x_scaled.npy"), x_scaled)

    def save_pickle(self):
        x_scaled = np.load(os.path.join(CONFIG[CFG_PATH]['root'], "stl", "x_scaled.npy"))
        spatial = np.load(os.path.join(CONFIG[CFG_PATH]['root'], "stl", "stl_spatial_robust.npy"))
        time = np.load(os.path.join(CONFIG[CFG_PATH]['root'], "stl", "stl_time_robust.npy"))
        output_dict = {"x_scaled": x_scaled, "rob": spatial, "time_rob": time}
        with open(os.path.join(CONFIG[CFG_PATH]['root'], "stl", f"mpe_agt_{CONFIG[CFG_ENV]['n_agents']}_adv_{CONFIG[CFG_ENV]['n_adversaries']}_{CONFIG[CFG_PATH]['dataset_name']}.pickle"), 'wb') as handle:
            pickle.dump(output_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    mp.set_start_method('spawn')
    STLChecker = STLRobustnessChecker()
    print("Saving state data")
    STLChecker.load_prefix()
    print("Generating spatial robustness")
    STLChecker.compute_stl_distribution()
    STLChecker.set_time_robust(True)
    print("Generating time robustness")
    STLChecker.compute_stl_distribution()
    print("Saving pickle to disk")
    STLChecker.save_pickle()
