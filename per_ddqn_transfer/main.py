from collections import deque
import os
import random
from tqdm import tqdm

import torch

from utils_drl import Agent
from utils_env import MyEnv
from utils_memory import ReplayMemory

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

restore = './init.weights'

GAMMA = 0.99
epsilon = 1e-5
GLOBAL_SEED = 0
MEM_SIZE = 100_000
RENDER = False
SAVE_PREFIX = "./models"
STACK_SIZE = 4

EPS_START = 0.1
EPS_END = 0.1
EPS_DECAY = 1000000
alpha = 0.6
beta = 1
beta_increment = 6e-8

BATCH_SIZE = 32
POLICY_UPDATE = 4
TARGET_UPDATE = 10_000
WARM_STEPS = 50_000
MAX_STEPS = 50_000_000
EVALUATE_FREQ = 100_000

rand = random.Random()
rand.seed(GLOBAL_SEED)
new_seed = lambda: rand.randint(0, 1000_000)

torch.manual_seed(new_seed())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = MyEnv(device)
agent = Agent(
    env.get_action_dim(),
    device,
    GAMMA,
    epsilon,
    new_seed(),
    EPS_START,
    EPS_END,
    EPS_DECAY,
    restore
)
memory = ReplayMemory(STACK_SIZE + 1, MEM_SIZE, alpha, beta, beta_increment, device)

#### Training ####
obs_queue: deque = deque(maxlen=5)
done = True

progressive = tqdm(range(MAX_STEPS), total=MAX_STEPS,
                   ncols=50, leave=False, unit="b")
for step in progressive:
    if done:
        observations, _, _ = env.reset()
        for obs in observations:
            obs_queue.append(obs)

    training = len(memory) > WARM_STEPS
    state = env.make_state(obs_queue).to(device).float()
    action = agent.run(state, training)
    obs, reward, done = env.step(action)
    obs_queue.append(obs)
    memory.push(env.make_folded_state(obs_queue), action, reward, done)

    if step % POLICY_UPDATE == 0 and training:
        agent.learn(memory, BATCH_SIZE)

    if step % TARGET_UPDATE == 0:
        agent.sync()

    if step % EVALUATE_FREQ == 0:
        avg_reward, frames = env.evaluate(obs_queue, agent, 1, render=RENDER)
        with open("rewards.txt", "a") as fp:
            fp.write(f"{step//EVALUATE_FREQ:3d} {step:8d} {avg_reward:.1f}\n")
        if RENDER:
            prefix = f"eval_{step//EVALUATE_FREQ:03d}"
            os.mkdir(prefix)
            for ind, frame in enumerate(frames):
                with open(os.path.join(prefix, f"{ind:06d}.png"), "wb") as fp:
                    frame.save(fp, format="png")
        agent.save(os.path.join(
            SAVE_PREFIX, f"model_{step//EVALUATE_FREQ:03d}"))
        done = True
