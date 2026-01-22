import rclpy
import wandb

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from wandb.integration.sb3 import WandbCallback

from stanley_env import StanleyEnvironment, StanleyLearningNode
from configs import ENV_CFG, OBS_CFG, REWARD_CFG, COMMAND_CFG


# =========================================================
# ENV FACTORY
# =========================================================

def make_env():
    node = StanleyLearningNode()
    env = StanleyEnvironment(
        node,
        ENV_CFG,
        OBS_CFG,
        REWARD_CFG,
        COMMAND_CFG
    )
    env = Monitor(env)
    env.ros_node = node
    return env


# =========================================================
# TRAINING
# =========================================================

def main():
    rclpy.init()

    env = DummyVecEnv([make_env])

    run = wandb.init(
        project="f1tenth_stanley_self_tuning",
        config={
            **ENV_CFG,
            **REWARD_CFG["scales"],
            **COMMAND_CFG
        },
        sync_tensorboard=True,
        save_code=True
    )

    model = PPO(
        policy="MlpPolicy",
        env=env,
        n_steps=2048,
        batch_size=256,
        learning_rate=3e-4,
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,
        verbose=1,
        tensorboard_log=f"runs/{run.id}",
        device="cpu"
    )

    model.learn(
        total_timesteps=30_000,
        callback=WandbCallback(
            gradient_save_freq=200,
            verbose=1
        )
    )

    model.save("stanley_racing_final")

    env.envs[0].ros_node.destroy_node()
    env.close()
    rclpy.shutdown()
    run.finish()


if __name__ == "__main__":
    main()
