import rclpy
import numpy as np
from stable_baselines3 import PPO

from stanley_env import StanleyEnvironment, StanleyLearningNode
from configs import ENV_CFG, OBS_CFG, REWARD_CFG, COMMAND_CFG


def main():
    rclpy.init()
    node = StanleyLearningNode()

    # Deterministic reset for evaluation
    eval_cfg = ENV_CFG.copy()
    eval_cfg["reset_noise"] = 0.0

    env = StanleyEnvironment(
        node,
        eval_cfg,
        OBS_CFG,
        REWARD_CFG,
        COMMAND_CFG
    )

    model = PPO.load("stanley_racing_final", env=env, device="cpu")

    obs, _ = env.reset()

    episode_reward = 0.0
    episode_steps = 0
    episode_count = 0

    try:
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, _ = env.step(action)

            dk, dv = action

            print(
                f"Δk={dk:+.2f} | "
                f"Δv={dv:+.2f} | "
                f"Speed={node.speed:.2f} | "
                f"CTE={abs(node.cte):.4f} | "
                f"Heading={abs(node.heading):.4f} | "
                f"Curvature={node.curvature:.3f}"
            )

            if done:
                print("Collision — stopping evaluation")
                break

    except KeyboardInterrupt:
        print("Evaluation stopped.")

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
