# configs.py

ENV_CFG = {
    # Environment / timing
    "step_delay": 0.02,
    "max_track_width": 0.8,
    "max_episode_steps": 500,

    # Reset
    "init_pos": [0.29, -0.77],
    "init_quat": [0.0, 0.0, 0.997, -0.072],
    "reset_noise": 0.5,

    # Curvature normalization
    "curvature_norm": 0.25,
}

OBS_CFG = {
    # kept for future extensibility
    "use_normalization": True
}

REWARD_CFG = {
    "scales": {
        # These now map 1-to-1 to the simple reward
        "speed": 1.0,
        "cte": 2.0,
        "heading": 1.0,
        "curvature_speed": 6.0,
        "collision": 30.0
    }
}

COMMAND_CFG = {
    # These no longer define absolute commands
    # They define expected operating ranges (for logging / sanity)
    "speed_min": 1.0,
    "speed_max": 5.0,

    # Correction magnitudes (used by controller design)
    "k_corr_scale": 0.2,
    "v_corr_scale": 0.3
}
