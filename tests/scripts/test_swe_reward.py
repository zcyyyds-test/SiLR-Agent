from scripts.train_swe_grpo import compute_swe_reward


def test_reward_zero_when_patch_rejected():
    r = compute_swe_reward(ast_ok=False, imports_ok=False,
                           regression_pass=False, target_pass=False)
    assert r == 0.0


def test_reward_target_pass_full():
    r = compute_swe_reward(ast_ok=True, imports_ok=True,
                           regression_pass=True, target_pass=True)
    assert r == 1.50


def test_reward_monotone():
    r1 = compute_swe_reward(True, False, False, False)     # 0.10
    r2 = compute_swe_reward(True, True,  False, False)     # 0.25
    r3 = compute_swe_reward(True, True,  True,  False)     # 0.50
    r4 = compute_swe_reward(True, True,  True,  True)      # 1.50
    assert r1 < r2 < r3 < r4
    assert abs(r1 - 0.10) < 1e-9
    assert abs(r2 - 0.25) < 1e-9
    assert abs(r3 - 0.50) < 1e-9
    assert abs(r4 - 1.50) < 1e-9
