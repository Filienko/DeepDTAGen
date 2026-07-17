"""Named model presets for the MPC tooling, so a config is one flag across
op-cost / profiling / export / NssMPClib. Reuses the exact `CNNDTA` kwargs the
runs were trained with.

  regB  -- the Davis accuracy winner (balAcc 0.848): ~800K dilated CNN+CNN,
           protein dilations 1,2,4, MAX pool. This is the model the user
           benchmarked under MP-SPDZ (2.6s CPU online) -- see ONLINE_TIME.md.
  cfgA  -- the smallest FSS-ideal model (267K): proj32 head, mean-friendly.
"""
CONFIGS = {
    "regB": dict(pool="max", proj_dim=0, head_layers=2, head_dim=1024,
                 drug_channels=[16, 32, 48], prot_channels=[32, 64, 96],
                 prot_dilations=[1, 2, 4], drug_kernel=4, prot_kernel=8, embed_dim=128),
    "cfgA": dict(pool="mean", proj_dim=32, head_layers=1, head_dim=1536,
                 drug_filters=32, prot_filters=32, drug_kernel=4, prot_kernel=8,
                 embed_dim=128),
}

# expected trainable param counts (a build-time sanity check)
EXPECTED_PARAMS = {"regB": 800417, "cfgA": 267073}


def build_cnndta(name):
    from models import CNNDTA, count_params
    if name not in CONFIGS:
        raise KeyError(f"unknown config {name!r}; have {list(CONFIGS)}")
    m = CNNDTA(**CONFIGS[name]).eval()
    exp = EXPECTED_PARAMS.get(name)
    if exp is not None:
        got = count_params(m)
        assert got == exp, f"{name}: built {got} params, expected {exp}"
    return m
