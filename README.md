# M^3RL: Mind-aware Multi-agent Management Reinforcement Learning
PyTorch implementation for our ICLR 2019 paper [M^3RL: Mind-aware Multi-agent Management Reinforcement Learning](https://arxiv.org/abs/1810.00147).


## Requirements
- Python 3.6
- Numpy >= 1.15.2
- PyTorch 0.3.1
- termcolor >= 1.1.0

## Task Settings
### Resource Collection

The environments **Collection_v1**, **Collection_v0**, and **Collection_v2** correspond to the S1, S2, and S3 settings in the paper respectively. The multiple bonus levels settings can be run in **Collection_v3** (each agent has 1 skill) and **Collection_v4** (each worker has 3 skills).

### Crafting

**Crafting_v0** and **Crafting_v1** stand for the standard setting and the multiple bonus levels setting respectively in the paper.

## How to Use

Please refer to [examples](./run_commands_examples.md) for how to run training and testing in different settings.

# License 
See [LICENSE](./LICENSE) for additional details.
