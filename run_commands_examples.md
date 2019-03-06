# Examples of training and testing

## Training on a fixed population
#### Example 1: Resource Collection (S1 setting), a population of 40 workers
```
python train.py --exp-name Collection_v1 --max-episode-length 30 \
 --max-nb-agents 4 --max-nb-resources 10 \
 --nb-resource-types 4 --nb-agent-types 4 --nb-pay-types 2 \
 --att-norm --IL \
 --pop-size 40 --interval 1 \
 --eps-episodes 5000 --max-nb-episodes 250000 --seed 1389 --noise 0.3
```

#### Example 2: Crafting, where the commitment constraint is 10 steps -- the manager is allowed to update contract proposals once for every 10 time steps
 ```
python train.py --exp-name Crafting_v0 --max-episode-length 30 \
    --max-nb-agents 8 --max-nb-resources 10 \
    --nb-resource-types 8 --nb-agent-types 4 --nb-pay-types 3 \
    --att-norm --IL \
    --pop-size 40 --interval 10 \
    --eps-episodes 10000 --max-nb-episodes 200000 --max-nb-episodes-warm 80000 --seed 1389 
```

## Training on a fixed population with randomized worker policies
#### Example: Resource Collection (S1 setting), 30% random actions in worker policies
```
python train.py --exp-name Collection_v1 --max-episode-length 30 \
    --max-nb-agents 4 --max-nb-resources 10 \
    --nb-resource-types 4 --nb-agent-types 4 --nb-pay-types 2 \
    --att-norm --IL \
    --pop-size 40 --interval 1 \
    --eps-episodes 5000 --max-nb-episodes 250000 --seed 1389 --noise 0.3
```

## Training in the settings of multiple bonus levels
#### Example 1: Resource Collection (each worker has 1 skill), 4 bonus levels
```
python train.py --exp-name Collection_v3 --max-episode-length 30 \
    --max-nb-agents 4 --max-nb-resources 10 \
    --nb-resource-types 4 --nb-agent-types 4 --nb-pay-types 4 \
    --att-norm --IL \
    --pop-size 40 --interval 1 \
    --eps-episodes 5000 --max-nb-episodes 250000 --seed 1389
```

#### Example 2: Crafting, 5 bonus levels
```
python train.py --exp-name Crafting_v1 --max-episode-length 30 \
    --max-nb-agents 8 --max-nb-resources 10 \
    --nb-resource-types 8 --nb-agent-types 4 --nb-pay-types 5 \
    --att-norm --IL \
    --pop-size 40 --interval 10 \
    --eps-episodes 10000 --max-nb-episodes 200000 --max-nb-episodes-warm 80000 --seed 1389
```

## Testing on the same population as in training
#### Example: Resource Collection (S1), model from checkpoint 250000
Same scenarios as in training
```
python test.py --exp-name Collection_v1 --max-episode-length 30 \
--max-nb-agents 4 --max-nb-resources 10 --test-max-nb-agents 4 \
--nb-resource-types 4 --nb-agent-types 4 --nb-pay-types 2 \
--att-norm --IL \
--pop-size 40 --interval 1 \
--seed 1389 --checkpoint 250000
```

New scenarios: 6 agents in each testing episode
```
python test.py --exp-name Collection_v1 --max-episode-length 30 \
--max-nb-agents 4 --max-nb-resources 10 --test-max-nb-agents 6 \
--nb-resource-types 4 --nb-agent-types 4 --nb-pay-types 2 \
--att-norm --IL \
--pop-size 40 --interval 1 \
--seed 1389 --checkpoint 250000
```

New scenarios: adding obstacles to the environment
```
python test.py --exp-name Collection_v1 --max-episode-length 30 \
--max-nb-agents 4 --max-nb-resources 10 --test-max-nb-agents 4 \
--nb-resource-types 4 --nb-agent-types 4 --nb-pay-types 2 \
--att-norm --IL \
--pop-size 40 --interval 1 \
--seed 1389 --checkpoint 250000 --obstacle
```

## Testing on an unseen and periodically changing population
#### Example: Resource Collection (S1), replacing 75% workers with new ones after every 2000 episodes
```
python test.py --exp-name Collection_v1 --max-episode-length 30 \
    --max-nb-agents 4 --max-nb-resources 10 --test-max-nb-agents 4 \
    --nb-resource-types 4 --nb-agent-types 4 --nb-pay-types 2 \
    --att-norm --IL \
    --pop-size 40 --interval 1 \
    --eps-episodes 5000 --seed 1389 \
    --checkpoint 250000 --new-pop --test-pop-size 4 --test-update-pct 0.75 --test-update-freq 2000
```

## Training on a changing population through multiple phases
#### Example: Resource Collection
**Phase 1**, skill distribution (how many agents have the skills): [100%, 0, 0, 0]
Note that in phase 1, we may set `nb_agent_types = 1` (i.e., all agents have skill 1) to get the desired skill distribution.
```
python train.py --exp-name Collection_v0 --max-episode-length 30 \
--max-nb-agents 4 --max-nb-resources 10 \
--nb-resource-types 4 --nb-agent-types 1 --nb-pay-types 2 \
--att-norm --IL --final-epsilon 0.1 \
--max-exp-steps 100000 --eps-episodes 1000000 --pop-size 40 --max-nb-episodes 100000 --seed 1389
```

**Phase 2**, skill distribution: [100%, 25%, 0, 0]
```
python train.py --exp-name Collection_v0 --max-episode-length 30 \
    --max-nb-agents 4 --max-nb-resources 10 \
    --nb-resource-types 4 --nb-agent-types 1 --nb-pay-types 2 \
    --att-norm --IL --final-epsilon 0.1 \
    --max-exp-steps 100000 --eps-episodes 1000000 --pop-size 40 --max-nb-episodes 200000 --seed 1389 \
    --init-checkpoint 100000 --update-skill-goal 1 --update-skill-pop 10 --update-skill-inc 1
```

**Phase 3**, skill distribution: [50%, 37.5%, 25%, 25%]
```
python train.py --exp-name Collection_v0 --max-episode-length 30 \
    --max-nb-agents 4 --max-nb-resources 10 \
    --nb-resource-types 4 --nb-agent-types 1 --nb-pay-types 2 \
    --att-norm --IL --final-epsilon 0.1 \
    --max-exp-steps 100000 --eps-episodes 1000000 --pop-size 40 --max-nb-episodes 300000 --seed 1389 \
    --init-checkpoint 200000 --update-skill-goal 0 1 2 3 --update-skill-pop 20 5 10 10 --update-skill-inc 0 1 1 1
```

**Phase 4**, skill distribution: [37.5%, 37.5%, 37.5%, 37.5%]
```
python train.py --exp-name Collection_v0 --max-episode-length 30 \
    --max-nb-agents 4 --max-nb-resources 10 \
    --nb-resource-types 4 --nb-agent-types 1 --nb-pay-types 2 \
    --att-norm --IL --final-epsilon 0.1 \
    --max-exp-steps 100000 --eps-episodes 1000000 --pop-size 40 --max-nb-episodes 400000 --seed 1389 \
    --init-checkpoint 300000 --update-skill-goal 0 2 3 --update-skill-pop 5 5 5 --update-skill-inc 0 1 1 
```