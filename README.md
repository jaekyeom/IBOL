# Information Bottleneck Option Learning (IBOL)

This is the code for our paper,
* *[Jaekyeom Kim](https://jaekyeom.github.io/)\*, Seohong Park\* and [Gunhee Kim](http://vision.snu.ac.kr/gunhee/)* (\*equal contribution). Unsupervised Skill Discovery with Bottleneck Option Learning. In *ICML*, 2021. [paper] [arxiv] [talk]

It includes the implementation for IBOL, specifically, the linearizer, the skill discovery method on top of it and the downstream tasks for the evaluation of them.

### Example Skills

We show some example skills discovered by IBOL in four MuJoCo environments without rewards.

#### Ant

###### Locomotion skills
<img src="https://bucket-california.s3-us-west-1.amazonaws.com/ibol/ant_xy_256.gif" width="400">

###### Rotation skills (complementary to Figure 6 in the main paper)
![](https://bucket-california.s3-us-west-1.amazonaws.com/ibol/ant_ori_1.gif)
![](https://bucket-california.s3-us-west-1.amazonaws.com/ibol/ant_ori_2.gif)
![](https://bucket-california.s3-us-west-1.amazonaws.com/ibol/ant_ori_3.gif)

#### Humanoid
![](https://bucket-california.s3-us-west-1.amazonaws.com/ibol/hum_1.gif)
![](https://bucket-california.s3-us-west-1.amazonaws.com/ibol/hum_2.gif)

![](https://bucket-california.s3-us-west-1.amazonaws.com/ibol/hum_3.gif)
![](https://bucket-california.s3-us-west-1.amazonaws.com/ibol/hum_4.gif)

#### HalfCheetah
![](https://bucket-california.s3-us-west-1.amazonaws.com/ibol/ch_4.gif)
![](https://bucket-california.s3-us-west-1.amazonaws.com/ibol/ch_2.gif)
![](https://bucket-california.s3-us-west-1.amazonaws.com/ibol/ch_1.gif)
![](https://bucket-california.s3-us-west-1.amazonaws.com/ibol/ch_3.gif)

#### Hopper (at 5x speed)
![](https://bucket-california.s3-us-west-1.amazonaws.com/ibol/hp_4.gif)
![](https://bucket-california.s3-us-west-1.amazonaws.com/ibol/hp_3.gif)
![](https://bucket-california.s3-us-west-1.amazonaws.com/ibol/hp_2.gif)
![](https://bucket-california.s3-us-west-1.amazonaws.com/ibol/hp_1.gif)

### Requirements
This code is tested in environments with the following conditions:
* Ubuntu 16.04 machine
* CUDA-compatible GPUs
* Python 3.7.10

### Environment Setup
1. Install [MuJoCo](http://mujoco.org/) version 2.0 binaries, following [the instructions](https://github.com/openai/mujoco-py#install-mujoco).
    Note that they offer multiple licensing choices including the 30-day free trials.
2. At the top-level directory, run the following command to set up the environment.
    ```
    pip install --no-deps -r requirements.txt
    ```

### Training
#### Linearizer
| Command                                                                     |       Environment     |
|-----------------------------------------------------------------------------|:---------------------:|
| `python tests/main.py --train_type linearizer --env ant`                    |          Ant          |
| `python tests/main.py --train_type linearizer --env half_cheetah`           |      HalfCheetah      |
| `python tests/main.py --train_type linearizer --env hopper`                 |         Hopper        |
| `python tests/main.py --train_type linearizer --env humanoid`               |        Humanoid       |
| `python tests/main.py --train_type linearizer --env dkitty_randomized `     |   D'Kitty Randomized  |


#### Skill discovery
| Command                                                                                                                 |       Environment     |
|-------------------------------------------------------------------------------------------------------------------------|:---------------------:|
| `python tests/main.py --train_type skill_discovery --env ant --cp_path "exp/L_ANT/sampling_policy.pt"`                  |          Ant          |
| `python tests/main.py --train_type skill_discovery --env half_cheetah --cp_path "exp/L_HC/sampling_policy.pt"`          |      HalfCheetah      |
| `python tests/main.py --train_type skill_discovery --env hopper --cp_path "exp/L_HP/sampling_policy.pt"`                |         Hopper        |
| `python tests/main.py --train_type skill_discovery --env humanoid --cp_path "exp/L_HUM/sampling_policy.pt"`             |        Humanoid       |
| `python tests/main.py --train_type skill_discovery --env dkitty_randomized --cp_path "exp/L_DK/sampling_policy.pt"`     |   D'Kitty Randomized  |


#### Downstream tasks
| Command                                                                                                                                                     |       Environment        |
|-------------------------------------------------------------------------------------------------------------------------------------------------------------|:------------------------:|
| `python tests/main.py --train_type downstream --env ant_goal --cp_path "exp/L_ANT/sampling_policy.pt" --dcp_path "exp/S_ANT/option_policy.pt"`              |         AntGoal          |
| `python tests/main.py --train_type downstream --env ant_multi_goals --cp_path "exp/L_ANT/sampling_policy.pt" --dcp_path "exp/S_ANT/option_policy.pt"`       |      AntMultiGoals       |
| `python tests/main.py --train_type downstream --env half_cheetah_goal --cp_path "exp/L_CH/sampling_policy.pt" --dcp_path "exp/S_CH/option_policy.pt"`       |       CheetahGoal        |
| `python tests/main.py --train_type downstream --env half_cheetah_imi --cp_path "exp/L_CH/sampling_policy.pt" --dcp_path "exp/S_CH/option_policy.pt"`        |     CheetahImitation     |


### Evaluation
* Each training command stores its results in an experiment directory under `exp/`.
* In each experiment directory, `plots/` (files) or `tb_plot/` (tensorboard) contain qualitative visualizations.
* For downstream tasks, the column named `TrainSp/IOD/SmoothedReward500` in `progress.csv` can be examined.

### Acknowledgments
This code is based on [garage](https://github.com/rlworkgroup/garage/).
