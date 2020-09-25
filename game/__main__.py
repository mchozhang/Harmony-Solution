import os
from game.game_env import GameEnv
from game.query_game_data import query_level
from game.mask_q_network import MaskedQNetwork
import game.utils
import tensorflow as tf
from tf_agents.environments import tf_py_environment
from tf_agents.agents.dqn import dqn_agent
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.networks import q_network
from tf_agents.metrics import tf_metrics
from tf_agents.drivers import dynamic_step_driver, dynamic_episode_driver
from tf_agents.policies import random_tf_policy
from tf_agents.trajectories import trajectory
from tf_agents.policies import policy_saver
from tf_agents.utils import common

learning_rate = 0.001
batch_size = 32
collect_steps_per_iteration = 1
initial_collect_steps = 1000
replay_buffer_max_length = 500000
log_interval = 10
save_interval = 50000
eval_interval = 2000
num_eval_episodes = 10
neuron_num_mapper = {3: 25, 4: 40, 5: 75}
num_iterations = 50000

dir_path = os.path.dirname(os.path.realpath(__file__))


def train_level(level,
                consecutive_flag=5,
                collect_random=True,
                max_iterations=num_iterations):
    """
    create DQN agent to train a level of the game
    """
    cells = query_level(level)
    size = len(cells)
    env = GameEnv(cells)
    env = tf_py_environment.TFPyEnvironment(env)
    eval_env = tf_py_environment.TFPyEnvironment(GameEnv(cells))

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    fc_layer_params = (150,)

    q_net = q_network.QNetwork(env.observation_spec()['state'],
                               env.action_spec(),
                               fc_layer_params=fc_layer_params,
                               activation_fn=tf.keras.activations.relu)

    global_step = tf.compat.v1.train.get_or_create_global_step()
    agent = dqn_agent.DdqnAgent(
        env.time_step_spec(),
        env.action_spec(),
        q_network=q_net,
        optimizer=optimizer,
        td_errors_loss_fn=common.element_wise_squared_loss,
        train_step_counter=global_step,
        observation_and_action_constraint_splitter=GameEnv.obs_and_mask_splitter)
    agent.initialize()

    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=agent.collect_data_spec,
        batch_size=env.batch_size,
        max_length=replay_buffer_max_length)

    # drivers
    collect_driver = dynamic_step_driver.DynamicStepDriver(
        env,
        policy=agent.collect_policy,
        observers=[replay_buffer.add_batch],
        num_steps=collect_steps_per_iteration)

    eval_metrics = [
        tf_metrics.AverageReturnMetric(buffer_size=num_eval_episodes),
        tf_metrics.AverageEpisodeLengthMetric(buffer_size=num_eval_episodes),
    ]

    eval_driver = dynamic_episode_driver.DynamicEpisodeDriver(
        eval_env,
        policy=agent.policy,
        observers=eval_metrics,
        num_episodes=num_eval_episodes
    )

    # checkpointer of the replay buffer and policy
    train_checkpointer = common.Checkpointer(
        ckpt_dir=os.path.join(dir_path, 'train'),
        max_to_keep=1,
        agent=agent,
        policy=agent.policy,
        global_step=global_step,
        replay_buffer=replay_buffer)

    # policy saver
    tf_policy_saver = policy_saver.PolicySaver(agent.policy)

    train_checkpointer.initialize_or_restore()

    # optimize by wrapping some of the code in a graph using TF function
    agent.train = common.function(agent.train)
    collect_driver.run = common.function(collect_driver.run)
    eval_driver.run = common.function(eval_driver.run)

    # collect initial replay data
    if collect_random:
        initial_collect_policy = random_tf_policy.RandomTFPolicy(
            time_step_spec=env.time_step_spec(),
            action_spec=env.action_spec(),
            observation_and_action_constraint_splitter=GameEnv.obs_and_mask_splitter)

        dynamic_step_driver.DynamicStepDriver(
            env,
            initial_collect_policy,
            observers=[replay_buffer.add_batch],
            num_steps=initial_collect_steps).run()

    # Dataset generates trajectories with shape [Bx2x...]
    dataset = replay_buffer.as_dataset(
        num_parallel_calls=3,
        sample_batch_size=batch_size,
        num_steps=2).prefetch(3)
    iterator = iter(dataset)

    # train the model until 5 consecutive evaluation have reward greater than 10
    consecutive_eval_win = 0
    train_iterations = 0
    while consecutive_eval_win < consecutive_flag and train_iterations < max_iterations:
        collect_driver.run()

        for _ in range(collect_steps_per_iteration):
            experience, _ = next(iterator)
            train_loss = agent.train(experience).loss

        # evaluate the training at intervals
        step = global_step.numpy()
        if step % eval_interval == 0:
            eval_driver.run()
            average_return = eval_metrics[0].result().numpy()
            average_len = eval_metrics[1].result().numpy()
            print("level: {0} step: {1} AverageReturn: {2} AverageLen: {3}".
                  format(level, step, average_return, average_len))

            # evaluate consecutive wins
            if average_return > 10:
                consecutive_eval_win += 1
            else:
                consecutive_eval_win = 0

        if step % save_interval == 0:
            train_checkpointer.save(global_step=step)

        train_iterations += 1

    # save the policy
    tf_policy_saver.save(os.path.join(dir_path, 'policy_{0}x{0}'.format(size)))


def main():
    game.utils.init_action_mappers()
    train_level(21, 5, True, max_iterations=100000)
    # train_iteration = 3
    # for j in range(train_iteration):
    #     collect_random = j == 0
    #     for i in range(21, 25):
    #         train_level(i, 5 - j,
    #                     collect_random=collect_random,
    #                     max_iterations=num_iterations)


if __name__ == '__main__':
    main()
