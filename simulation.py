
import numpy as np
import time
from policy_gradient import PolicyGradient
import random
# sample from a with equal prob.
import os
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def simulation():
	users_num = 1
	action_rewards = [10, 9, 1, 1, 1, 1, 1, 1, 1, 1]
	actions = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
	observations = [[random.randint(0, i * 10) for i in range(1, 4)] for j in range(1, 101)]
	# nums of items to recommend
	K = 2
	load_version = 1
	save_version = load_version + 1

	load_path = "output/weights/topk{}.ckpt".format(load_version)
	save_path = "output/weights/topk{}.ckpt".format(save_version)

	EPISODES = 5000
	RENDER_ENV = True
	rewards = []

	PG = PolicyGradient(
		n_x=len(observations[0]),
		n_y=len(actions),
		s0=observations[random.randint(0, len(observations) - 1)],
		learning_rate=0.005,
		reward_decay=1,
		load_path=None,
		save_path=save_path,
		weight_capping_c=2**3,
		k=K,
		b_distribution='uniform'
	)

	for episode in range(EPISODES):

		episode_reward = 0

		tic = time.clock()
		done = False

		while True:
			'''
			TODO:initialize the env
			'''
			if RENDER_ENV:
				observation = observations[random.randint(0, len(observations) - 1)]

			# 1. Choose an action based on observation
			# action = PG.uniform_choose_action(observation)
			action = PG.choose_action(observation)

			# 2. Take action in the environment
			observation_, reward = observations[random.randint(0, len(observations) - 1)], action_rewards[action]

			# 4. Store transition for training
			PG.store_transition(observation, action, reward)

			toc = time.clock()
			elapsed_sec = toc - tic
			if elapsed_sec > 120:
				done = True
			if len(PG.episode_observations) > 100:
				done = True

			if done:
				episode_rewards_sum = sum(PG.episode_rewards)
				rewards.append(episode_rewards_sum)
				max_reward_so_far = np.amax(rewards)
				PG.cost_history.append(episode_rewards_sum)
				print("==========================================")
				print("Episode: ", episode)
				print("Seconds: ", elapsed_sec)
				print("Reward: ", episode_rewards_sum)
				print("Max reward so far: ", max_reward_so_far)
				#print(PG.outputs_softmax)
				print("distribution at {} is :{}".format(PG.s0, PG.get_distribution(PG.s0)))
				# 5. Train neural network
				discounted_episode_rewards_norm = PG.learn()
				break

			# Save new observation
			observation = observation_

	PG.plot_cost()
	plt.bar(actions, PG.get_distribution(PG.s0))
	plt.xlabel("action")
	# 显示纵轴标签
	plt.ylabel("probability")
	# 显示图标题
	plt.title("top-k correction policy")
	plt.show()
simulation()