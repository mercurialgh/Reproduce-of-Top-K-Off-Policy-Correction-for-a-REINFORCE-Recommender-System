import numpy as np
import time
from policy_gradient import PolicyGradient
import matplotlib.pyplot as plt
import random
# sample from a with equal prob.
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

def simulation():
	users_num = 1
	'''
	action_rewards = {'11':4,'12':1,'13':1,'14':1,'21':1,'22':2,'23':3,'24':16,'31':1,'32':2,'33':3,'34':4}
	observation_action_transfer = {'11':[2],'12':[2],'13':[2],'14':[2],'21':[3],'22':[3],'23':[3],'24':[3],\
			'31':[1],'32':[1],'33':[3],'34':[3]}
	actions = [1,2,3,4]
	observations = [[1],[2],[3]]
	'''

	action_rewards = {'11': 5,'12': 0,'13': 0,'14':0,'15':0,'16':13, \
					  '21': 10,'22': 0, '23': 0,'24':0,'25':0,'26':8}
	observation_action_transfer = {'11': [1,1], '12': [1,1], '13': [1,1],'14':[1,1],'15':[1,1],'16':[1,1], \
								   '21': [1,1], '22': [1,1], '23': [1,1],'24':[1,1],'25':[1,1],'26':[0,1]}

	actions = [1, 2, 3, 4, 5, 6]
	observations = [[0,1], [1,1]]

	# nums of items to recommend
	K = 2
	load_version = 4
	save_version = load_version + 1

	load_path = "output/weights/topk{}.ckpt".format(load_version)
	save_path = "output/weights/topk{}.ckpt".format(save_version)

	EPISODES = 3000
	RENDER_ENV = True
	rewards = []

	PG = PolicyGradient(
		n_x=len(observations[0]),
		n_y=len(actions),
		s0=observations[-1],
		learning_rate=0.001,
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
				observation = PG.episode_observations[-1]
				#print(observation)

			# 1. Choose an action based on observation
			#action = PG.uniform_choose_action(observation)
			action = PG.choose_action(observation)

			# 2. Take action in the environment
			observation_, reward = observation_action_transfer[str(sum(observation))+str(actions[action])], \
								   action_rewards[str(sum(observation))+str(actions[action])]

			# 4. Store transition for training
			PG.store_transition(observation_, action, reward)
			#print(PG.episode_observations)
			#print(PG.episode_actions)
			#print(PG.episode_rewards)
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
				#print(PG.episode_rewards)
				# 5. Train neural network
				print("distribution at {} is :{}".format(observations[0],PG.get_distribution(observations[0])))
				print("distribution at {} is :{}".format(observations[1], PG.get_distribution(observations[1])))
				discounted_episode_rewards_norm = PG.learn()

				break

			# Save new observation
			observation = observation_
	PG.plot_cost()
	plt.bar(actions, PG.get_distribution(observations[0]))
	plt.xlabel("action at state[0,1]")
	# 显示纵轴标签
	plt.ylabel("probability")
	# 显示图标题
	plt.title("policy distribution at state[0,1]")
	plt.show()
	plt.bar(actions, PG.get_distribution(observations[1]))
	plt.xlabel("action at state[1,1]")
	# 显示纵轴标签
	plt.ylabel("probability")
	# 显示图标题
	plt.title("policy distribution at state[1,1]")
	plt.show()
simulation()
