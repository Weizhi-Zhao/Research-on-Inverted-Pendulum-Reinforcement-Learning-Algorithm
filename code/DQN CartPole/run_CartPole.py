import gym
from RL_brain import DeepQNetwork
import time

env = gym.make('CartPole-v0')

#RL = DeepQNetwork(actions = env.action_so)

RL = DeepQNetwork(actions=env.action_space.n,
                  features=env.observation_space.shape[0],
                  learning_rate=0.005, 
                  reward_decay = 0.95,
                  greedy=0.9995,
                  replace_target_iter=100,
                  memory_size=500,
                  greedy_increment=0.00008)

print(env.action_space.n)
print(env.observation_space.shape[0])

total_steps = 0

for episode in range(250001):
    state = env.reset()
    ep_r = 0
    start_time = time.time()
    print(start_time)
    while True:
        if episode % 20 == 0: 
            env.render()
        action = RL.choose_action(state)
        nxt_state, reward, done, info = env.step(action)

        # the smaller theta and closer to center the better
        position, velocity, angle, velocity_at_tip = nxt_state
        #print('angle:',angle,'  position:', position)
        #r1 = (env.x_threshold - abs(position))/env.x_threshold - 0.8
        #r2 = (env.theta_threshold_radians - abs(angle))/env.theta_threshold_radians - 0.5
        r1 = (2.4 - abs(position) ) / 2.4 - 0.5
        r2 = (0.209 - abs(angle) ) / 0.209 - 0.5
        reward = r1 + r2

        RL.store_transition(state, action, reward, nxt_state)
        print(state)

        ep_r += reward
        if total_steps > 1000:
            RL.learn()
        if abs(position) > 4.1 or abs(angle) > 1.5 or time.time() - start_time > 300:
            print('episode: ', episode,
                  'ep_r: ', round(ep_r, 2),
                  'epsilon: ', round(RL.greedy, 2))
            break

        state = nxt_state
        total_steps += 1
        #time.sleep(1)
    if episode % 1000 == 0:
        RL.save_model()
RL.plot_cost()