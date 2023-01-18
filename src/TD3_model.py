from datetime import datetime
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import tracks

from Base_model import Base_model

racer = tracks.Racer()


class TD3(Base_model):
    def __init__(self, load_weights=True, model_name="TD3", weight_path=None):
        """Constructor for the model

        Args:
            load_weights (bool, optional): If set to true loads weights for the model a weight file. Defaults to True.
            model_name (str, optional): Name of the model (will be used to find weight paths). Defaults to "td3".
            weight_path (str, optional): Base folder for weights. Defaults to "../weights".
        """
        super().__init__(model_name)
        self.num_states = 5  # we reduce the state dim through observation (see below)
        self.num_actions = 2  # acceleration and steering
        print("State Space dim: {}, Action Space dim: {}".format(
            self.num_states, self.num_actions))

        if weight_path is None:
            weight_path = f"./{model_name}/{model_name}_weights"

        # Actor weights path
        self.weights_file_actor = f"{weight_path}/{model_name}_actor_model_car"
        self.weights_file_critic = f"{weight_path}/{model_name}_critic_model_car"
        self.weights_file_critic2 = f"{weight_path}/{model_name}_critic2_model_car"

        # creating models
        self.actor_model = self.get_actor()
        self.load_weights = load_weights
        if load_weights:
            self.actor_model = keras.models.load_model(self.weights_file_actor)

    # The actor choose the move, given the state
    def get_actor(self):
        # no special initialization is required
        # Initialize weights between -3e-3 and 3-e3
        # last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

        inputs = layers.Input(shape=(self.num_states, ))
        out = layers.Dense(64, activation="relu")(inputs)
        out = layers.Dense(64, activation="relu")(out)
        # outputs = layers.Dense(num_actions, kernel_regularizer=regularizers.l2(0.01), kernel_initializer=last_init)(out)
        # outputs = layers.Activation('tanh')(outputs)
        # outputs = layers.Dense(num_actions, name="out", activation="tanh", kernel_initializer=last_init)(out)
        outputs = layers.Dense(self.num_actions, name="out",
                               activation="tanh")(out)

        # outputs = outputs * upper_bound
        model = tf.keras.Model(inputs, outputs, name="actor")
        return model

    # the critic compute the q-value, given the state and the action
    def get_critic(self):
        # State as input
        state_input = layers.Input(shape=(self.num_states))
        # Action as input
        action_input = layers.Input(shape=(self.num_actions))

        concat = layers.Concatenate()([state_input, action_input])

        out = layers.Dense(64, activation="relu")(concat)
        out = layers.Dense(64, activation="relu")(out)
        outputs = layers.Dense(1)(out)  # Outputs single value

        model = tf.keras.Model([state_input, action_input],
                               outputs,
                               name="critic")

        return model

    # Replay buffer
    class Buffer:
        def __init__(self, model, buffer_capacity=100000, batch_size=64):
            # Max Number of tuples that can be stored
            self.buffer_capacity = buffer_capacity
            # Num of tuples used for training
            self.batch_size = batch_size

            # Current number of tuples in buffer
            self.buffer_counter = 0

            # We have a different array for each tuple element
            self.state_buffer = np.zeros(
                (self.buffer_capacity, model.num_states))
            self.action_buffer = np.zeros(
                (self.buffer_capacity, model.num_actions))
            self.reward_buffer = np.zeros((self.buffer_capacity, 1))
            self.done_buffer = np.zeros((self.buffer_capacity, 1))
            self.next_state_buffer = np.zeros(
                (self.buffer_capacity, model.num_states))

        # Stores a transition (s,a,r,s') in the buffer
        def record(self, obs_tuple):
            s, a, r, T, sn = obs_tuple
            # restart form zero if buffer_capacity is exceeded, replacing old records
            index = self.buffer_counter % self.buffer_capacity

            self.state_buffer[index] = tf.squeeze(s)
            self.action_buffer[index] = a
            self.reward_buffer[index] = r
            self.done_buffer[index] = T
            self.next_state_buffer[index] = tf.squeeze(sn)

            self.buffer_counter += 1

        def sample_batch(self):
            # Get sampling range
            record_range = min(self.buffer_counter, self.buffer_capacity)
            # Randomly sample indices
            batch_indices = np.random.choice(record_range, self.batch_size)

            s = self.state_buffer[batch_indices]
            a = self.action_buffer[batch_indices]
            r = self.reward_buffer[batch_indices]
            T = self.done_buffer[batch_indices]
            sn = self.next_state_buffer[batch_indices]
            return (s, a, r, T, sn)

    # Slowly updating target parameters according to the tau rate <<1
    @tf.function
    def update_target(target_weights, weights, tau):
        for (a, b) in zip(target_weights, weights):
            a.assign(b * tau + a * (1 - tau))

    def update_weights(target_weights, weights, tau):
        return target_weights * (1 - tau) + weights * tau

    def policy(self, state, verbose=False):
        # the policy used for training just add noise to the action
        # the amount of noise is kept constant during training
        sampled_action = tf.squeeze(self.actor_model(state))
        noise = np.random.normal(scale=0.1, size=2)
        # we may change the amount of noise for actions during training
        noise[0] *= 2
        noise[1] *= 0.5
        # Adding noise to action
        sampled_action = sampled_action.numpy()
        sampled_action += noise
        # in verbose mode, we may print information about selected actions
        if verbose and sampled_action[0] < 0:
            print("decelerating")

        # Finally, we ensure actions are within bounds
        legal_action = np.clip(sampled_action, self.lower_bound,
                               self.upper_bound)

        return [np.squeeze(legal_action)]

    def policy_target(self, state, verbose=False):
        # the policy used for training just add noise to the action
        # the amount of noise is kept constant during training
        newactions = []
        sampled_action = tf.squeeze(self.target_actor(state))
        for a in sampled_action:
            noise = np.random.normal(scale=0.1, size=2)
            # we may change the amount of noise for actions during training
            noise[0] *= 2
            noise[1] *= 0.5
            noise = np.clip(noise, -self.noise_clip, self.noise_clip)
            # Adding noise to action
            a = a.numpy()
            a += noise
            legal_action = np.clip(a, self.lower_bound, self.upper_bound)
            legal_action = np.squeeze(legal_action)
            newactions.append(legal_action)

        return np.asarray(tf.squeeze(newactions))

    def compose(self, actor, critic):
        state_input = layers.Input(shape=(self.num_states))
        a = actor(state_input)
        q = critic([state_input, a])
        # reg_weights = actor.get_layer('out').get_weights()[0]
        # print(tf.reduce_sum(0.01 * tf.square(reg_weights)))

        m = tf.keras.Model(state_input, q)
        # the loss function of the compound model is just the opposite of the critic output
        m.add_loss(-q)
        return m

    # We introduce a probability of doing n empty actions to separate the environment time-step from the agent
    def step(self, action):
        n = 1
        t = np.random.randint(0, n)
        state, reward, done = racer.step(action)
        for i in range(t):
            if not done:
                state, t_r, done = racer.step([0, 0])
                # state ,t_r, done =racer.step(action)
                reward += t_r
        return (state, reward, done)

    def train(
        self,
        total_iterations=50000,
        save_weights=True,
        save_name=None,
        plot_results=True,
    ):
        i = 0
        mean_speed = 0
        ep = 0
        avg_reward = 0

        self.noise_clip = 0.5
        target_freq = 2
        # Discount factor
        gamma = 0.99
        # Target network parameter update factor, for double DQN
        tau = 0.005
        # Learning rate for actor-critic models
        critic_lr = 0.001
        aux_lr = 0.001

        buffer_dim = 50000
        batch_size = 64

        self.upper_bound = 1
        self.lower_bound = -1
        print("Min and Max Value of Action: {}".format(self.lower_bound,
                                                       self.upper_bound))

        critic_model = self.get_critic()
        critic2_model = self.get_critic()
        # actor_model.summary()
        # critic_model.summary()

        buffer = self.Buffer(self, buffer_dim, batch_size)

        # we create the target model for double learning (to prevent a moving target phenomenon)
        self.target_actor = self.get_actor()
        self.target_critic = self.get_critic()
        self.target_critic2 = self.get_critic()
        self.target_actor.trainable = False
        self.target_critic.trainable = False
        self.target_critic2.trainable = False
        aux_model = self.compose(self.actor_model, self.target_critic)

        ## TRAINING ##
        if self.load_weights:
            self.target_actor(layers.Input(shape=(self.num_states)))
            self.target_critic([
                layers.Input(shape=(self.num_states)),
                layers.Input(shape=(self.num_actions)),
            ])
            self.target_critic2([
                layers.Input(shape=(self.num_states)),
                layers.Input(shape=(self.num_actions)),
            ])
            critic_model = keras.models.load_model(self.weights_file_critic)
            critic2_model = keras.models.load_model(self.weights_file_critic2)

        # Making the weights equal initially
        self.target_actor_weights = self.actor_model.get_weights()
        self.target_critic_weights = critic_model.get_weights()
        self.target_critic2_weights = critic2_model.get_weights()
        self.target_actor.set_weights(self.target_actor_weights)
        self.target_critic.set_weights(self.target_critic_weights)
        self.target_critic2.set_weights(self.target_critic2_weights)

        critic_optimizer = tf.keras.optimizers.Adam(critic_lr)
        aux_optimizer = tf.keras.optimizers.Adam(aux_lr)

        critic_model.compile(loss="mse", optimizer=critic_optimizer)
        critic2_model.compile(loss="mse", optimizer=critic_optimizer)
        aux_model.compile(optimizer=aux_optimizer)

        # History of rewards per episode
        ep_reward_list = []
        # Average reward history of last few episodes
        avg_reward_list = []

        start_t = datetime.now()

        while i < total_iterations:
            prev_state = racer.reset()
            episodic_reward = 0
            mean_speed += prev_state[4]
            done = False

            while not (done):
                i = i + 1

                tf_prev_state = tf.expand_dims(
                    tf.convert_to_tensor(prev_state), 0)

                # our policy is always noisy
                action = self.policy(tf_prev_state)[0]
                # Get state and reward from the environment
                state, reward, done = self.step(action)
                # we distinguish between termination with failure (state = None) and succesfull termination on track completion
                # succesfull termination is stored as a normal tuple
                fail = done and len(state) < 5
                buffer.record((prev_state, action, reward, fail, state))
                if not (done):
                    mean_speed += state[4]

                episodic_reward += reward

                if buffer.buffer_counter > batch_size:
                    states, actions, rewards, dones, newstates = buffer.sample_batch(
                    )
                    newactions = self.policy_target(states)
                    minQ = tf.math.minimum(
                        self.target_critic([newstates, newactions]),
                        self.target_critic2([newstates, newactions]),
                    )
                    targetQ = rewards + (1 - dones) * gamma * (minQ)

                    loss1 = critic_model.train_on_batch([states, actions],
                                                        targetQ)
                    loss2 = critic2_model.train_on_batch([states, actions],
                                                         targetQ)

                    if i % target_freq == 0:
                        loss3 = aux_model.train_on_batch(states)
                        update_target(target_actor.variables,
                                      actor_model.variables, tau)
                        update_target(target_critic.variables,
                                      critic_model.variables, tau)
                        update_target(target_critic2.variables,
                                      critic2_model.variables, tau)

                prev_state = state
                if i % 100 == 0:
                    avg_reward_list.append(avg_reward)

            ep_reward_list.append(episodic_reward)

            # Mean of last 40 episodes
            avg_reward = np.mean(ep_reward_list[-40:])
            print(
                "Episode {}: Iterations {}, Avg. Reward = {}, Last reward = {}. Avg. speed = {}"
                .format(ep, i, avg_reward, episodic_reward, mean_speed / i))
            print("\n")

            if ep > 0 and ep % 40 == 0:
                print("## Evaluating policy ##")
                tracks.metrics_run(actor_model, 10)
            ep += 1

        if total_iterations > 0:
            if save_weights:
                critic_model.save(self.weights_file_critic)
                critic2_model.save(self.weights_file_critic2)
                self.actor_model.save(self.weights_file_actor)
            # Plotting Episodes versus Avg. Rewards
            plt.plot(avg_reward_list)
            plt.xlabel("Training steps x100")
            plt.ylabel("Avg. Episodic Reward")
            plt.ylim(-3.5, 7)
            plt.show(block=True)
            plt.pause(0.001)
            print("### TD3 Training ended ###")
            print("Trained over {} steps".format(i))

        end_t = datetime.now()
        print("Time elapsed: {}".format(end_t - start_t))

    def test(self):
        tracks.newrun([self.get_actor_model()])


if __name__ == "__main__":
    car = TD3()
    car.train(total_iterations=50)
    car.test()
