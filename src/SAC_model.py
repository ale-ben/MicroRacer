from datetime import datetime
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import tracks

from Base_model import Base_model


class SAC(Base_model):
    def __init__(self, load_weights=True, model_name="SAC", weight_path=None):
        """Constructor for the model

        Args:
            load_weights (bool, optional): If set to true loads weights for the model a weight file. Defaults to True.
            model_name (str, optional): Name of the model (will be used to find weight paths). Defaults to "sac".
            weight_path (str, optional): Base folder for weights. Defaults to "../weights".
        """
        super().__init__(model_name)

        self.racer = tracks.Racer()

        ########################################
        ###### HYPERPARAMETERS #################

        # Discount factor
        # gamma = tf.constant(0.99, dtype=tf.float32)
        self.gamma = 0.99
        # Target network parameter update factor, for double DQN
        self.tau = 0.005
        # Learning rate for actor-critic models
        self.learning_rate = 0.001

        self.num_states = 5  # we reduce the state dim through observation (see below)
        self.num_actions = 2  # acceleration and steering

        if weight_path is None:
            weight_path = f"./{model_name}/{model_name}_weights"
        self.weights_file_actor = f"{weight_path}/{model_name}_actor_model_car"
        self.weights_file_critic = f"{weight_path}/{model_name}_critic_model_car"
        self.weights_file_critic2 = f"{weight_path}/{model_name}_critic2_model_car"

        self.upper_bound = 1
        self.lower_bound = -1

        self.buffer_dim = 50000
        self.batch_size = 64

        print(
            "State Space dim: {}, Action Space dim: {}".format(
                self.num_states, self.num_actions
            )
        )

        print(
            "Min and Max Value of Action: {}".format(self.lower_bound, self.upper_bound)
        )

        # Adaptive Entropy to maximize exploration
        self.target_entropy = -tf.constant(self.num_actions, dtype=tf.float32)
        self.log_alpha = tf.Variable(0.0, dtype=tf.float32)
        self.alpha = tfp.util.DeferredTensor(self.log_alpha, tf.exp)

        # creating models
        self.actor_model = self.Get_actor(self)

        self.load_weights = load_weights
        if load_weights:
            self.actor_model = keras.models.load_model(self.weights_file_actor)

    # The actor choose the move, given the state
    class Get_actor(tf.keras.Model):
        def __init__(self, model):
            super().__init__()
            self.d1 = layers.Dense(64, activation="relu")
            self.d2 = layers.Dense(64, activation="relu")
            self.m = layers.Dense(model.num_actions)
            self.s = layers.Dense(model.num_actions)
            self.model = model

        def call(self, inputs):
            out = self.d1(inputs)
            out = self.d2(out)
            mu = self.m(out)
            log_sigma = self.s(out)
            sigma = tf.exp(log_sigma)

            dist = tfp.distributions.Normal(mu, sigma)
            # action = dist.sample()
            action = mu + sigma * tfp.distributions.Normal(0, 1).sample(
                self.model.num_actions
            )
            valid_action = tf.tanh(action)

            log_p = dist.log_prob(action)
            # correct log_p after the tanh squashing on action
            log_p = log_p - tf.reduce_sum(
                tf.math.log(1 - valid_action**2 + 1e-16), axis=1, keepdims=True
            )

            if len(log_p.shape) > 1:
                log_p = tf.reduce_sum(log_p, 1)
            else:
                log_p = tf.reduce_sum(log_p)
            log_p = tf.reshape(log_p, (-1, 1))

            eval_action = tf.tanh(mu)

            return eval_action, valid_action, log_p

        @property
        def trainable_variables(self):
            return (
                self.d1.trainable_variables
                + self.d2.trainable_variables
                + self.m.trainable_variables
                + self.s.trainable_variables
            )

    # the critic compute the q-value, given the state and the action
    class Get_critic(tf.keras.Model):
        def __init__(self):
            super().__init__()
            self.d1 = layers.Dense(64, activation="relu")
            self.d2 = layers.Dense(64, activation="relu")
            self.o = layers.Dense(1)

        def call(self, inputs):
            state, action = inputs
            state_action = tf.concat([state, action], axis=1)
            out = self.d1(state_action)
            out = self.d2(out)
            q = self.o(out)
            return q

        @property
        def trainable_variables(self):
            return (
                self.d1.trainable_variables
                + self.d2.trainable_variables
                + self.o.trainable_variables
            )

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
            self.state_buffer = np.zeros((self.buffer_capacity, model.num_states))
            self.action_buffer = np.zeros((self.buffer_capacity, model.num_actions))
            self.reward_buffer = np.zeros((self.buffer_capacity, 1))
            self.done_buffer = np.zeros((self.buffer_capacity, 1))
            self.next_state_buffer = np.zeros((self.buffer_capacity, model.num_states))

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
    def update_target(self, target_weights, weights, tau):
        for (a, b) in zip(target_weights, weights):
            a.assign(b * tau + a * (1 - tau))

    def update_weights(self, target_weights, weights, tau):
        return target_weights * (1 - tau) + weights * tau

    # We introduce a probability of doing n empty actions to separate the environment time-step from the agent
    def step(self, action):
        n = 1
        t = np.random.randint(0, n)
        action = tf.squeeze(action)
        state, reward, done = self.racer.step(action)
        for i in range(t):
            if not done:
                state, t_r, done = self.racer.step([0, 0])
                # state ,t_r, done =racer.step(action)
                reward += t_r
        return (state, reward, done)

    @tf.function
    def update_critics(self, states, actions, rewards, dones, newstates):
        entropy_scale = tf.convert_to_tensor(self.alpha)
        _, new_policy_actions, log_probs = self.actor_model(newstates)
        q1_t = self.target_critic([newstates, new_policy_actions])
        q2_t = self.target_critic2([newstates, new_policy_actions])
        tcritic_v = tf.reduce_min([q1_t, q2_t], axis=0)
        newvalue = tcritic_v - entropy_scale * log_probs
        q_hat = tf.stop_gradient(rewards + self.gamma * newvalue * (1 - dones))
        with tf.GradientTape(persistent=True) as tape1:
            q1 = self.critic_model([states, actions])
            q2 = self.critic2_model([states, actions])
            loss_c1 = tf.reduce_mean((q1 - q_hat) ** 2)
            loss_c2 = tf.reduce_mean((q2 - q_hat) ** 2)
        critic1_gradient = tape1.gradient(
            loss_c1, self.critic_model.trainable_variables
        )
        critic2_gradient = tape1.gradient(
            loss_c2, self.critic2_model.trainable_variables
        )
        self.critic_model.optimizer.apply_gradients(
            zip(critic1_gradient, self.critic_model.trainable_variables)
        )
        self.critic2_model.optimizer.apply_gradients(
            zip(critic2_gradient, self.critic2_model.trainable_variables)
        )

    @tf.function
    def update_actor(self, states):
        entropy_scale = tf.convert_to_tensor(self.alpha)
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(self.actor_model.trainable_variables)
            _, new_policy_actions, log_probs = self.actor_model(states)
            q1_n = self.critic_model([states, new_policy_actions])
            q2_n = self.critic2_model([states, new_policy_actions])
            critic_v = tf.reduce_min([q1_n, q2_n], axis=0)
            actor_loss = critic_v - entropy_scale * log_probs
            actor_loss = -tf.reduce_mean(actor_loss)
        actor_gradient = tape.gradient(actor_loss, self.actor_model.trainable_variables)
        self.actor_model.optimizer.apply_gradients(
            zip(actor_gradient, self.actor_model.trainable_variables)
        )

    @tf.function
    def update_entropy(self, states):
        _, _, log_probs = self.actor_model(states)
        with tf.GradientTape() as tape:
            alpha_loss = tf.reduce_mean(
                -self.alpha * tf.stop_gradient(log_probs + self.target_entropy)
            )
        alpha_grad = tape.gradient(alpha_loss, [self.log_alpha])
        self.alpha_optimizer.apply_gradients(zip(alpha_grad, [self.log_alpha]))

    def train(
        self,
        total_iterations=50000,
        save_weights=True,
        save_name=None,
        plot_results=True,
    ):
        """Train the model and (optionally) save the weights.
            To train an empty model set load_weights to false in model constructor.

        Args:
            total_iterations (int, optional): Number of iteration to run for training. Defaults to 600.
            save_weights (bool, optional): If true save the weights to the specified path. WARNING: note that this overwrites previous weight files. Defaults to True.
            save_name (_type_, optional): If set specifies a different name to save the weights, otherwise the model will use model_name. Defaults to None.
            plot_results (bool, optional): If true plots a training graph at the end of training. NOTE: Set to false if running training on a CLI only machine. Defaults to True.
        """

        self.critic_model = self.Get_critic()
        self.critic2_model = self.Get_critic()

        # we create the target model for double learning (to prevent a moving target phenomenon)
        self.target_critic = self.Get_critic()
        self.target_critic2 = self.Get_critic()
        self.target_critic.trainable = False
        self.target_critic2.trainable = False

        ## TRAINING ##
        if self.load_weights:
            self.target_critic(
                [
                    layers.Input(shape=(self.num_states)),
                    layers.Input(shape=(self.num_actions)),
                ]
            )
            self.target_critic2(
                [
                    layers.Input(shape=(self.num_states)),
                    layers.Input(shape=(self.num_actions)),
                ]
            )
            self.critic_model = keras.models.load_model(self.weights_file_critic)
            self.critic2_model = keras.models.load_model(self.weights_file_critic2)

        # Making the weights equal initially
        target_critic_weights = self.critic_model.get_weights()
        target_critic2_weights = self.critic2_model.get_weights()
        self.target_critic.set_weights(target_critic_weights)
        self.target_critic2.set_weights(target_critic2_weights)

        actor_optimizer = tf.keras.optimizers.Adam(self.learning_rate)
        critic_optimizer = tf.keras.optimizers.Adam(self.learning_rate)
        critic2_optimizer = tf.keras.optimizers.Adam(self.learning_rate)
        self.alpha_optimizer = tf.keras.optimizers.Adam(self.learning_rate)

        self.critic_model.compile(optimizer=critic_optimizer)
        self.critic2_model.compile(optimizer=critic2_optimizer)
        self.actor_model.compile(optimizer=actor_optimizer)

        buffer = self.Buffer(self, self.buffer_dim, self.batch_size)

        # History of rewards per episode
        ep_reward_list = []
        # Average reward history of last few episodes
        avg_reward_list = []

        #### TRAINING ####
        start_t = datetime.now()

        i = 0
        mean_speed = 0
        ep = 0
        avg_reward = 0
        while i < total_iterations:

            prev_state = self.racer.reset()
            episodic_reward = 0
            mean_speed += prev_state[4]
            done = False

            while not (done):
                i = i + 1

                tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)
                _, action, _ = self.actor_model(tf_prev_state)
                state, reward, done = self.step(action)

                # we distinguish between termination with failure (state = None) and succesfull termination on track completion
                # succesfull termination is stored as a normal tuple
                fail = done and len(state) < 5
                buffer.record((prev_state, action, reward, fail, state))
                if not (done):
                    mean_speed += state[4]

                episodic_reward += reward

                if buffer.buffer_counter > self.batch_size:
                    states, actions, rewards, dones, newstates = buffer.sample_batch()
                    states = tf.stack(tf.convert_to_tensor(states, dtype=tf.float32))
                    actions = tf.stack(tf.convert_to_tensor(actions, dtype=tf.float32))
                    rewards = tf.stack(tf.convert_to_tensor(rewards, dtype=tf.float32))
                    dones = tf.stack(tf.convert_to_tensor(dones, dtype=tf.float32))
                    newstates = tf.stack(
                        tf.convert_to_tensor(newstates, dtype=tf.float32)
                    )

                    self.update_critics(states, actions, rewards, dones, newstates)
                    self.update_actor(states)
                    self.update_entropy(states)
                    self.update_target(
                        self.target_critic.variables,
                        self.critic_model.variables,
                        self.tau,
                    )
                    self.update_target(
                        self.target_critic2.variables,
                        self.critic2_model.variables,
                        self.tau,
                    )

                prev_state = state

                if i % 100 == 0:
                    avg_reward_list.append(avg_reward)

            ep_reward_list.append(episodic_reward)

            # Mean of last 40 episodes
            avg_reward = np.mean(ep_reward_list[-40:])
            print(
                "Episode {}: Iterations {}, Avg. Reward = {}, Last reward = {}. Avg. speed = {}".format(
                    ep, i, avg_reward, episodic_reward, mean_speed / i
                )
            )
            print("\n")

            if ep > 0 and ep % 40 == 0:
                print("## Evaluating policy ##")
                tracks.metrics_run(self.actor_model, 10)
            ep += 1

        if total_iterations > 0:
            if save_weights:
                if save_name is not None:
                    self.weights_file_actor = (
                        f"{weight_path}/{save_name}_actor_model_car"
                    )
                    self.weights_file_critic = (
                        f"{weight_path}/{save_name}_critic_model_car"
                    )
                    self.weights_file_critic2 = (
                        f"{weight_path}/{save_name}_critic2_model_car"
                    )
                self.critic_model.save(self.weights_file_critic)
                self.critic2_model.save(self.weights_file_critic2)
                self.actor_model.save(self.weights_file_actor)

            if plot_results:
                # Plotting Episodes versus Avg. Rewards
                plt.plot(avg_reward_list)
                plt.xlabel("Training steps x100")
                plt.ylabel("Avg. Episodic Reward")
                plt.ylim(-3.5, 7)
                plt.show(block=True)
                plt.pause(10)

            print("### SAC Training ended ###")
            print("Trained over {} steps".format(i))

        end_t = datetime.now()
        print("Training completed.\nTime elapsed: {}".format(end_t - start_t))

    def test(self):
        tracks.newrun([self.get_actor_model()])


if __name__ == "__main__":
    car = SAC()
    car.train(total_iterations=5)
