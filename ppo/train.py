@dataclass
class PPOConfig:
    batch_size: int = 64
    epochs: int = 10
    clip_ratio: float = 0.2
    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 1e-3


def train_step(agent: PPOAgent, buffer: AdvantageBuffer, config: PPOConfig) -> Tuple[float, float]:
    states, actions, old_log_probs, returns, advantages = buffer.get()

    # Normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    actor_losses, critic_losses = [], []

    # Optimize in multiple epochs
    for _ in range(config.epochs):
        # Create mini-batches
        indices = np.random.permutation(len(states))
        for start in range(0, len(states), config.batch_size):
            end = start + config.batch_size
            mini_batch_indices = indices[start:end]

            mini_batch_states = tf.convert_to_tensor(states[mini_batch_indices], dtype=tf.float32)
            mini_batch_actions = tf.convert_to_tensor(actions[mini_batch_indices], dtype=tf.int32)
            mini_batch_old_log_probs = tf.convert_to_tensor(old_log_probs[mini_batch_indices], dtype=tf.float32)
            mini_batch_returns = tf.convert_to_tensor(returns[mini_batch_indices], dtype=tf.float32)
            mini_batch_advantages = tf.convert_to_tensor(advantages[mini_batch_indices], dtype=tf.float32)

            with tf.GradientTape() as actor_tape, tf.GradientTape() as critic_tape:
                # Actor loss
                logits = agent.actor(mini_batch_states)
                new_log_probs = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=mini_batch_actions, logits=logits
                )
                ratio = tf.exp(new_log_probs - mini_batch_old_log_probs)
                surrogate1 = ratio * mini_batch_advantages
                surrogate2 = (
                    tf.clip_by_value(ratio, 1.0 - config.clip_ratio, 1.0 + config.clip_ratio) * mini_batch_advantages
                )
                actor_loss = -tf.reduce_mean(tf.minimum(surrogate1, surrogate2))

                # Critic loss
                values = tf.squeeze(agent.critic(mini_batch_states))
                critic_loss = tf.reduce_mean(tf.square(mini_batch_returns - values))

            # Update actor
            actor_grads = actor_tape.gradient(actor_loss, agent.actor.trainable_variables)
            agent.actor_optimizer.apply_gradients(zip(actor_grads, agent.actor.trainable_variables))

            # Update critic
            critic_grads = critic_tape.gradient(critic_loss, agent.critic.trainable_variables)
            agent.critic_optimizer.apply_gradients(zip(critic_grads, agent.critic.trainable_variables))

            actor_losses.append(actor_loss.numpy())
            critic_losses.append(critic_loss.numpy())

    buffer.clear()
    return np.mean(actor_losses), np.mean(critic_losses)


def train_ppo(config: PPOConfig, episodes: int, storage_path: Path) -> str:
    env = EncodedTwentyFortyEight()
    actor_model = build_attention_model(state_size=(4, 4, 31), action_size=4)
    critic_model = build_attention_model(state_size=(4, 4, 31), action_size=1)

    agent = PPOAgent(actor_model, critic_model)
    advantage_buffer = AdvantageBuffer(capacity=1000)  # This needs to be implemented

    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        while not done:
            action, action_prob = agent.sample_action(state)
            next_state, reward, done = env.step(action)
            advantage_buffer.add(state, action, reward, next_state, done, action_prob)
            episode_reward += reward

            if len(advantage_buffer) >= config.batch_size:
                actor_loss, critic_loss = train_step(agent, advantage_buffer, config)
                print(f"Episode {episode}, Actor Loss: {actor_loss:.4f}, Critic Loss: {critic_loss:.4f}")

            state = next_state
        print(f"Episode {episode} finished with reward {episode_reward}")

    model_name = "ppo_model.keras"
    agent.actor.save(storage_path / model_name)
    return model_name
