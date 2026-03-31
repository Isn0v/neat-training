import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

# 1. Создаем саму нейронную сеть
class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        # Для CartPole достаточно простой полносвязной сети из 3 слоев
        self.network = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, output_dim)
        )

    def forward(self, x):
        return self.network(x)

# 2. Создаем Агента, который будет принимать решения и обучаться
class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Гиперпараметры
        self.gamma = 0.99           # Скидка будущих наград (насколько важны будущие шаги)
        self.epsilon = 1.0          # Вероятность случайного действия (исследование)
        self.epsilon_min = 0.01     # Минимальная вероятность случайного действия
        self.epsilon_decay = 0.995  # Как быстро агент перестает исследовать
        self.learning_rate = 0.001
        self.batch_size = 64
        
        # Буфер памяти для обучения на прошлом опыте (Experience Replay)
        self.memory = deque(maxlen=10000)
        
        # Инициализация нейросети и оптимизатора
        self.model = QNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss() # Среднеквадратичная ошибка

    def act(self, state):
        # Epsilon-жадная стратегия: иногда делаем случайный шаг
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        
        # Иначе просим нейросеть предсказать лучшее действие
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        return q_values.argmax().item()

    def remember(self, state, action, reward, next_state, done):
        # Запоминаем шаг
        self.memory.append((state, action, reward, next_state, done))

    def train(self):
        # Если накопилось мало опыта — не обучаемся
        if len(self.memory) < self.batch_size:
            return

        # Берем случайную выборку (батч) из памяти
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Конвертируем в тензоры PyTorch
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones).unsqueeze(1)

        # Q-значения, которые сеть предсказывает для текущих состояний
        current_q = self.model(states).gather(1, actions)

        # Максимальные Q-значения для следующих состояний
        with torch.no_grad():
            max_next_q = self.model(next_states).max(1)[0].unsqueeze(1)
            # Уравнение Беллмана
            target_q = rewards + (1 - dones) * self.gamma * max_next_q

        # Считаем ошибку и обновляем веса нейросети
        loss = self.loss_fn(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Уменьшаем epsilon (агент становится более "уверенным")
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# 3. Основной цикл обучения
def main():
    # Создаем среду
    env = gym.make('CartPole-v1')
    
    state_dim = env.observation_space.shape[0] # 4 параметра (координаты, скорости, углы)
    action_dim = env.action_space.n            # 2 действия (влево, вправо)
    
    agent = DQNAgent(state_dim, action_dim)
    episodes = 500

    for episode in range(episodes):
        state, info = env.reset()
        total_reward = 0
        done = False
        truncated = False

        while not (done or truncated):
            # Агент выбирает действие
            action = agent.act(state)
            
            # Среда реагирует на действие
            next_state, reward, done, truncated, info = env.step(action)
            
            # Запоминаем результат и обучаемся
            agent.remember(state, action, reward, next_state, done)
            agent.train()
            
            state = next_state
            total_reward += reward

        print(f"Эпизод: {episode + 1}/{episodes} | Награда (Очки): {total_reward} | Epsilon: {agent.epsilon:.2f}")

        # Среда считается решенной, если агент удерживает шест 500 шагов
        if total_reward >= 500:
            print(f"Среда успешно решена за {episode + 1} эпизодов!")
            break

    env.close()
    
    # 4. Демонстрация обученного агента
    print("Запускаем демонстрацию...")
    env_eval = gym.make('CartPole-v1', render_mode='human')
    state, info = env_eval.reset()
    agent.epsilon = 0.0 # Отключаем случайные действия
    
    while True:
        action = agent.act(state)
        state, reward, done, truncated, info = env_eval.step(action)
        if done or truncated:
            break
    env_eval.close()

if __name__ == "__main__":
    main()