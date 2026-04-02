import os

import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
import re


PATH = os.path.dirname(os.path.abspath(__file__))


# --- Настройки ---
LOG_PATH = f'{PATH}/../rl-training.log'            # Файл логов обучения DQN
MODEL_WEIGHTS = f'{PATH}/../results/dqn_cartpole_winner.pth' # Сохраненные веса модели
EPISODES = 100                            # Количество эпизодов для валидации



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

def parse_rl_log(log_path):
    """
    Парсит лог обучения DQN для подсчета метрик Описания 2.1.
    """
    total_steps = 0
    episodes_count = 0
    with open(log_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    for line in lines:
        # Ищем строки вида "Награда (Очки): 17.0"
        match = re.search(r"Награда \(Очки\): ([\d\.]+)", line)
        if match:
            reward = float(match.group(1))
            # В CartPole 1 очко награды = 1 шаг взаимодействия со средой
            total_steps += int(reward)
            episodes_count += 1
            
    # Проверяем, была ли среда решена
    is_successful = any("Среда успешно решена" in line for line in lines)
    
    time_line = [line for line in lines if "Общее время обучения" in line]
    time_match = re.search(r"Общее время обучения: ([\d\.]+) секунд", time_line[0])

    
    total_time = float(time_match.group(1))
    return {
        "episodes": episodes_count,
        "total_steps": total_steps,
        "total_time": total_time,
        "is_successful": is_successful
    } 

def evaluate_rl_agent(env, model, episodes, noise_std=0.0):
    """
    Оценка качества модели на валидационной выборке.
    """
    rewards = []
    model.eval() # Переводим сеть в режим инференса (отключаем градиенты/dropout)
    
    for _ in range(episodes):
        obs, _ = env.reset()
        episode_reward = 0
        done = False
        truncated = False
        
        while not (done or truncated):
            # Добавление шума для метрики робастности
            if noise_std > 0:
                obs = obs + np.random.normal(0, noise_std, size=obs.shape)
                
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            
            with torch.no_grad():
                q_values = model(obs_tensor)
                action = q_values.argmax().item()
                
            obs, reward, done, truncated, _ = env.step(action)
            episode_reward += reward
            
        rewards.append(episode_reward)
    return np.array(rewards)

def main():
    print("=== РАСЧЕТ МЕТРИК ДЛЯ DQN (БАЗОВАЯ МОДЕЛЬ) ===")

    # ---------------------------------------------------------
    # Определение 2.1: Скорость обучения
    # ---------------------------------------------------------
    log_metrics = parse_rl_log(LOG_PATH)
    if log_metrics:
        print("\n[Определение 2.1] Скорость обучения:")
        print(f"  Количество эпизодов до схождения (E): {log_metrics['episodes']}")
        print(f"  Общее число взаимодействий со средой (N_steps): {log_metrics['total_steps']}")
        print(f"  Цель достигнута: {'Да' if log_metrics['is_successful'] else 'Нет'}")

    # Загрузка среды и инициализация модели
    env = gym.make('CartPole-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    model = QNetwork(state_dim, action_dim)
    try:
        model.load_state_dict(torch.load(MODEL_WEIGHTS))
        print("\n[INFO] Веса модели успешно загружены.")
    except FileNotFoundError:
        print(f"\n[ВНИМАНИЕ] Файл {MODEL_WEIGHTS} не найден! Запустите обучение DQN для создания файла.")
        return

    # ---------------------------------------------------------
    # Определение 2.2: Качество решения
    # ---------------------------------------------------------
    R_array = evaluate_rl_agent(env, model, EPISODES)
    R_mean = np.mean(R_array) 
    SD_R = np.std(R_array)    
    
    print("\n[Определение 2.2] Качество решения (на 100 эпизодах):")
    print(f"  Средняя награда (R_mean): {R_mean:.2f}")
    print(f"  Стабильность (SD_R): {SD_R:.2f}")

    # ---------------------------------------------------------
    # Определение 2.3: Сложность модели
    # ---------------------------------------------------------
    # Подсчет всех обучаемых параметров (веса + смещения)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("\n[Определение 2.3] Сложность модели:")
    print(f"  Количество параметров (P): {total_params}")
    print(f"  Плотность связей (D): 1.0000 (100.0% - полносвязная сеть)")

    # ---------------------------------------------------------
    # Определение 2.4: Робастность
    # ---------------------------------------------------------
    # 1. Шум
    R_noisy_array = evaluate_rl_agent(env, model, EPISODES, noise_std=0.05)
    R_noisy_mean = np.mean(R_noisy_array)
    robustness_noise = ((R_mean - R_noisy_mean) / R_mean) * 100 if R_mean > 0 else 0
    
    # 2. Изменение физики
    env_mod = gym.make('CartPole-v1')
    env_mod.unwrapped.masspole *= 2.0  # Утяжеляем маятник
    env_mod.unwrapped.length *= 1.5    # Удлиняем маятник
    
    R_mod_array = evaluate_rl_agent(env_mod, model, EPISODES)
    R_mod_mean = np.mean(R_mod_array)
    robustness_env = ((R_mean - R_mod_mean) / R_mean) * 100 if R_mean > 0 else 0

    print("\n[Определение 2.4] Робастность:")
    print(f"  Средняя награда с шумом (R_noisy): {R_noisy_mean:.2f}")
    print(f"  Снижение эффективности от шума: {robustness_noise:.2f}%")
    print(f"  Средняя награда в измененной среде (R_modified): {R_mod_mean:.2f}")
    print(f"  Снижение эффективности от изменения среды: {robustness_env:.2f}%")

if __name__ == '__main__':
    main()