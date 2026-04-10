import os
import re
import numpy as np
import gymnasium as gym
from stable_baselines3 import DQN

# --- Настройки путей ---
PATH = os.path.dirname(os.path.abspath(__file__))

# Базовая модель (без ветра)
BASE_MODEL_PATH = os.path.join(PATH, '..', 'results', 'best_model.zip')
BASE_LOG_PATH = os.path.join(PATH, '..', 'rl-training(without_wind).log')

# Универсальная модель (с ветром)
WINDY_MODEL_PATH = os.path.join(PATH, '..', 'results_universal', 'best_model.zip')
WINDY_LOG_PATH = os.path.join(PATH, '..', 'rl-training-windy.log')

EPISODES = 100 
SUCCESS_THRESHOLD = 200.0

def parse_dqn_log(log_path):
    """
    Парсит текстовый лог Stable-Baselines3 для нахождения точки схождения.
    Ищет первое достижение награды > 200 от EvalCallback.
    """
    metrics = {
        "solved_step": None,
        "solved_time": None,
        "is_successful": False
    }
    
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        current_time_elapsed = 0
        
        for i, line in enumerate(lines):
            # Отслеживаем примерное время симуляции из таблиц rollout/time
            time_match = re.search(r"\|\s+time_elapsed\s+\|\s+(\d+)\s+\|", line)
            if time_match:
                current_time_elapsed = int(time_match.group(1))
                
            # Ищем логи EvalCallback
            eval_match = re.search(r"Eval num_timesteps=(\d+), episode_reward=([\d\.\-]+)", line)
            if eval_match:
                step = int(eval_match.group(1))
                reward = float(eval_match.group(2))
                
                # Если порог пробит впервые
                if reward >= SUCCESS_THRESHOLD and not metrics["is_successful"]:
                    metrics["is_successful"] = True
                    metrics["solved_step"] = step
                    metrics["solved_time"] = current_time_elapsed
                    
    except FileNotFoundError:
        print(f"[ОШИБКА] Лог {log_path} не найден.")
        return None
        
    return metrics

def evaluate_dqn_agent(env, model, episodes, noise_std=0.0):
    """
    Прогоняет DQN агента через N эпизодов.
    """
    rewards = []
    for _ in range(episodes):
        obs, _ = env.reset()
        episode_reward = 0
        done = False
        truncated = False
        
        while not (done or truncated):
            if noise_std > 0:
                obs = obs + np.random.normal(0, noise_std, size=obs.shape)
                
            # deterministic=True - используем жадную стратегию без случайностей
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, _ = env.step(int(action))
            episode_reward += reward
            
        rewards.append(episode_reward)
    return np.array(rewards)

def calculate_model_complexity(model):
    """
    Считает количество обучаемых весов (параметров) в PyTorch модели DQN.
    """
    # model.q_net содержит веса нейросети в SB3
    total_params = sum(p.numel() for p in model.q_net.parameters() if p.requires_grad)
    return total_params

def analyze_model(model_name, model_path, log_path):
    print(f"\n{'='*50}")
    print(f" АНАЛИЗ МОДЕЛИ: {model_name.upper()}")
    print(f"{'='*50}")

    # Загрузка
    if not os.path.exists(model_path):
        print(f"[ОШИБКА] Файл модели не найден: {model_path}")
        return
    model = DQN.load(model_path)
    
    # ---------------------------------------------------------
    # 1. Скорость обучения
    # ---------------------------------------------------------
    log_metrics = parse_dqn_log(log_path)
    if log_metrics:
        print("\n[Определение 2.1] Скорость обучения:")
        if log_metrics['is_successful']:
            print(f"  Порог фитнеса (>200) достигнут: Да")
            print(f"  Общее число взаимодействий со средой (N_steps): {log_metrics['solved_step']:,}")
            print(f"  Астрономическое время обучения до успеха (T_train): ~{log_metrics['solved_time']} сек")
        else:
            print("  Порог фитнеса (>200) достигнут: Нет (алгоритм не сошелся)")

    # ---------------------------------------------------------
    # 2. Сложность модели
    # ---------------------------------------------------------
    P = calculate_model_complexity(model)
    # Для полносвязных сетей (MLP) плотность всегда 100%
    print("\n[Определение 2.3] Сложность модели:")
    print(f"  Количество обучаемых параметров (P): {P:,}")
    print(f"  Плотность связей (D): 1.0000 (100.0%) - Полносвязная архитектура")

    # ---------------------------------------------------------
    # 3 & 4. Качество решения и Робастность
    # ---------------------------------------------------------
    print(f"\n[Запуск симуляций... Это займет около минуты]")
    
    # Среда 1: Штиль (Базовая)
    env_base = gym.make('LunarLander-v3')
    R_array = evaluate_dqn_agent(env_base, model, EPISODES)
    R_mean = np.mean(R_array)
    SD_R = np.std(R_array)
    
    # Среда 2: Зашумленные сенсоры (Базовая среда + шум в функции оценки)
    R_noisy_array = evaluate_dqn_agent(env_base, model, EPISODES, noise_std=0.1)
    R_noisy_mean = np.mean(R_noisy_array)
    
    # Среда 3: Модифицированная физика (Сильный ветер)
    env_windy = gym.make('LunarLander-v3', enable_wind=True, wind_power=15.0, turbulence_power=1.5)
    R_mod_array = evaluate_dqn_agent(env_windy, model, EPISODES)
    R_mod_mean = np.mean(R_mod_array)
    
    # Расчет снижения эффективности
    robustness_noise = ((R_mean - R_noisy_mean) / R_mean) * 100 if R_mean > 0 else 0
    robustness_env = ((R_mean - R_mod_mean) / R_mean) * 100 if R_mean > 0 else 0

    print("\n[Определение 2.2] Качество решения (Среда без ветра):")
    print(f"  Средняя награда (R_mean): {R_mean:.2f}")
    print(f"  Стабильность (SD_R): {SD_R:.2f}")

    print("\n[Определение 2.4] Робастность:")
    print(f"  Награда с зашумленными сенсорами (R_noisy): {R_noisy_mean:.2f}")
    print(f"  Снижение от шума (Robustness_noise): {robustness_noise:.2f}%")
    print(f"  Награда при сильном ветре (R_modified): {R_mod_mean:.2f}")
    print(f"  Снижение от изменения среды (s_env): {robustness_env:.2f}%")

if __name__ == '__main__':
    print("Начинаем сравнительный анализ моделей DQN...")
    
    # Анализируем базовую модель
    analyze_model("DQN Базовая (Штиль)", BASE_MODEL_PATH, BASE_LOG_PATH)
    
    # Анализируем универсальную модель
    analyze_model("DQN Универсальная (Штиль + Ветер)", WINDY_MODEL_PATH, WINDY_LOG_PATH)