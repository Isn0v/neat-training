import os

import gymnasium as gym
import neat
import pickle
import numpy as np
import re

# --- Настройки ---
PATH = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = f'{PATH}/../neat.cfg'
WINNER_PATH = f'{PATH}/../results/best_agent.pkl' 
LOG_PATH = f'{PATH}/../neat-training.log' # Путь к файлу логов
EPISODES = 100 
POPULATION_SIZE = 50 # Из neat.cfg

def parse_training_log(log_path):
    """
    Парсит лог обучения NEAT для получения метрик Скорости обучения (Опр. 2.1).
    """
    total_steps = 0
    total_time = 0.0
    generations = 0
    
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Разбиваем лог на блоки поколений
        gen_blocks = re.split(r"\*\*\*\*\*\* Running generation \d+ \*\*\*\*\*\*", content)[1:]
        
        for block in gen_blocks:
            generations += 1
            
            # Извлекаем среднюю награду популяции
            avg_fit_match = re.search(r"average fitness: ([\d\.]+)", block)
            if avg_fit_match:
                avg_fitness = float(avg_fit_match.group(1))
                # В CartPole 1 балл фитнеса = 1 фрейм симуляции
                # Общее число шагов в поколении = средний фитнес * размер популяции
                total_steps += avg_fitness * POPULATION_SIZE
                
            # Извлекаем время, затраченное на поколение
            time_match = re.search(r"Generation time: ([\d\.]+) sec", block)
            if time_match:
                total_time += float(time_match.group(1))
                
        is_successful = "meets fitness threshold" in content
                
        return {
            "generations": generations,
            "total_steps": int(total_steps),
            "total_time": total_time,
            "is_successful": is_successful
        }
    except FileNotFoundError:
        print(f"[ОШИБКА] Файл лога {log_path} не найден.")
        return None

def evaluate_agent(env, net, episodes, noise_std=0.0):
    """
    Прогоняет агента через N эпизодов и возвращает массив наград R_i.
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
                
            action_values = net.activate(obs)
            action = np.argmax(action_values) 
            
            obs, reward, done, truncated, _ = env.step(action)
            episode_reward += reward
            
        rewards.append(episode_reward)
    return np.array(rewards)

def main():
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         CONFIG_PATH)
    with open(WINNER_PATH, 'rb') as f:
        winner = pickle.load(f)
    net = neat.nn.FeedForwardNetwork.create(winner, config)
    
    print("=== РАСЧЕТ МЕТРИК ПО ФОРМУЛАМ ИЗ ВКР ===")

    # ---------------------------------------------------------
    # Определение 2.1: Скорость обучения (из логов)
    # ---------------------------------------------------------
    log_metrics = parse_training_log(LOG_PATH)
    if log_metrics:
        print("\n[Определение 2.1] Скорость обучения:")
        print(f"  Количество поколений до схождения: {log_metrics['generations']}")
        print(f"  Общее число взаимодействий со средой (N_steps): {log_metrics['total_steps']}")
        print(f"  Астрономическое время обучения (T_train): {log_metrics['total_time']:.3f} сек")
        print(f"  Порог фитнеса достигнут: {'Да' if log_metrics['is_successful'] else 'Нет'}")

    # ---------------------------------------------------------
    # Определение 2.2: Качество решения
    # ---------------------------------------------------------
    env = gym.make('CartPole-v1')
    R_array = evaluate_agent(env, net, EPISODES)
    
    R_mean = np.mean(R_array) 
    SD_R = np.std(R_array)    
    
    print("\n[Определение 2.2] Качество решения:")
    print(f"  Средняя награда (R_mean): {R_mean:.2f}")
    print(f"  Стабильность (SD_R): {SD_R:.2f}")

    # ---------------------------------------------------------
    # Определение 2.3: Сложность модели
    # ---------------------------------------------------------
    E_active = sum(1 for cg in winner.connections.values() if cg.enabled)
    num_nodes = len(winner.nodes)
    P = E_active + num_nodes
    
    num_inputs = len(config.genome_config.input_keys)
    num_outputs = len(config.genome_config.output_keys)
    E_max = num_inputs * num_outputs 
    D = E_active / E_max if E_max > 0 else 0
    
    print("\n[Определение 2.3] Сложность модели:")
    print(f"  Количество параметров (P): {P}")
    print(f"  Плотность связей (D): {D:.4f} ({D*100:.1f}%)")

    # ---------------------------------------------------------
    # Определение 2.4: Робастность
    # ---------------------------------------------------------
    R_noisy_array = evaluate_agent(env, net, EPISODES, noise_std=0.1)
    R_noisy_mean = np.mean(R_noisy_array)
    
    robustness_noise = ((R_mean - R_noisy_mean) / R_mean) * 100 if R_mean != 0 else 0
    
    env_modified = gym.make('CartPole-v1')
    env_modified.unwrapped.masspole *= 2.0
    env_modified.unwrapped.length *= 1.5
    
    R_mod_array = evaluate_agent(env_modified, net, EPISODES)
    R_mod_mean = np.mean(R_mod_array)
    
    robustness_env = ((R_mean - R_mod_mean) / R_mean) * 100 if R_mean != 0 else 0

    print("\n[Определение 2.4] Робастность:")
    print(f"  Средняя награда с шумом (R_noisy): {R_noisy_mean:.2f}")
    print(f"  Снижение от шума (Robustness_noise): {robustness_noise:.2f}%")
    print(f"  Средняя награда в измененной среде (R_modified): {R_mod_mean:.2f}")
    print(f"  Снижение от изменения среды (s_env): {robustness_env:.2f}%")

if __name__ == '__main__':
    main()