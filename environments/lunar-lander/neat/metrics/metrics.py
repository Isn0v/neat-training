import os
import re
import pickle
import numpy as np
import gymnasium as gym
import neat

# --- Настройки ---
PATH = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(PATH, '..', 'neat.cfg')
WINNER_PATH = os.path.join(PATH, '..', 'results', 'best_lunar_pilot.pkl')
LOG_PATH = os.path.join(PATH, '..', 'neat-training.log')
EPISODES = 100 
POPULATION_SIZE = 200

def parse_training_log(log_path):
    """
    Парсит лог обучения NEAT для получения метрик.
    """
    total_time = 0.0
    generations = 0
    
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Разбиваем лог на блоки поколений
        gen_blocks = re.split(r"\*\*\*\*\*\* Running generation \d+ \*\*\*\*\*\*", content)[1:]
        
        for block in gen_blocks:
            generations += 1
                
            # Извлекаем время, затраченное на поколение
            time_match = re.search(r"[Gg]eneration time: ([\d\.]+) sec", block)
            if time_match:
                total_time += float(time_match.group(1))
                
        is_successful = "meets fitness threshold" in content
                
        return {
            "generations": generations,
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
            # Добавление гауссовского шума к сенсорам агента (Опр. 2.4)
            if noise_std > 0:
                obs = obs + np.random.normal(0, noise_std, size=obs.shape)
                
            action_values = net.activate(obs)
            action = np.argmax(action_values) 
            
            obs, reward, done, truncated, _ = env.step(action)
            episode_reward += reward
            
        rewards.append(episode_reward)
    return np.array(rewards)

def main():
    # Загрузка конфигурации и лучшего генома
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         CONFIG_PATH)
                         
    try:
        with open(WINNER_PATH, 'rb') as f:
            winner = pickle.load(f)
    except FileNotFoundError:
        print(f"[ОШИБКА] Модель не найдена по пути: {WINNER_PATH}")
        return
        
    net = neat.nn.FeedForwardNetwork.create(winner, config)
    
    print("=== РАСЧЕТ МЕТРИК ПО ФОРМУЛАМ ИЗ ВКР ===")

   # ---------------------------------------------------------
    # Определение 2.1: Скорость обучения (из логов)
    # ---------------------------------------------------------
    log_metrics = parse_training_log(LOG_PATH)
    if log_metrics:
        # Вводные данные для аппроксимации
        episodes_per_genome = 5  # Задавали в model.py
        avg_steps_per_episode = 200 # Аналитическое допущение для LunarLander
        
        # Расчет
        total_episodes = log_metrics['generations'] * POPULATION_SIZE * episodes_per_genome
        approx_total_steps = total_episodes * avg_steps_per_episode
        
        print("\n[Определение 2.1] Скорость обучения:")
        print(f"  Количество поколений до схождения: {log_metrics['generations']}")
        print(f"  Астрономическое время обучения (T_train): {log_metrics['total_time']:.3f} сек")
        print(f"  Порог фитнеса (200) достигнут: {'Да' if log_metrics['is_successful'] else 'Нет'}")
        print(f"  Общее число симулированных эпизодов: {total_episodes}")
        print(f"  Аппроксимированное число шагов среды (N_steps): ~{approx_total_steps:,} " 
              f"(при S_avg={avg_steps_per_episode})")

    # ---------------------------------------------------------
    # Определение 2.2: Качество решения
    # ---------------------------------------------------------
    env = gym.make('LunarLander-v3')
    R_array = evaluate_agent(env, net, EPISODES)
    
    R_mean = np.mean(R_array) 
    SD_R = np.std(R_array)    
    
    print("\n[Определение 2.2] Качество решения:")
    print(f"  Средняя награда (R_mean): {R_mean:.2f} (Успехом считается > 200)")
    print(f"  Стабильность (SD_R): {SD_R:.2f}")

    # ---------------------------------------------------------
    # Определение 2.3: Сложность модели
    # ---------------------------------------------------------
    # Считаем только активные (enabled) связи, формирующие фенотип
    E_active = sum(1 for cg in winner.connections.values() if cg.enabled)
    
    num_inputs = len(config.genome_config.input_keys)
    # В neat-python winner.nodes содержит только скрытые (H) и выходные (O) узлы
    num_hidden_and_outputs = len(winner.nodes) 
    
    # Общее количество параметров (Узлы + Активные связи)
    P = E_active + num_hidden_and_outputs 
    
    # Правильный расчет E_max для направленного ациклического графа
    # 1. Связи от входов ко всем остальным узлам: I * N
    # 2. Максимально возможное количество связей между самими (H + O) узлами: N * (N - 1) / 2
    E_max = (num_inputs * num_hidden_and_outputs) + (num_hidden_and_outputs * (num_hidden_and_outputs - 1) / 2)
    
    D = E_active / E_max if E_max > 0 else 0
    
    print("\n[Определение 2.3] Сложность модели:")
    print(f"  Количество параметров графа (P): {P}")
    print(f"  Активных связей (E_active): {E_active}")
    print(f"  Теоретический максимум связей (E_max): {int(E_max)}")
    print(f"  Плотность связей (D): {D:.4f} ({D*100:.1f}%)")

    # ---------------------------------------------------------
    # Определение 2.4: Робастность
    # ---------------------------------------------------------
    # Тест 1: Зашумление сенсоров на 10%
    R_noisy_array = evaluate_agent(env, net, EPISODES, noise_std=0.01)
    R_noisy_mean = np.mean(R_noisy_array)
    
    robustness_noise = ((R_mean - R_noisy_mean) / R_mean) * 100 if R_mean != 0 else 0
    
    # Тест 2: Модификация физики (Включение сильного ветра и турбулентности)
    env_modified = gym.make('LunarLander-v3', enable_wind=True, wind_power=15.0, turbulence_power=1.5)
    
    R_mod_array = evaluate_agent(env_modified, net, EPISODES)
    R_mod_mean = np.mean(R_mod_array)
    
    robustness_env = ((R_mean - R_mod_mean) / R_mean) * 100 if R_mean != 0 else 0

    print("\n[Определение 2.4] Робастность:")
    print(f"  Средняя награда с зашумленными сенсорами (R_noisy): {R_noisy_mean:.2f}")
    print(f"  Снижение эффективности от шума (Robustness_noise): {robustness_noise:.2f}%")
    print(f"  Средняя награда при сильном ветре (R_modified): {R_mod_mean:.2f}")
    print(f"  Снижение эффективности от изменения среды (s_env): {robustness_env:.2f}%")

if __name__ == '__main__':
    main()