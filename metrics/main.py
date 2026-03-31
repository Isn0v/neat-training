import gymnasium as gym
import neat
import pickle
import numpy as np
import os

# --- Настройки ---
CONFIG_PATH = 'neat/cart-pole/neat.cfg'
WINNER_PATH = 'neat/cart-pole/results/best_agent.pkl'
EPISODES = 1000

def evaluate_model(env, net, episodes, noise_std=0.0):
    """Прогон модели в среде с возможностью добавления шума"""
    rewards = []
    for _ in range(episodes):
        obs, _ = env.reset()
        episode_reward = 0
        done = False
        truncated = False
        
        while not (done or truncated):
            # Добавление гауссовского шума к наблюдениям (Метрика 2.4.1)
            if noise_std > 0:
                obs = obs + np.random.normal(0, noise_std, size=obs.shape)
            
            # Активация сети NEAT
            action_values = net.activate(obs)
            
            # Для CartPole: берем индекс узла с максимальной активацией
            action = np.argmax(action_values) 
            
            obs, reward, done, truncated, _ = env.step(action)
            episode_reward += reward
            
        rewards.append(episode_reward)
    
    # Возвращаем среднюю награду и стандартное отклонение
    return np.mean(rewards), np.std(rewards)

def main():
    # 1. Загрузка конфигурации и лучшего генома
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         CONFIG_PATH)

    with open(WINNER_PATH, 'rb') as f:
        winner_genome = pickle.load(f)

    # Компиляция фенотипа (рабочей нейросети)
    net = neat.nn.FeedForwardNetwork.create(winner_genome, config)

    print("="*40)
    print(" АНАЛИЗ МЕТРИК ДЛЯ СРЕДЫ CARTPOLE")
    print("="*40)

    # ---------------------------------------------------------
    # МЕТРИКА 1: Сложность модели (Определение 2.3)
    # ---------------------------------------------------------
    num_nodes = len(winner_genome.nodes)
    # Считаем только активные (включенные) связи
    num_connections = sum(1 for cg in winner_genome.connections.values() if cg.enabled)
    
    print("\n[1] Сложность модели:")
    print(f"  Количество скрытых и выходных узлов: {num_nodes}")
    print(f"  Количество активных связей (весов): {num_connections}")


    # ---------------------------------------------------------
    # МЕТРИКА 2: Качество решения (Определение 2.2)
    # ---------------------------------------------------------
    env = gym.make('CartPole-v1')
    mean_r, std_r = evaluate_model(env, net, EPISODES)
    
    print("\n[2] Качество решения (100 эпизодов):")
    print(f"  Средняя накопленная награда (R_mean): {mean_r:.2f}")
    print(f"  Стабильность поведения (SD_R): {std_r:.2f}")


    # ---------------------------------------------------------
    # МЕТРИКА 3: Робастность (Определение 2.4)
    # ---------------------------------------------------------
    
    # 3.1. Устойчивость к шуму (добавляем 5% гауссовского шума)
    mean_r_noise, _ = evaluate_model(env, net, EPISODES, noise_std=0.05)
    # Считаем процентное снижение по формуле 2.4.1
    robustness_noise = ((mean_r - mean_r_noise) / mean_r) * 100 if mean_r > 0 else 0
    
    print("\n[3] Робастность:")
    print(f"  Средняя награда с шумом (std=0.05): {mean_r_noise:.2f}")
    print(f"  Снижение эффективности от шума: {robustness_noise:.2f}%")

    # 3.2. Адаптивность к изменениям среды
    # Создаем модифицированную среду (например, делаем шест в 2 раза тяжелее и длиннее)
    env_mod = gym.make('CartPole-v1')
    env_mod.unwrapped.masspole = env_mod.unwrapped.masspole * 2.0
    env_mod.unwrapped.length = env_mod.unwrapped.length * 1.5
    
    mean_r_mod, _ = evaluate_model(env_mod, net, EPISODES)
    # Считаем процентное снижение по формуле 2.4.2
    robustness_env = ((mean_r - mean_r_mod) / mean_r) * 100 if mean_r > 0 else 0
    
    print(f"  Средняя награда в измененной среде (x2 масса, x1.5 длина): {mean_r_mod:.2f}")
    print(f"  Снижение эффективности от изменения физики: {robustness_env:.2f}%")
    print("="*40)

if __name__ == '__main__':
    main()