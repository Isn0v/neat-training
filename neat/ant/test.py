import gymnasium as gym
import neat
import pickle
import numpy as np
import matplotlib.pyplot as plt

# 1. Настройки путей к файлам
CONFIG_PATH = 'neat/ant/neat.cfg' # Укажите имя вашего файла конфигурации NEAT
WINNER_PATH = 'neat/ant/results/ant_pilot(2 arms inactive).pkl'         # Укажите имя файла с сохраненным геномом

def run_and_plot_winner(config_path, winner_path):
    # 2. Загрузка конфигурации и лучшего генома
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    with open(winner_path, 'rb') as f:
        winner = pickle.load(f)

    # 3. Создание нейросети из генома
    net = neat.nn.FeedForwardNetwork.create(winner, config)

    # 4. Инициализация среды с визуализацией
    # render_mode='human' откроет окно симуляции
    env = gym.make('Ant-v4', render_mode='human', )

    obs, info = env.reset()
    done = False
    
    # Списки для сбора данных для графиков
    step_rewards = []
    cumulative_rewards = []
    current_cumulative = 0.0

    print("Запускаем симуляцию...")

    # 5. Основной цикл симуляции
    while not done:
        # NEAT ожидает на вход плоский список, obs обычно numpy array
        action_values = net.activate(obs)
        
        # Преобразуем выход сети в numpy array и обрезаем под лимиты среды (обычно от -1 до 1)
        action = np.array(action_values)
        action = np.clip(action, env.action_space.low, env.action_space.high)

        # Делаем шаг в среде
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Собираем статистику
        step_rewards.append(reward)
        current_cumulative += reward
        cumulative_rewards.append(current_cumulative)

        # Среда завершается, если робот упал (terminated) или вышло время (truncated)
        done = terminated or truncated

    env.close()
    print(f"Симуляция завершена! Итоговый фитнес: {current_cumulative:.2f}")

    # 6. Построение графиков
    plot_results(step_rewards, cumulative_rewards)

def plot_results(step_rewards, cumulative_rewards):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # График 1: Награда за каждый шаг (показывает стабильность походки)
    ax1.plot(step_rewards, color='blue', alpha=0.7)
    ax1.set_title('Награда на каждом шаге (Step Reward)')
    ax1.set_xlabel('Шаг')
    ax1.set_ylabel('Награда')
    ax1.grid(True)

    # График 2: Накопленная награда (показывает общий рост фитнеса)
    ax2.plot(cumulative_rewards, color='green', linewidth=2)
    ax2.set_title('Накопленная награда (Cumulative Fitness)')
    ax2.set_xlabel('Шаг')
    ax2.set_ylabel('Суммарный фитнес')
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    run_and_plot_winner(CONFIG_PATH, WINNER_PATH)