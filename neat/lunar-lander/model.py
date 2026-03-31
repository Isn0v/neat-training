import gymnasium as gym
import neat
import numpy as np
import pickle


def eval_genomes(genomes, config):
    # Создаем среду. Версия v3 актуальна для новых версий Gymnasium
    env = gym.make('LunarLander-v3')
    
    # СКОЛЬКО РАЗ ТЕСТИРУЕМ КАЖДОГО ПИЛОТА
    episodes_per_genome = 3 
    
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        total_fitness = 0.0
        
        for _ in range(episodes_per_genome):
          state, info = env.reset()
          episode_fitness = 0.0
          done = False
          truncated = False
          
          while not (done or truncated):
              # Нейросеть выдает 4 числа
              output = net.activate(state)
              
              # Выбираем двигатель, у которого уверенность (число) больше всего
              action = np.argmax(output) 
              state, reward, done, truncated, info = env.step(action)
              episode_fitness += reward 
              
          total_fitness += episode_fitness
          
        genome.fitness = total_fitness / episodes_per_genome  # Средний результат за 3 попытки
        
    env.close()

def run_neat():
    
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         "./neat/lunar-lander/neat.cfg")

    p = neat.Population(config)

    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    print("Начинаем тренировку пилотов...")
    # Ставим максимум 200 поколений. Но обычно решает за 50-100.
    winner = p.run(eval_genomes, 200)

    print(f'\nОбучение завершено! Лучший геном:\n{winner}')

    # Сохраняем мозг лучшего пилота в файл, чтобы не обучать заново каждый раз
    with open("best_lunar_pilot.pkl", "wb") as f:
        pickle.dump(winner, f)

    # Демонстрация
    print("\nЗапускаем демонстрацию лучшего пилота...")
    env_eval = gym.make('LunarLander-v3', render_mode='human')
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    
    # Покажем 3 посадки подряд
    for _ in range(3):
        state, info = env_eval.reset()
        done = False
        truncated = False
        total_reward = 0
        
        while not (done or truncated):
            output = winner_net.activate(state)
            action = np.argmax(output)
            state, reward, done, truncated, info = env_eval.step(action)
            total_reward += reward
            
        print(f"Результат посадки: {total_reward:.1f} очков")
        
    env_eval.close()

if __name__ == '__main__':
    run_neat()
