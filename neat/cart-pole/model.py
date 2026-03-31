import gymnasium as gym
import neat
import os
import numpy as np
import pickle  # <-- Добавлен импорт для сохранения модели

# 3. Фитнес-функция (Оценка выживаемости)
def eval_genomes(genomes, config):
    """
    Эта функция запускается для КАЖДОГО поколения.
    Она берет каждого "генома" (нейросеть) из популяции, играет им в игру
    и присваивает ему очки (fitness) за то, как долго он продержался.
    """
    env = gym.make('CartPole-v1')
    
    for genome_id, genome in genomes:
        # Создаем нейросеть прямого распространения из генома
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        
        state, info = env.reset()
        fitness = 0.0
        done = False
        truncated = False
        
        while not (done or truncated):
            # Передаем 4 параметра среды в нейросеть. 
            # На выходе получаем 2 числа (уверенность в шаге влево и вправо)
            output = net.activate(state)
            
            # Выбираем действие с максимальной уверенностью (0 или 1)
            action = np.argmax(output)
            
            # Делаем шаг в среде
            state, reward, done, truncated, info = env.step(action)
            fitness += reward
            
        # Записываем результат генома (сколько шагов он удержал палку)
        genome.fitness = fitness
        
    env.close()

# 4. Основной цикл эволюции
def run_neat():
    
    # Загружаем настройки
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         "./neat/cart-pole/neat.cfg")

    # Создаем популяцию
    p = neat.Population(config)

    # Добавляем репортеры, чтобы видеть в консоли статистику по поколениям (Generations)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    print("Начинаем эволюцию...")
    # Запускаем эволюцию! Максимум 50 поколений.
    # Если кто-то наберет 500 очков (fitness_threshold в конфиге), эволюция остановится досрочно.
    winner = p.run(eval_genomes, 50)

    print(f'\nЭволюция завершена! Лучший геном: {winner}')

    model_filename = 'neat/cart-pole/results/best_agent.pkl'
    with open(model_filename, 'wb') as f:
        pickle.dump(winner, f)
    print(f'\n[INFO] Лучшая модель успешно сохранена в файл: {model_filename}')
    # ---------------------------------------------

    # 5. Демонстрация лучшего генома-победителя
    print("\nЗапускаем демонстрацию лучшего агента...")
    env_eval = gym.make('CartPole-v1', render_mode='human')
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    
    state, info = env_eval.reset()
    done = False
    truncated = False
    
    while not (done or truncated):
        output = winner_net.activate(state)
        action = np.argmax(output)
        state, reward, done, truncated, info = env_eval.step(action)
        
    env_eval.close()

if __name__ == '__main__':
    run_neat()