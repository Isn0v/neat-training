import gymnasium as gym
import neat
import numpy as np
import pickle


# 1. ФУНКЦИЯ ДЛЯ ОДНОГО ГЕНОМА (Её будут вызывать ядра процессора)
def eval_single_genome(genome, config):
    """Оценивает одного пилота 3 раза и возвращает средний балл"""
    env = gym.make('LunarLander-v3')
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    
    episodes = 3
    total_fitness = 0.0
    
    for _ in range(episodes):
        state, info = env.reset()
        done = False
        truncated = False
        episode_fitness = 0.0
        
        while not (done or truncated):
            output = net.activate(state)
            action = np.argmax(output) 
            state, reward, done, truncated, info = env.step(action)
            episode_fitness += reward
            
        total_fitness += episode_fitness
        
    env.close()
    
    # ВАЖНО: При многопоточности функция должна ВОЗВРАЩАТЬ число!
    return total_fitness / episodes 

def run_neat():
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         "./neat/lunar-lander/neat.cfg")
    
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    print("Начинаем тренировку пилотов (Ищем как минимум ТРОИХ победителей)...")
    
    # Создаем многопоточный оценщик
    pe = neat.ParallelEvaluator(4, eval_single_genome)  # 4 ядра процессора
    
    # Сюда мы сохраним нашу тройку (или больше) победителей
    champions_list = []

    # Создаем функцию-обертку прямо внутри run_neat
    def evaluate_and_check(genomes, config):
        # 1. Запускаем стандартную многопоточную оценку. 
        # После этой строки у каждого генома появится свой fitness
        pe.evaluate(genomes, config)
        
        # 2. Считаем, сколько геномов набрали 200 очков или больше
        current_champions = [g for genome_id, g in genomes if g.fitness is not None and g.fitness >= 200]
        
        # 3. Если их 3 или больше — прерываем эволюцию!
        if len(current_champions) >= 3:
            nonlocal champions_list
            champions_list = current_champions
            # Генерируем исключение, чтобы выпрыгнуть из цикла p.run()
            raise StopIteration(f"Успех! Найдено {len(current_champions)} пилотов-профи.")

    # Запускаем эволюцию, обернув её в блок try-except
    try:
        # Обрати внимание: передаем нашу новую функцию evaluate_and_check
        p.run(evaluate_and_check, 200)
    except StopIteration as e:
        print(f"\n[Эволюция досрочно остановлена]: {e}")

    # Подводим итоги
    if champions_list:
        # Сортируем победителей по количеству очков (от большего к меньшему)
        champions_list.sort(key=lambda x: x.fitness, reverse=True)
        winner = champions_list[0] # Берем самого лучшего из сдавших экзамен
        print(f"\nЛучший из команды победителей набрал: {winner.fitness:.1f} очков!")
    else:
        print("\nЗа 200 поколений не удалось найти 3-х идеальных пилотов.")
        winner = p.best_genome # Берем хотя бы того, кто есть

    # Сохраняем мозг лучшего
    with open("best_lunar_pilot.pkl", "wb") as f:
        pickle.dump(winner, f)

    # Демонстрация
    print("\nЗапускаем демонстрацию лучшего пилота из команды...")
    env_eval = gym.make('LunarLander-v3', render_mode='human')
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    
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
            
        print(f"Результат демонстрационной посадки: {total_reward:.1f} очков")
        
    env_eval.close()
    
if __name__ == "__main__":
    run_neat()