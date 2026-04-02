import gymnasium as gym
import neat
import numpy as np
import pickle
import multiprocessing
import os



# Оценка одного муравья (выполняется на разных ядрах)
# def eval_single_genome_old1(genome, config):
#     # Используем Ant-v5. render_mode здесь не нужен для скорости.
#     env = gym.make('Ant-v4', use_contact_forces=False)
#     net = neat.nn.FeedForwardNetwork.create(genome, config)
#     
#     episodes = 3
#     total_fitness = 0.0
#     
#     for _ in range(episodes):
#         state, info = env.reset()
#         done = False
#         truncated = False
#         episode_fitness = 0.0
#         
#         while not (done or truncated):
#             output = net.activate(state)
#             # В MuJoCo действия должны быть в диапазоне [-1, 1]
#             action = np.array(output) 
#             state, reward, done, truncated, info = env.step(action) # [cite: 17, 137]
#             episode_fitness += reward
#             
#         total_fitness += episode_fitness
#         
#     env.close()
#     return total_fitness / episodes # [cite: 138]


def eval_single_genome_old2(genome, config):
    # Убрали use_contact_forces=False, так как мы используем базовый Ant-v4,
    # где этот параметр не нужен по умолчанию, а бонус за выживание включен.
    env = gym.make('Ant-v4')
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    
    episodes = 3
    total_fitness = 0.0
    
    for _ in range(episodes):
        state, info = env.reset()
        done = False
        truncated = False
        episode_fitness = 0.0
        
        # --- Инициализация таймера лени ---
        # --- Инициализация перед циклом while ---
        max_x = info.get('x_position', 0.0)
        steps_without_progress = 0
        
        while not (done or truncated):
            output = net.activate(state)
            action = np.clip(np.array(output), env.action_space.low, env.action_space.high) 
            state, reward, done, truncated, info = env.step(action)
            
            # --- НОВЫЙ УМНЫЙ ТАЙМЕР ЛЕНИ ---
            current_x = info.get('x_position', max_x)
        
            
            # Проверяем, ушел ли он дальше своего исторического максимума (даже на миллиметр)
            if current_x > max_x + 0.05: 
                max_x = current_x
                steps_without_progress = 0 # Сбрасываем таймер: молодец, ползешь!
            else:
                steps_without_progress += 1
            
            # Даем ему 150 шагов (около 3-5 секунд), чтобы придумать, как сдвинуться дальше
            if steps_without_progress > 50:
                break
                
            episode_fitness += (reward - 1.0)
            
        total_fitness += episode_fitness
        
    env.close()
    return total_fitness / episodes


def eval_single_genome_old3(genome, config):
    env = gym.make('Ant-v4')
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    
    episodes = 3
    total_fitness = 0.0
    
    for _ in range(episodes):
        state, info = env.reset()
        done = False
        truncated = False
        
        # Запоминаем стартовую позицию
        max_x = info.get('x_position', 0.0) 
        steps_without_progress = 0  # Возвращаем счетчик
        energy_penalty = 0.0
        
        while not (done or truncated):
            output = net.activate(state)
            action = np.clip(np.array(output), env.action_space.low, env.action_space.high) 
            state, reward, done, truncated, info = env.step(action)
            
            energy_penalty += np.sum(np.square(action))
            # Отслеживаем, как далеко он продвинулся
            current_x = info.get('x_position', max_x)
            if current_x > max_x + 0.01:
                max_x = current_x
                steps_without_progress = 0
            else:
                steps_without_progress += 1
                
            if steps_without_progress > 50:
              break
                
        # Фитнес эпизода — это чистая дистанция в метрах!
        total_fitness += max_x - energy_penalty * 0.0005  # Применяем штраф за потребление энергии
        
    env.close()
    return total_fitness / episodes
  
def eval_single_genome(genome, config):
    env = gym.make('Ant-v4') 
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    
    episodes = 3
    total_fitness = 0.0
    
    for _ in range(episodes):
        state, info = env.reset()
        done = False
        truncated = False
        
        max_x = info.get('x_position', 0.0) 
        steps_without_progress = 0
        episode_fitness = 0.0
        
        while not (done or truncated):
            output = net.activate(state)
            action = np.clip(np.array(output), env.action_space.low, env.action_space.high) 
            state, reward, done, truncated, info = env.step(action)
            
            current_x = info.get('x_position', max_x)
            
            # Жесткий таймер: требуем продвижения на 5 см за 50 шагов
            if current_x > max_x + 0.05:
                max_x = current_x
                steps_without_progress = 0
            else:
                steps_without_progress += 1
                
            if steps_without_progress > 50:
                break
                
            # Собираем ВСТРОЕННУЮ награду среды. 
            # В ней уже заложена математика естественной, симметричной ходьбы!
            episode_fitness += reward
            
        total_fitness += episode_fitness
        
    env.close()
    return total_fitness / episodes

def run_neat_old():
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         "./neat/ant/neat.cfg")
    
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    print("Начинаем обучение Муравья (Многопоточность)...")
    
    # Используем все доступные ядра процессора [cite: 44, 140]
    pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), eval_single_genome)
    
    champions_list = []

    def evaluate_and_check(genomes, config):
        pe.evaluate(genomes, config)
        
        # Находим агентов, которые начали уверенно ходить (например, 1000+ очков)
        current_champions = [g for genome_id, g in genomes if g.fitness is not None and g.fitness >= 1000] # [cite: 141]
        
        # Если нашли 3 таких агента, можно считать этап пройденным [cite: 116]
        if len(current_champions) >= 3:
            nonlocal champions_list
            champions_list = current_champions
            raise StopIteration(f"Найдено {len(current_champions)} успешных моделей.")

    try:
        p.run(evaluate_and_check, 500) # Даем до 500 поколений [cite: 211]
    except StopIteration as e:
        print(f"\n[Эволюция прервана]: {e}")

    winner = champions_list[0] if champions_list else p.best_genome # [cite: 142, 143]

    if winner is not None:
        with open("best_ant_pilot.pkl", "wb") as f:
            pickle.dump(winner, f)
        print(f"\nЛучший результат: {winner.fitness:.1f}")


def run_neat(model_path="./neat/ant/", checkpoint_file=None, generations=500):
    # checkpoint_file = f"{model_path}/checkpoints/neat-checkpoint-148" if checkpoint_file is None else checkpoint_file
  
    # 1. Загрузка конфигурации
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         f"{model_path}/neat.cfg")
    
    # 2. Инициализация популяции: с нуля или из чекпоинта
    if checkpoint_file and os.path.exists(checkpoint_file):
        print(f"\n[Загрузка] Найден чекпоинт '{checkpoint_file}'. Восстанавливаем прогресс...")
        p = neat.Checkpointer.restore_checkpoint(checkpoint_file)
        p.config = config  # Обновляем конфигурацию на всякий случай (может быть полезно при изменении параметров)
    else:
        print("\n[Старт] Запуск обучения с первого поколения...")
        p = neat.Population(config)

    # 3. Добавление репортеров (вывод в консоль и статистика)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    # 4. ДОБАВЛЕНИЕ ЧЕКПОИНТЕРА
    # Сохраняем состояние каждые 25 поколений или каждые 10 минут (600 секунд)
    checkpointer = neat.Checkpointer(
        generation_interval=25, 
        time_interval_seconds=600, 
        filename_prefix=f"{model_path}/checkpoints/neat-checkpoint-"
    )
    p.add_reporter(checkpointer)

    print(f"Используем {multiprocessing.cpu_count()} ядер(а) процессора для оценки...")
    
    # 5. Инициализация многопоточного оценщика
    pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), eval_single_genome)
    
    champions_list = []

    # 6. Кастомная функция оценки с логикой ранней остановки
    def evaluate_and_check(genomes, config):
        # Оцениваем всех агентов параллельно
        pe.evaluate(genomes, config)
        
        # Находим агентов, которые начали уверенно ходить (например, 1000+ очков)
        current_champions = [g for genome_id, g in genomes if g.fitness is not None and g.fitness >= 1000]
        
        # Если нашли 3 таких агента, можно считать этап пройденным
        if len(current_champions) >= 3:
            nonlocal champions_list
            champions_list = current_champions
            # Прерываем цикл p.run() через генерацию исключения
            raise StopIteration(f"Найдено {len(current_champions)} успешных моделей.")

    # 7. Запуск эволюции
    try:
        p.run(evaluate_and_check, generations)
    except StopIteration as e:
        print(f"\n[Эволюция прервана]: {e}")
    except Exception as e:
        # Отлавливаем другие возможные ошибки (например, KeyboardInterrupt при ручной остановке Ctrl+C)
        print(f"\n[Остановка]: {e}")

    # 8. Сохранение абсолютного победителя
    # Если вышли по ранней остановке - берем первого из чемпионов. Иначе - лучшего в популяции.
    winner = champions_list[0] if champions_list else p.best_genome

    if winner is not None:
        with open("best_ant_pilot.pkl", "wb") as f:
            pickle.dump(winner, f)
        print(f"\nЛучший результат: {winner.fitness:.1f}")
        print("Лучший геном успешно сохранен в 'best_ant_pilot.pkl'")
    
    return winner


if __name__ == '__main__':
    multiprocessing.freeze_support() # [cite: 147, 150]
    run_neat()