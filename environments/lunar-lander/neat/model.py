import os
import multiprocessing
import random
import gymnasium as gym
import neat
import numpy as np
import pickle

EPISODES_PER_GENOME = 5  # Количество общих испытаний для текущего поколения
EPISODES_EVOLUTION = 1000 # Максимальное количество поколений
PATH = os.path.dirname(os.path.abspath(__file__))

# ======================================================================
# 1. ФУНКЦИЯ ОЦЕНКИ (Принимает сгенерированные сиды)
# ======================================================================
def eval_genome_with_seeds(genome, config, current_seeds):
    """
    Оценивает одного агента, используя строго заданный список сидов.
    Эта функция выполняется изолированно в отдельном ядре процессора.
    """
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    
    total_fitness = 0.0
    
    for i, seed in enumerate(current_seeds):
        # Устанавливаем конкретный сид для этого испытания
        if i % 2 == 0:
            env = gym.make('LunarLander-v3', enable_wind=True, wind_power=15.0, turbulence_power=1.5)
        else:
            env = gym.make('LunarLander-v3') # Обычные условия
            
        state, info = env.reset(seed=seed)
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
    return total_fitness / len(current_seeds)

# ======================================================================
# 2. КАСТОМНЫЙ ПАРАЛЛЕЛЬНЫЙ ЭВАЛЮАТОР
# ======================================================================
class GenerationSeedEvaluator:
    """
    Свой класс для многопоточности. Генерирует сиды 1 раз за поколение
    и раздает их всем потокам.
    """
    def __init__(self, num_workers, episodes):
        self.num_workers = num_workers
        self.episodes = episodes
        self.pool = multiprocessing.Pool(num_workers)

    def evaluate(self, genomes, config):
        # 1. Генерируем 5 случайных сидов для ВСЕГО текущего поколения
        current_seeds = [random.randint(0, 100000) for _ in range(self.episodes)]
        
        # 2. Подготавливаем задачи для ядер процессора (добавляем сиды к геному)
        jobs = []
        for genome_id, genome in genomes:
            jobs.append((genome, config, current_seeds))

        # 3. Раскидываем задачи по ядрам (используем starmap для распаковки аргументов)
        fitnesses = self.pool.starmap(eval_genome_with_seeds, jobs)

        # 4. Сохраняем результаты обратно в геномы
        for (genome_id, genome), fitness in zip(genomes, fitnesses):
            genome.fitness = fitness

    def close(self):
        self.pool.close()
        self.pool.join()

# ======================================================================
# 3. ОСНОВНОЙ ЦИКЛ NEAT
# ======================================================================
def run_neat(checkpoint_file=None):
    config_path = os.path.join(PATH, "neat.cfg")
    config = neat.Config(
        neat.DefaultGenome, 
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet, 
        neat.DefaultStagnation,
        config_path
    )
    
    # 2. Инициализация популяции: с нуля или из чекпоинта
    if checkpoint_file and os.path.exists(checkpoint_file):
        print(f"\n[Загрузка] Найден чекпоинт '{checkpoint_file}'. Восстанавливаем прогресс...")
        p = neat.Checkpointer.restore_checkpoint(checkpoint_file)
        p.config = config  # Обновляем конфигурацию на всякий случай (может быть полезно при изменении параметров)
    else:
        print("\n[Старт] Запуск обучения с первого поколения...")
        p = neat.Population(config)
    
    p.add_reporter(neat.StdOutReporter(True))
    p.add_reporter(neat.StatisticsReporter())
    
    num_cores = multiprocessing.cpu_count()
    print(f"\n[INFO] Запуск честной оценки на {num_cores} ядрах (Подход 2)...")
    
    # Используем наш кастомный эвалюатор
    evaluator = GenerationSeedEvaluator(num_cores, EPISODES_PER_GENOME)
    
    checkpointer = neat.Checkpointer(
        generation_interval=50, 
        time_interval_seconds=600, 
        filename_prefix=f"{PATH}/checkpoints/neat-checkpoint-"
    )
    p.add_reporter(checkpointer)
    
    try:
        winner = p.run(evaluator.evaluate, EPISODES_EVOLUTION)
    except StopIteration:
        print("\n[INFO] Обучение остановлено (достигнут fitness_threshold).")
        winner = p.best_genome
    finally:
        evaluator.close() # Важно корректно закрыть пул процессов

    print(f"\nЛучший геном:\n{winner}")
    
    # Безопасное сохранение модели (с созданием папки)
    results_dir = os.path.join(PATH, 'results')
    os.makedirs(results_dir, exist_ok=True)
    winner_model_path = os.path.join(results_dir, "best_lunar_pilot.pkl")
    with open(winner_model_path, "wb") as f:
        pickle.dump(winner, f)
    print(f"\n[INFO] Лучшая модель сохранена в {winner_model_path}")

    # ======================================================================
    # 4. ДЕМОНСТРАЦИЯ ЛУЧШЕГО АГЕНТА (с абсолютно случайными условиями)
    # ======================================================================
    print("\nЗапускаем демонстрацию лучшего пилота (покажем 3 посадки)...")
    env_eval = gym.make('LunarLander-v3', render_mode='human')
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    
    for i in range(3):
        # При демонстрации сиды уже не передаем, чтобы проверить обобщение
        state, info = env_eval.reset()
        done = False
        truncated = False
        total_reward = 0.0
        
        while not (done or truncated):
            output = winner_net.activate(state)
            action = np.argmax(output)
            state, reward, done, truncated, info = env_eval.step(action)
            total_reward += reward
            
        print(f"Демонстрация {i+1}: Набрано очков: {total_reward:.1f}")
        
    env_eval.close()


if __name__ == '__main__':
    checkpoint_file = os.path.join(PATH, "checkpoints", "neat-checkpoint-1000") # Укажите имя вашего чекпоинта, если хотите восстановить
    multiprocessing.freeze_support()
    run_neat(checkpoint_file=checkpoint_file)