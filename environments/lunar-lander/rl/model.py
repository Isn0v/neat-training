import os
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback

# Пути для сохранения
PATH = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(PATH, "results")
os.makedirs(MODELS_DIR, exist_ok=True)

def train_dqn():
    """Запускает обучение агента с помощью DQN."""
    # 1. Создаем среду обучения
    env = gym.make('LunarLander-v3')

    # 2. Создаем тестовую среду для оценки (чтобы сохранять лучшую модель)
    eval_env = gym.make('LunarLander-v3')
    
    # Callback будет проверять агента каждые 10 000 шагов 
    # и сохранять модель, если средняя награда выросла
    eval_callback = EvalCallback(
        eval_env, 
        best_model_save_path=MODELS_DIR,
        log_path=PATH, 
        eval_freq=10000,
        deterministic=True, 
        render=False
    )

    # 3. Инициализируем модель DQN
    # Параметры подобраны специально под LunarLander
    model = DQN(
        "MlpPolicy",               # Обычная многослойная полносвязная сеть
        env,
        learning_rate=1e-3,        # Скорость обучения
        buffer_size=100000,        # Размер памяти (Replay Buffer)
        learning_starts=1000,      # Начать учиться после 1000 случайных шагов
        batch_size=128,            # Размер пакета данных для градиентного спуска
        gamma=0.99,                # Дисконтирующий множитель (взгляд в будущее)
        exploration_fraction=0.1,  # Доля времени на исследование (epsilon decay)
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        target_update_interval=250,# Как часто обновлять Target Network
        verbose=1,                 # Выводить инфу в консоль
        tensorboard_log=PATH   # Папка для логов
    )

    print("\n[INFO] Начинаем обучение DQN...")
    # Обучаем агента. Для LunarLander обычно нужно от 300k до 500k шагов.
    # В отличие от NEAT, здесь мы считаем не поколения, а именно шаги среды (N_steps)!
    model.learn(total_timesteps=500000, callback=eval_callback, tb_log_name="DQN_run")
    
    print("\n[INFO] Обучение завершено.")
    # Сохраняем модель последнего шага (на всякий случай)
    model.save(os.path.join(MODELS_DIR, "dqn_lunar_final"))
    env.close()

def evaluate_best_dqn():
    """Демонстрирует работу лучшего обученного агента."""
    print("\n[INFO] Запуск демонстрации...")
    env = gym.make('LunarLander-v3', render_mode='human')
    
    # Загружаем лучшую модель, которую сохранил EvalCallback
    model_path = os.path.join(MODELS_DIR, "best_model.zip")
    if not os.path.exists(model_path):
        print(f"[ОШИБКА] Модель не найдена по пути {model_path}")
        return

    model = DQN.load(model_path)

    for i in range(3):
        obs, info = env.reset()
        done = False
        truncated = False
        total_reward = 0.0
        
        while not (done or truncated):
            # deterministic=True заставляет сеть выбирать лучшее действие (жадная стратегия)
            # без случайных исследовательских шагов
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            
        print(f"Демонстрация {i+1}: Набрано очков: {total_reward:.1f}")
        
    env.close()

if __name__ == '__main__':
    # Сначала запускаем обучение (займет некоторое время)
    # train_dqn()
    
    # Затем смотрим, как он летает
    evaluate_best_dqn()