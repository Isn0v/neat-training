import os
import random
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

# Пути для сохранения
PATH = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(PATH, "results_universal")
os.makedirs(MODELS_DIR, exist_ok=True)

# ======================================================================
# 1. ОБЕРТКА ДЛЯ РАНДОМИЗАЦИИ ПОГОДЫ (Domain Randomization)
# ======================================================================
class UniversalWindWrapper(gym.Wrapper):
    """
    При каждом сбросе среды случайным образом меняет силу ветра.
    """
    def __init__(self, env):
        super().__init__(env)
    
    def reset(self, **kwargs):
        # 50% вероятность штиля, 50% вероятность ветра
        if random.random() < 0.5:
            self.env.unwrapped.enable_wind = False
            self.env.unwrapped.wind_power = 0.0
            self.env.unwrapped.turbulence_power = 0.0
        else:
            self.env.unwrapped.enable_wind = True
            # Случайная сила ветра (от умеренного до урагана)
            self.env.unwrapped.wind_power = random.uniform(10.0, 20.0) 
            self.env.unwrapped.turbulence_power = random.uniform(0.5, 2.0)
            
        return self.env.reset(**kwargs)

# ======================================================================
# 2. ОСНОВНАЯ ФУНКЦИЯ ОБУЧЕНИЯ
# ======================================================================
def train_universal_dqn():
    """Запускает обучение универсального DQN-агента."""
    
    # Создаем базовую среду с поддержкой ветра, затем оборачиваем ее
    base_env = gym.make('LunarLander-v3', enable_wind=True)
    env = UniversalWindWrapper(base_env)
    env = Monitor(env) # Необходимо для корректного логирования в SB3

    # Для оценки (EvalCallback) тоже используем рандомизированную среду,
    # чтобы алгоритм сохранял модель только если она хороша ВЕЗДЕ.
    eval_base_env = gym.make('LunarLander-v3', enable_wind=True)
    eval_env = UniversalWindWrapper(eval_base_env)
    eval_env = Monitor(eval_env)
    
    eval_callback = EvalCallback(
        eval_env, 
        best_model_save_path=MODELS_DIR,
        log_path=MODELS_DIR, 
        eval_freq=10000,
        deterministic=True, 
        render=False
    )

    # Инициализируем модель с надежными параметрами (защита от забывания)
    model = DQN(
        "MlpPolicy",
        env,
        policy_kwargs=dict(net_arch=[512, 512]), # Увеличенная "емкость" мозга для сложных условий
        learning_rate=1e-4,        
        buffer_size=200000,        
        learning_starts=10000,     
        batch_size=256,            
        gamma=0.99,                
        target_update_interval=1000, 
        exploration_fraction=0.2,  
        verbose=1,                 
        tensorboard_log=os.path.join(PATH, "dqn_universal_logs")
    )

    print("\n[INFO] Начинаем смешанное обучение DQN (Штиль + Ветер)...")
    # Для смешанных условий агенту потребуется больше опыта.
    model.learn(total_timesteps=600000, callback=eval_callback, tb_log_name="Universal_DQN")
    
    print("\n[INFO] Обучение завершено.")
    model.save(os.path.join(MODELS_DIR, "dqn_universal_final"))
    env.close()

# ======================================================================
# 3. ДЕМОНСТРАЦИЯ
# ======================================================================
def evaluate_universal_dqn():
    print("\n[INFO] Запуск демонстрации...")
    
    # Тестируем на жестких условиях ветра, чтобы проверить результат
    env = gym.make('LunarLander-v3', enable_wind=True, wind_power=15.0, render_mode='human')
    
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
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            
        print(f"Демонстрация {i+1}: Набрано очков (При ветре 15.0): {total_reward:.1f}")
        
    env.close()

if __name__ == '__main__':
    # train_universal_dqn()
    # После обучения закомментируйте строку выше и раскомментируйте нижнюю
    evaluate_universal_dqn()