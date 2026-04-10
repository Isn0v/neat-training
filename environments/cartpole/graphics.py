import os

import matplotlib.pyplot as plt
import numpy as np

labels = ['DQN', 'NEAT']

PATH = os.path.dirname(os.path.abspath(__file__))

n_steps = [25923, 10548]
parameters = [114, 9]
density = [100.0, 75.0]
seconds = [0.138, 15.08]

reward_normal = [500.0, 500.0]
reward_noisy = [354.3, 500.0]

x = np.arange(len(labels))
width = 0.25

plt.style.use('seaborn-v0_8-whitegrid')
fig, axes = plt.subplots(3, 2, figsize=(14, 10))

# 1. Время
axes[2, 0].bar(labels, seconds, color=['#4C72B0', '#55A868'], width=0.5)
axes[2, 0].set_title('Время обучения\n(секунды)', fontsize=12)
for i, v in enumerate(seconds):
    axes[2, 0].text(i, v + 0.5, f'{v:.1f}', ha='center', fontsize=11, fontweight='bold')

# 1. Скорость
axes[0, 0].bar(labels, n_steps, color=['#4C72B0', '#55A868'], width=0.5)
axes[0, 0].set_title('Скорость обучения\n(число шагов)', fontsize=12)
for i, v in enumerate(n_steps):
    axes[0, 0].text(i, v + 500, str(v), ha='center', fontsize=11, fontweight='bold')

# 2. Параметры
axes[0, 1].bar(labels, parameters, color=['#4C72B0', '#55A868'], width=0.5)
axes[0, 1].set_title('Сложность модели\n(количество параметров)', fontsize=12)
for i, v in enumerate(parameters):
    axes[0, 1].text(i, v + 2, str(v), ha='center', fontsize=11, fontweight='bold')

# 3. Плотность
axes[1, 0].bar(labels, density, color=['#4C72B0', '#55A868'], width=0.5)
axes[1, 0].set_title('Плотность связей (%)\n', fontsize=12)
axes[1, 0].set_ylim(0, 120)
for i, v in enumerate(density):
    axes[1, 0].text(i, v + 2, f"{v}%", ha='center', fontsize=11, fontweight='bold')

# 4. Награды
rects1 = axes[1, 1].bar(x - width, reward_normal, width, label='Обычная среда', color='#4C72B0')
rects2 = axes[1, 1].bar(x, reward_noisy, width, label='Среда с шумом', color='#DD8452')

axes[1, 1].set_title('Робастность и качество\n(сравнение наград)', fontsize=12)
axes[1, 1].set_xticks(x)
axes[1, 1].set_xticklabels(labels)
axes[1, 1].set_ylim(0, 650)
axes[1, 1].legend(loc='upper right')

for rect in rects1 + rects2:
    height = rect.get_height()
    axes[1, 1].text(rect.get_x() + rect.get_width()/2., height + 10, f'{height:.0f}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig(f'{PATH}/cartpole_full_metrics.png', dpi=300, bbox_inches='tight')