# Лабораторная работа №1. 
# Анализ защищенности систем искусственного интеллекта
## Группа: ББМО-02-23
## Студент: Васильев Григорий Максимович
## Вариант: 3

# Получаем копию репозитория EEL6812_DeepFool_Project на нашем локальном устройстве

```
!git clone https://github.com/ewatson2/EEL6812_DeepFool_Project
```

# Переходим в директорию EEL6812_DeepFool_Project, созданную ранее с помощью команды git clone

```
%cd EEL6812_DeepFool_Project/
```

# Импорт библиотек

* warnings: Модуль для управления предупреждениями, позволяющий игнорировать или фильтровать их.
* numpy: Библиотека для работы с многомерными массивами и выполнением численных операций.
* json: Модуль для работы с данными в формате JSON, включая их парсинг и сериализацию.
* torch: Основная библиотека для создания и обучения нейронных сетей с использованием PyTorch.
* torch.utils.data: Модуль, предоставляющий инструменты для загрузки и обработки данных, включая DataLoader и random_split.
* torchvision: Библиотека для работы с изображениями, предоставляющая набор инструментов для обработки и трансформации изображений.
* torchvision.transforms: Модуль для применения различных преобразований к изображениям, таких как нормализация и аугментация.

```
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import json, torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models
from torchvision.transforms import transforms
```

# Импортируем пользовательские модели и утилиты из соответствующих модулей проекта.

Из models.project_models загружаются архитектуры нейронных сетей, такие как FC_500_150, LeNet_CIFAR, LeNet_MNIST и Net, а из utils.project_utils — функции get_clip_bounds, evaluate_attack и display_attack, необходимые для обработки данных, оценки атак на модели и визуализации результатов.

```
from models.project_models import FC_500_150, LeNet_CIFAR, LeNet_MNIST, Net
from utils.project_utils import get_clip_bounds, evaluate_attack, display_attack
rand_seed = 3 # мой номер
```

# Устанавливаются семена случайных чисел для двух библиотек: numpy и torch

* np.random.seed(rand_seed): Эта команда фиксирует генерацию случайных чисел в библиотеке NumPy, чтобы гарантировать воспроизводимость результатов при повторных запусках кода.
* torch.manual_seed(rand_seed): Устанавливает тот же самый семя для генератора случайных чисел в PyTorch, что также помогает обеспечить стабильность и воспроизводимость экспериментов, особенно при работе с нейронными сетями.

```
np.random.seed(rand_seed)
torch.manual_seed(rand_seed)
```

# Выполняется импорт библиотеки torch, которая является основным пакетом для работы с тензорами и построением нейронных сетей в PyTorch

```
import torch
device = torch.device("cuda")
```

**Задаются параметры нормализации для набора данных MNIST**, включая среднее значение и стандартное отклонение, и вычисляются минимальные и максимальные границы для обрезки значений пикселей. Затем определяются трансформации для изображений: преобразование в тензоры и нормализация, а также случайное горизонтальное отражение для обучающих изображений. После этого загружается набор данных MNIST, который делится на обучающую и валидационную выборки, а также создается тестовый набор данных с применением соответствующих трансформаций.

```
mnist_mean = 0.5  # Среднее значение для нормализации MNIST
mnist_std = 0.5   # Стандартное отклонение для нормализации MNIST
mnist_dim = 28    # Размерность изображений MNIST (28x28 пикселей)

# Получаем минимальные и максимальные границы для обрезки значений пикселей
mnist_min, mnist_max = get_clip_bounds(mnist_mean, mnist_std, mnist_dim)
mnist_min = mnist_min.to(device)  # Перемещаем минимальное значение на устройство (GPU или CPU)
mnist_max = mnist_max.to(device)  # Перемещаем максимальное значение на устройство (GPU или CPU)

# Определяем трансформации для тестовых изображений MNIST
mnist_tf = transforms.Compose([
    transforms.ToTensor(),  # Преобразуем изображения в тензоры
    transforms.Normalize(mean=mnist_mean, std=mnist_std)  # Нормализуем тензоры
])

# Определяем трансформации для обучающих изображений MNIST с добавлением случайного горизонтального отражения
mnist_tf_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),  # Случайно отражаем изображения по горизонтали
    transforms.ToTensor(),  # Преобразуем изображения в тензоры
    transforms.Normalize(mean=mnist_mean, std=mnist_std)  # Нормализуем тензоры
])

# Обратная нормализация изображений
mnist_tf_inv = transforms.Compose([
    transforms.Normalize(mean=0.0, std=np.divide(1.0, mnist_std)),  # Инвертируем нормализацию для визуализации
    transforms.Normalize(mean=np.multiply(-1.0, mnist_std), std=1.0)  # Завершаем обратную нормализацию
])

# Загружаем набор данных MNIST для обучения
mnist_temp = datasets.MNIST(root='datasets/mnist', train=True, download=True, transform=mnist_tf_train)
mnist_train, mnist_val = random_split(mnist_temp, [50000, 10000])  # Делим набор данных на обучающий и валидационный
mnist_test = datasets.MNIST(root='datasets/mnist', train=False, download=True, transform=mnist_tf)  # Загружаем тестовый набор данных MNIST
```

## Тут происходит подготовка данных для работы с набором CIFAR-10. Сначала задаются средние значения и стандартные отклонения для нормализации изображений, затем определяются минимальные и максимальные границы для обрезки значений пикселей. Устанавливаются трансформации для обучения и тестирования, включая случайную обрезку, горизонтальное отражение и нормализацию. Далее загружается обучающий и тестовый набор данных CIFAR-10, при этом обучающий набор разбивается на обучающую и валидационную выборки. В конце задаются названия классов, содержащихся в наборе данных.
