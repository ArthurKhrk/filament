import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten

with open('time_data.txt', 'r') as f:  # файл со всеми данными
    time_data = []  # текущее время
    ch_data = []  # данные филамента (?)
    lines = f.readlines()[:-3]
    for line in lines[1:]:
        numbers = [float(num) for num in line.split()]
        time_data.append(numbers[0])
        ch_data.append(numbers[1])

with open('filament_times.txt', 'r') as f:  # файл со всеми филаментами
    filament_time = []  # время филамента
    filament_count = []  # номер филамента
    for line in f:
        numbers = [float(num) for num in line.split()]
        filament_time.append(numbers[0])
        filament_count.append(int(numbers[1]))

window_size = 10  # метод скользящего окна(?)
input_data = []
output_data = []
for i in range(window_size, len(filament_count)-1):
    current_time_data = time_data[i-window_size:i]
    current_ch_data = ch_data[i-window_size:i]
    current_filament_time = filament_time[i]
    print(current_time_data)
    print()
    input_data.append([current_time_data, current_ch_data])
    output_data.append(current_filament_time)

# Закодируем данные в форму, которую можно использовать для обучения модели
X = np.array(input_data).reshape((-1, window_size, 2, 1))
y = np.array(output_data).reshape((-1, 1))

# Создание модели
model = Sequential()

# Добавление сверточного слоя
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(window_size, 2, 1)))

# Добавление слоя пулинга
model.add(MaxPooling2D(pool_size=(2, 1)))

# Добавление слоя Flatten
model.add(Flatten())

# Добавление полносвязного слоя
model.add(Dense(64, activation='relu'))

# Добавление выходного слоя
model.add(Dense(5, activation='linear'))

# Скомпилируем и обучим модель
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])
model.fit(X, y, epochs=100, validation_split=0.2)

# Используем модель для предсказания времени филамента
test_data = []
for i in range(200, 210):
    test_data.append([time_data[i], ch_data[i]])

print(test_data)

input_data = np.array(test_data)
input_data = input_data.reshape((1, window_size, 2, 1))
prediction = model.predict(input_data)
print(prediction)
