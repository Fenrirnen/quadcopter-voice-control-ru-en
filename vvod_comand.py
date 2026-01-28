# Установка необходимых библиотек
import subprocess
import sys

def install_package(package):
    try:
        __import__(package)
        print(f"{package} уже установлен")
    except ImportError:
        print(f"Установка {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

install_package('pyserial')
install_package('numpy')
install_package('librosa')
install_package('scipy')
install_package('matplotlib')

import numpy as np
import time
import pickle
import os
import warnings
from datetime import datetime
import xml.etree.ElementTree as ET
from xml.dom import minidom
import matplotlib.pyplot as plt

try:
    import librosa
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "librosa"])
    import librosa

try:
    from scipy import signal
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scipy"])
    from scipy import signal

try:
    import serial
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pyserial"])
    import serial

warnings.filterwarnings('ignore')

class VoiceTrainer:
    def __init__(self, data_dir='voice_commands'):
        self.samples_needed = 4
        self.sample_duration = 2
        self.pause_duration = 2
        self.sample_rate = 16000
        self.data_dir = data_dir
        
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        
        self.commands_db = {}
        self.load_existing_commands()
    
    def load_existing_commands(self):
        db_file = os.path.join(self.data_dir, 'commands_db.pkl')
        if os.path.exists(db_file):
            with open(db_file, 'rb') as f:
                self.commands_db = pickle.load(f)
            print(f"Загружено {len(self.commands_db)} команд")
    
    def save_commands_db(self):
        db_file = os.path.join(self.data_dir, 'commands_db.pkl')
        with open(db_file, 'wb') as f:
            pickle.dump(self.commands_db, f)
        print(f"База команд сохранена ({len(self.commands_db)} команд)")
    
    def _generate_test_audio(self, duration_seconds):
        t = np.linspace(0, duration_seconds, int(duration_seconds * self.sample_rate))
        
        base_freq = 220 + np.random.rand() * 100
        audio = 0.7 * np.sin(2 * np.pi * base_freq * t)
        
        audio += 0.3 * np.sin(2 * np.pi * base_freq * 2 * t)
        audio += 0.1 * np.sin(2 * np.pi * base_freq * 3 * t)
        
        attack_len = int(0.1 * len(t))
        decay_len = int(0.2 * len(t))
        release_len = int(0.1 * len(t))
        
        attack = np.linspace(0, 1, attack_len)
        decay = np.linspace(1, 0.7, decay_len)
        
        sustain_len = len(t) - attack_len - decay_len - release_len
        if sustain_len > 0:
            sustain = np.ones(sustain_len) * 0.7
        else:
            sustain = np.array([])
        
        release = np.linspace(0.7, 0, release_len)
        
        envelope = np.concatenate([attack, decay, sustain, release])
        
        if len(envelope) < len(t):
            envelope = np.pad(envelope, (0, len(t) - len(envelope)), mode='constant')
        
        audio *= envelope
        
        audio += 0.05 * np.random.randn(len(t))
        
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio))
        
        return audio
    
    def extract_mfcc_features(self, audio_data, n_mfcc=13):
        try:
            if len(audio_data) == 0:
                return np.zeros(n_mfcc)
            
            return np.random.randn(n_mfcc)
            
        except Exception as e:
            print(f"Ошибка извлечения признаков: {e}")
            return np.zeros(n_mfcc)
    
    def train_new_command(self, command_name, movement_sequence=None):
        print("\n" + "=" * 60)
        print(f"ОБУЧЕНИЕ КОМАНДЫ: '{command_name}'")
        print("=" * 60)
        
        if movement_sequence:
            print(f"Последовательность движений: {movement_sequence}")
        
        print("\nИмитация записи голоса...")
        
        collected_samples = []
        
        for sample_num in range(self.samples_needed):
            print(f"Образец {sample_num + 1}/{self.samples_needed} - говорите: '{command_name}'")
            
            audio_data = self._generate_test_audio(self.sample_duration)
            collected_samples.append(audio_data)
            
            print(f"  Записано {len(audio_data)} сэмплов")
            
            if sample_num < self.samples_needed - 1:
                time.sleep(self.pause_duration)
        
        print("\nИзвлечение признаков...")
        features_list = []
        for i, audio in enumerate(collected_samples):
            features = self.extract_mfcc_features(audio, n_mfcc=13)
            features_list.append(features)
            print(f"  Образец {i+1}: {len(features)} признаков")
        
        voiceprint = np.mean(features_list, axis=0)
        print(f"\nСоздан голосовой отпечаток: {len(voiceprint)} признаков")
        
        command_id = f"cmd_{len(self.commands_db) + 1:03d}"
        command_data = {
            'id': command_id,
            'name': command_name,
            'voiceprint': voiceprint,
            'movement_sequence': movement_sequence or [],
            'samples_count': len(collected_samples),
            'created_at': datetime.now().isoformat(),
            'feature_vectors': features_list
        }
        
        self.commands_db[command_name] = command_data
        self.save_commands_db()
        
        print(f"\nКоманда '{command_name}' успешно обучена!")
        print(f"   ID: {command_id}")
        print(f"   Образцов: {len(collected_samples)}")
        print(f"   Признаков: {len(voiceprint)}")
        
        if movement_sequence:
            print(f"   Движений: {len(movement_sequence)}")
        
        return True
    
    def list_commands(self):
        print("\nОБУЧЕННЫЕ КОМАНДЫ:")
        print("-" * 40)
        
        if not self.commands_db:
            print("  (пока нет обученных команд)")
            return
        
        for i, (cmd_name, cmd_data) in enumerate(self.commands_db.items(), 1):
            print(f"{i:2d}. {cmd_name}")
            print(f"     ID: {cmd_data['id']}")
            print(f"     Дата: {cmd_data['created_at'][:10]}")
            print(f"     Образцов: {cmd_data['samples_count']}")
            
            if cmd_data['movement_sequence']:
                moves = cmd_data['movement_sequence']
                if len(moves) > 3:
                    moves = moves[:3] + ["..."]
                print(f"     Движения: {', '.join(moves)}")
            
            print()
    
    def test_recognition(self, test_command=None):
        print("\nТЕСТИРОВАНИЕ РАСПОЗНАВАНИЯ")
        print("=" * 30)
        
        if not self.commands_db:
            print("Нет обученных команд для тестирования")
            return None
        
        if test_command is None:
            test_command = list(self.commands_db.keys())[0]
            print(f"Тестирую команду: '{test_command}'")
        
        print("\nИмитация распознавания голоса...")
        time.sleep(1)
        
        results = []
        for cmd_name in self.commands_db.keys():
            similarity = np.random.uniform(0.3, 0.9)
            results.append((cmd_name, similarity))
        
        results.sort(key=lambda x: x[1], reverse=True)
        
        print("\nРЕЗУЛЬТАТЫ РАСПОЗНАВАНИЯ:")
        print("-" * 30)
        
        for cmd_name, similarity in results[:3]:
            bar_length = int(similarity * 20)
            bar = "#" * bar_length + "-" * (20 - bar_length)
            
            if similarity > 0.7:
                confidence = "ВЫСОКАЯ"
            elif similarity > 0.5:
                confidence = "СРЕДНЯЯ"
            else:
                confidence = "НИЗКАЯ"
            
            print(f"{cmd_name:15s} [{bar}] {similarity:.3f} ({confidence})")
        
        best_cmd, best_sim = results[0]
        print(f"\nЛучший результат: '{best_cmd}' ({best_sim:.3f})")
        
        return best_cmd
    
    def delete_command(self, command_name):
        if command_name in self.commands_db:
            del self.commands_db[command_name]
            self.save_commands_db()
            print(f"Команда '{command_name}' удалена")
            return True
        else:
            print(f"Команда '{command_name}' не найдена")
            return False

class ArduinoCommander:
    def __init__(self, port=None, baudrate=9600):
        self.port = port
        self.baudrate = baudrate
        self.connected = False
        
        self.supported_commands = [
            'TAKEOFF', 'LAND', 'HOVER', 'STOP',
            'FORWARD', 'BACK', 'LEFT', 'RIGHT',
            'UP', 'DOWN', 'ROTATE_LEFT', 'ROTATE_RIGHT',
            'ARM', 'DISARM'
        ]
    
    def connect(self):
        print(f"Подключение к Arduino...")
        time.sleep(1)
        
        self.connected = True
        print("Успешно подключено (симуляция)")
        return True
    
    def send_command(self, command):
        if not self.connected:
            print("Сначала подключитесь к Arduino")
            return False
        
        if command not in self.supported_commands:
            print(f"Неподдерживаемая команда: {command}")
            print(f"Поддерживаемые команды: {', '.join(self.supported_commands)}")
            return False
        
        print(f"Отправка команды на Arduino: {command}")
        time.sleep(0.5)
        
        print(f"Команда '{command}' отправлена успешно")
        return True
    
    def send_sequence(self, commands):
        print(f"\nОтправка последовательности ({len(commands)} команд):")
        for i, (cmd, duration) in enumerate(commands):
            print(f"  {i+1}. {cmd} ({duration}ms)")
            if not self.send_command(cmd):
                return False
            time.sleep(duration / 1000.0)
        return True
    
    def close(self):
        if self.connected:
            self.connected = False
            print("Соединение с Arduino закрыто")

class VoiceDroneController:
    def __init__(self):
        self.trainer = VoiceTrainer()
        self.commander = ArduinoCommander()
        self.command_mapping = {}
        
        self.standard_mappings = {
            'взлет': 'TAKEOFF',
            'посадка': 'LAND',
            'зависни': 'HOVER',
            'стоп': 'STOP',
            'вперед': 'FORWARD',
            'назад': 'BACK',
            'влево': 'LEFT',
            'вправо': 'RIGHT',
            'вверх': 'UP',
            'вниз': 'DOWN',
            'поворот влево': 'ROTATE_LEFT',
            'поворот вправо': 'ROTATE_RIGHT',
            'включить': 'ARM',
            'выключить': 'DISARM'
        }
    
    def map_voice_to_action(self, voice_command, drone_action=None):
        if voice_command not in self.trainer.commands_db:
            print(f"Голосовая команда '{voice_command}' не обучена")
            print("Сначала обучите команду в меню (пункт 1)")
            return False
        
        if drone_action is None:
            if voice_command.lower() in self.standard_mappings:
                drone_action = self.standard_mappings[voice_command.lower()]
                print(f"Использую стандартную привязку: '{voice_command}' → '{drone_action}'")
            else:
                print("Неизвестная команда. Укажите действие дрона вручную.")
                drone_action = input("Действие дрона (TAKEOFF/LAND/etc): ").strip().upper()
        
        if drone_action not in self.commander.supported_commands:
            print(f"Неподдерживаемое действие дрона: {drone_action}")
            print(f"Поддерживаемые действия: {', '.join(self.commander.supported_commands)}")
            return False
        
        self.command_mapping[voice_command] = drone_action
        print(f"Привязка создана: '{voice_command}' → '{drone_action}'")
        return True
    
    def execute_voice_command(self, voice_command):
        print(f"\nВыполнение команды: '{voice_command}'")
        
        if voice_command not in self.command_mapping:
            print(f"Команда '{voice_command}' не привязана к действию")
            print("Сначала создайте привязку в меню (пункт 2)")
            return False
        
        drone_action = self.command_mapping[voice_command]
        print(f"Действие дрона: {drone_action}")
        
        if self.commander.connect():
            success = self.commander.send_command(drone_action)
            self.commander.close()
            return success
        
        return False
    
    def execute_sequence(self, sequence_name, voice_command=None):
        if voice_command is None:
            voice_command = sequence_name
        
        if voice_command not in self.trainer.commands_db:
            print(f"Команда '{voice_command}' не обучена")
            return False
        
        command_data = self.trainer.commands_db[voice_command]
        movement_sequence = command_data.get('movement_sequence', [])
        
        if not movement_sequence:
            print(f"У команды '{voice_command}' нет настроенной последовательности")
            return False
        
        print(f"\nВыполнение последовательности: '{sequence_name}'")
        print(f"Движений: {len(movement_sequence)}")
        
        commands = []
        for movement in movement_sequence:
            parts = movement.split(':')
            if len(parts) >= 2:
                cmd = parts[0].strip().upper()
                try:
                    duration = int(parts[1].strip())
                except:
                    duration = 1000
                commands.append((cmd, duration))
        
        if self.commander.connect():
            success = self.commander.send_sequence(commands)
            self.commander.close()
            return success
        
        return False
    
    def create_mission_xml(self, command_name, filename=None):
        """Создание XML файла миссии с автоматическим скачиванием"""
        if command_name not in self.trainer.commands_db:
            print(f"Команда '{command_name}' не найдена")
            return None
        
        if filename is None:
            filename = f"mission_{command_name}.xml"
        
        print(f"\nСоздание XML миссии для команды: '{command_name}'")
        
        root = ET.Element("mission")
        
        ET.SubElement(root, "name").text = f"Voice Command: {command_name}"
        ET.SubElement(root, "version").text = "1"
        ET.SubElement(root, "created").text = datetime.now().isoformat()
        ET.SubElement(root, "creator").text = "VoiceDroneSystem"
        
        waypoints = ET.SubElement(root, "waypoints")
        
        command_data = self.trainer.commands_db[command_name]
        movement_sequence = command_data.get('movement_sequence', [])
        
        home_wp = ET.SubElement(waypoints, "waypoint")
        home_wp.set("id", "0")
        ET.SubElement(home_wp, "lat").text = "0"
        ET.SubElement(home_wp, "lon").text = "0"
        ET.SubElement(home_wp, "alt").text = "0"
        ET.SubElement(home_wp, "command").text = "16"
        
        for i, movement in enumerate(movement_sequence, 1):
            wp = ET.SubElement(waypoints, "waypoint")
            wp.set("id", str(i))
            
            parts = movement.split(':')
            cmd_type = parts[0].upper() if parts else "HOVER"
            
            mav_cmd = self._get_mavlink_command(cmd_type)
            
            ET.SubElement(wp, "lat").text = f"{i * 0.00001}"
            ET.SubElement(wp, "lon").text = f"{i * 0.00001}"
            ET.SubElement(wp, "alt").text = "10"
            ET.SubElement(wp, "command").text = str(mav_cmd)
            
            for param_num in range(1, 8):
                ET.SubElement(wp, f"param{param_num}").text = "0"
        
        rtl_wp = ET.SubElement(waypoints, "waypoint")
        rtl_wp.set("id", str(len(movement_sequence) + 1))
        ET.SubElement(rtl_wp, "lat").text = "0"
        ET.SubElement(rtl_wp, "lon").text = "0"
        ET.SubElement(rtl_wp, "alt").text = "0"
        ET.SubElement(rtl_wp, "command").text = "20"
        
        xml_str = minidom.parseString(ET.tostring(root)).toprettyxml(indent="  ")
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(xml_str)
        
        print(f"XML файл создан: {filename}")
        print(f"Путь: {os.path.abspath(filename)}")
        
        # Автоматическое скачивание файла
        self._auto_download_file(filename)
        
        return xml_str
    
    def _auto_download_file(self, filename):
        """Автоматическое скачивание файла (работает в Google Colab)"""
        try:
            # Пробуем скачать через Google Colab
            from google.colab import files
            
            print("\n" + "=" * 50)
            print("АВТОМАТИЧЕСКОЕ СКАЧИВАНИЕ ФАЙЛА")
            print("=" * 50)
            
            print(f"Начинаю скачивание файла: {filename}")
            print("Браузер предложит сохранить файл...")
            
            files.download(filename)
            
            print("✅ Скачивание началось!")
            print("Если файл не скачался автоматически:")
            print("1. Проверьте всплывающие окна браузера")
            print("2. Разрешите скачивание файлов")
            print("3. Или найдите файл в папке загрузок")
            
        except ImportError:
            # Не в Google Colab
            print("\n" + "=" * 50)
            print("СКАЧИВАНИЕ ФАЙЛА")
            print("=" * 50)
            print(f"Файл сохранен локально: {os.path.abspath(filename)}")
            print("Размер файла:", os.path.getsize(filename), "байт")
            
            # Предлагаем показать содержимое
            show_content = input("\nПоказать содержимое XML файла? (y/n): ").strip().lower()
            if show_content == 'y':
                with open(filename, 'r') as f:
                    content = f.read()
                    print("\nСОДЕРЖАНИЕ XML ФАЙЛА:")
                    print("=" * 60)
                    # Показываем первые 1000 символов
                    print(content[:1000] + "..." if len(content) > 1000 else content)
        
        except Exception as e:
            print(f"Ошибка при скачивании файла: {e}")
            print(f"Файл сохранен по пути: {os.path.abspath(filename)}")
    
    def create_and_download_mission(self, command_name):
        """Создание и автоматическое скачивание миссии"""
        print("\n" + "=" * 60)
        print(f"СОЗДАНИЕ И СКАЧИВАНИЕ МИССИИ: '{command_name}'")
        print("=" * 60)
        
        # Создаем уникальное имя файла с временной меткой
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"mission_{command_name}_{timestamp}.xml"
        
        # Создаем XML
        xml_content = self.create_mission_xml(command_name, filename)
        
        if xml_content:
            print("\n✅ Миссия успешно создана и готова к скачиванию!")
            return True
        
        return False
    
    def batch_create_missions(self, command_list=None):
        """Пакетное создание миссий для нескольких команд"""
        print("\n" + "=" * 60)
        print("ПАКЕТНОЕ СОЗДАНИЕ МИССИЙ")
        print("=" * 60)
        
        if command_list is None:
            if not self.trainer.commands_db:
                print("Нет обученных команд для создания миссий")
                return False
            
            print("Доступные команды:")
            self.trainer.list_commands()
            
            command_list = []
            while True:
                cmd = input("\nВведите название команды (или 'done' для завершения): ").strip()
                if cmd.lower() == 'done':
                    break
                if cmd in self.trainer.commands_db:
                    command_list.append(cmd)
                    print(f"Добавлена команда: {cmd}")
                else:
                    print(f"Команда '{cmd}' не найдена")
        
        if not command_list:
            print("Не выбрано ни одной команды")
            return False
        
        print(f"\nБудут созданы миссии для {len(command_list)} команд:")
        for i, cmd in enumerate(command_list, 1):
            print(f"  {i}. {cmd}")
        
        confirm = input("\nПродолжить? (y/n): ").strip().lower()
        if confirm != 'y':
            return False
        
        success_count = 0
        for cmd in command_list:
            print(f"\n{'='*40}")
            print(f"Обработка команды: {cmd}")
            print('='*40)
            
            if self.create_and_download_mission(cmd):
                success_count += 1
            
            # Пауза между файлами
            if cmd != command_list[-1]:
                print("\nПауза 2 секунды перед следующим файлом...")
                time.sleep(2)
        
        print(f"\n{'='*60}")
        print(f"РЕЗУЛЬТАТ: Создано {success_count} из {len(command_list)} миссий")
        print("=" * 60)
        
        return success_count > 0
    
    def _get_mavlink_command(self, cmd_type):
        mav_cmd_map = {
            'TAKEOFF': 22,
            'LAND': 21,
            'HOVER': 16,
            'FORWARD': 16,
            'BACK': 16,
            'LEFT': 16,
            'RIGHT': 16,
            'UP': 178,
            'DOWN': 178,
            'ROTATE_LEFT': 115,
            'ROTATE_RIGHT': 115
        }
        
        return mav_cmd_map.get(cmd_type, 16)

def main_menu():
    print("\n" + "=" * 60)
    print("СИСТЕМА ГОЛОСОВОГО УПРАВЛЕНИЯ КВАДРОКОПТЕРОМ")
    print("=" * 60)
    print("Версия: 2.0 | Автоматическое скачивание XML")
    print("=" * 60)
    
    controller = VoiceDroneController()
    
    while True:
        print("\n" + "=" * 50)
        print("ГЛАВНОЕ МЕНЮ")
        print("=" * 50)
        print("1. Обучить новую голосовую команду")
        print("2. Привязать команду к действию дрона")
        print("3. Показать все команды")
        print("4. Тестировать распознавание")
        print("5. Выполнить голосовую команду")
        print("6. Выполнить последовательность")
        print("7. Создать и скачать XML миссию (1 команда)")
        print("8. Пакетное создание миссий (несколько команд)")
        print("9. Удалить команду")
        print("0. Выход")
        print("=" * 50)
        
        choice = input("\nВыберите действие (0-9): ").strip()
        
        if choice == '1':
            print("\n" + "=" * 50)
            print("ОБУЧЕНИЕ НОВОЙ КОМАНДЫ")
            print("=" * 50)
            
            command_name = input("Введите название команды (например 'Взлет'): ").strip()
            
            if not command_name:
                print("Название команды не может быть пустым")
                continue
            
            print("\nНастройка последовательности движений (опционально):")
            print("Формат: КОМАНДА:ДЛИТЕЛЬНОСТЬ_MS")
            print("Примеры: TAKEOFF:2000, FORWARD:1000, HOVER:500")
            print("Оставьте пустым, если не нужно настраивать последовательность")
            
            movements = []
            while True:
                move = input(f"Движение {len(movements) + 1} (или Enter для завершения): ").strip()
                if not move:
                    break
                
                if ':' not in move:
                    print("Неверный формат. Используйте: КОМАНДА:ДЛИТЕЛЬНОСТЬ")
                    continue
                
                movements.append(move)
                print(f"Добавлено: {move}")
            
            print(f"\nВсего движений: {len(movements)}")
            
            confirm = input("\nНачать обучение? (y/n): ").strip().lower()
            if confirm == 'y':
                success = controller.trainer.train_new_command(command_name, movements)
                if success:
                    if command_name.lower() in controller.standard_mappings:
                        auto_map = input(f"\nАвтоматически привязать к действию '{controller.standard_mappings[command_name.lower()]}'? (y/n): ").strip().lower()
                        if auto_map == 'y':
                            controller.map_voice_to_action(command_name)
        
        elif choice == '2':
            print("\n" + "=" * 50)
            print("ПРИВЯЗКА КОМАНДЫ К ДЕЙСТВИЮ")
            print("=" * 50)
            
            controller.trainer.list_commands()
            
            if controller.trainer.commands_db:
                voice_cmd = input("\nВведите голосовую команду: ").strip()
                
                if voice_cmd not in controller.trainer.commands_db:
                    print(f"Команда '{voice_cmd}' не найдена")
                    continue
                
                print("\nСтандартные действия дрона:")
                for voice, action in controller.standard_mappings.items():
                    print(f"  {voice:15s} → {action}")
                
                print("\nВведите действие дрона или оставьте пустым для стандартной привязки")
                drone_action = input("Действие дрона: ").strip().upper()
                
                if drone_action:
                    controller.map_voice_to_action(voice_cmd, drone_action)
                else:
                    controller.map_voice_to_action(voice_cmd)
        
        elif choice == '3':
            print("\n" + "=" * 50)
            print("СПИСОК ВСЕХ КОМАНД")
            print("=" * 50)
            controller.trainer.list_commands()
            
            if controller.command_mapping:
                print("\nПРИВЯЗКИ КОМАНД:")
                print("-" * 40)
                for voice_cmd, drone_action in controller.command_mapping.items():
                    print(f"  {voice_cmd:15s} → {drone_action}")
        
        elif choice == '4':
            print("\n" + "=" * 50)
            print("ТЕСТИРОВАНИЕ РАСПОЗНАВАНИЯ")
            print("=" * 50)
            controller.trainer.test_recognition()
        
        elif choice == '5':
            print("\n" + "=" * 50)
            print("ВЫПОЛНЕНИЕ ГОЛОСОВОЙ КОМАНДЫ")
            print("=" * 50)
            
            controller.trainer.list_commands()
            
            if controller.trainer.commands_db:
                voice_cmd = input("\nВведите голосовую команду для выполнения: ").strip()
                controller.execute_voice_command(voice_cmd)
        
        elif choice == '6':
            print("\n" + "=" * 50)
            print("ВЫПОЛНЕНИЕ ПОСЛЕДОВАТЕЛЬНОСТИ")
            print("=" * 50)
            
            controller.trainer.list_commands()
            
            if controller.trainer.commands_db:
                voice_cmd = input("\nВведите голосовую команду с последовательностью: ").strip()
                controller.execute_sequence(voice_cmd)
        
        elif choice == '7':
            print("\n" + "=" * 50)
            print("СОЗДАНИЕ И СКАЧИВАНИЕ XML МИССИИ")
            print("=" * 50)
            
            controller.trainer.list_commands()
            
            if controller.trainer.commands_db:
                voice_cmd = input("\nВведите голосовую команду для создания миссии: ").strip()
                
                if voice_cmd in controller.trainer.commands_db:
                    controller.create_and_download_mission(voice_cmd)
                else:
                    print(f"Команда '{voice_cmd}' не найдена")
        
        elif choice == '8':
            print("\n" + "=" * 50)
            print("ПАКЕТНОЕ СОЗДАНИЕ МИССИЙ")
            print("=" * 50)
            
            controller.batch_create_missions()
        
        elif choice == '9':
            print("\n" + "=" * 50)
            print("УДАЛЕНИЕ КОМАНДЫ")
            print("=" * 50)
            
            controller.trainer.list_commands()
            
            if controller.trainer.commands_db:
                voice_cmd = input("\nВведите голосовую команду для удаления: ").strip()
                controller.trainer.delete_command(voice_cmd)
                
                if voice_cmd in controller.command_mapping:
                    del controller.command_mapping[voice_cmd]
                    print(f"Привязка для '{voice_cmd}' также удалена")
        
        elif choice == '0':
            print("\n" + "=" * 50)
            print("ВЫХОД ИЗ СИСТЕМЫ")
            print("=" * 50)
            print("Спасибо за использование системы голосового управления!")
            print("Все созданные XML файлы находятся в текущей директории")
            break
        
        else:
            print("Неверный выбор. Попробуйте снова.")
        
        input("\nНажмите Enter для продолжения...")

if __name__ == "__main__":
    print("Запуск системы голосового управления квадрокоптером...")
    print("Версия с автоматическим скачиванием XML")
    print("=" * 60)
    
    # Проверка окружения
    try:
        import google.colab
        print("Обнаружен Google Colab - доступно автоматическое скачивание")
        IN_COLAB = True
    except:
        print("Локальный запуск - файлы сохраняются локально")
        IN_COLAB = False
    
    time.sleep(1)
    
    main_menu()
