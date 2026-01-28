// drone_controller.ino
#include <Wire.h>
#include <SoftwareSerial.h>

// Конфигурация
#define RPI_RX 2
#define RPI_TX 3
SoftwareSerial rpiSerial(RPI_RX, RPI_TX);

// Структуры данных
struct ReceiverData {
  int channels[4];
  volatile byte last_channel[4];
  volatile unsigned long timer[4];
} rc;

struct PID {
  float kp, ki, kd;
  float i_term, prev_error;
} pid_pitch, pid_roll, pid_yaw;

struct DroneState {
  float pitch_angle = 0;
  float roll_angle = 0;
  float yaw_rate = 0;
  int throttle = 1500;
  bool armed = false;
  bool voice_mode = false;
} state;

struct MotorOutput {
  int esc1, esc2, esc3, esc4;
} motors;

// Глобальные переменные
unsigned long loop_timer;
int16_t gyro_data[3], accel_data[3];

void setup() {
  // Инициализация пинов
  DDRD |= B11110000;
  pinMode(13, OUTPUT);
  
  // Инициализация Serial
  Serial.begin(115200);
  rpiSerial.begin(9600);
  
  // Инициализация I2C
  Wire.begin();
  TWBR = 12;
  
  // Инициализация гироскопа
  init_mpu6050();
  
  // Инициализация прерываний для приемника
  init_receiver_interrupts();
  
  // Настройка PID
  setup_pid_controllers();
  
  Serial.println("Дрон инициализирован");
}

void loop() {
  // Чтение команд от Raspberry Pi
  check_voice_commands();
  
  // Чтение данных с гироскопа
  read_mpu6050();
  
  if (!state.armed) {
    // Режим ожидания
    motors.esc1 = motors.esc2 = motors.esc3 = motors.esc4 = 1000;
    
    // Проверка арминга
    if (rc.channels[0] < 1050 && rc.channels[1] < 1050 && 
        rc.channels[2] < 1050 && rc.channels[3] > 1950) {
      state.armed = true;
      Serial.println("Дрон включен");
    }
  } else {
    // Основной цикл полета
    if (state.voice_mode) {
      // Голосовое управление
      process_voice_control();
    } else {
      // Ручное управление
      process_manual_control();
    }
    
    // Расчет PID
    calculate_pid();
    
    // Расчет моторов
    calculate_motor_outputs();
    
    // Ограничение значений моторов
    limit_motor_outputs();
    
    // Проверка дизарминга
    if (rc.channels[0] > 1950 && rc.channels[1] < 1050 && 
        rc.channels[2] < 1050 && rc.channels[3] < 1050) {
      state.armed = false;
      state.voice_mode = false;
      Serial.println("Дрон выключен");
    }
  }
  
  // Отправка сигналов на ESC
  output_motor_signals();
  
  // Синхронизация цикла
  while (micros() - loop_timer < 4000);
  loop_timer = micros();
}

// Обработка голосовых команд
void check_voice_commands() {
  if (rpiSerial.available()) {
    String command = rpiSerial.readStringUntil('\n');
    command.trim();
    execute_voice_command(command);
  }
}

void execute_voice_command(String cmd) {
  Serial.print("Голосовая команда: ");
  Serial.println(cmd);
  
  state.voice_mode = true;
  
  if (cmd == "TAKEOFF") {
    state.throttle = 1600;
    Serial.println("Взлет");
  } 
  else if (cmd == "LAND") {
    state.throttle = 1400;
    Serial.println("Посадка");
  }
  else if (cmd == "HOVER") {
    state.throttle = 1550;
    Serial.println("Зависание");
  }
  else if (cmd == "FORWARD") {
    // Наклон вперед
    state.pitch_angle = 10;
    Serial.println("Вперед");
  }
  else if (cmd == "BACK") {
    state.pitch_angle = -10;
    Serial.println("Назад");
  }
  else if (cmd == "LEFT") {
    state.roll_angle = -10;
    Serial.println("Влево");
  }
  else if (cmd == "RIGHT") {
    state.roll_angle = 10;
    Serial.println("Вправо");
  }
  else if (cmd == "STOP") {
    state.voice_mode = false;
    Serial.println("Стоп");
  }
}

// Остальные функции (PID, моторы, гироскоп) аналогичны вашему коду
// ... 

void process_manual_control() {
  state.throttle = rc.channels[2];
  // Конвертация стиков в углы
}

void calculate_pid() {
  // PID расчеты
}

void calculate_motor_outputs() {
  motors.esc1 = state.throttle - pid_pitch.i_term - pid_roll.i_term - pid_yaw.i_term;
  motors.esc2 = state.throttle - pid_pitch.i_term + pid_roll.i_term + pid_yaw.i_term;
  motors.esc3 = state.throttle + pid_pitch.i_term + pid_roll.i_term - pid_yaw.i_term;
  motors.esc4 = state.throttle + pid_pitch.i_term - pid_roll.i_term + pid_yaw.i_term;
}

void output_motor_signals() {
  // Отправка PWM на ESC
}
