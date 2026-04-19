#include <Servo.h>

Servo servos[5];

const int pins[5] = {3, 6, 9, 10, 11};
int angulos[5] = {0, 0, 0, 0, 0};
int angulos_previos[5] = {-1, -1, -1, -1, -1}; // Forzar actualización inicial

void setup() {
  Serial.begin(9600);
  for(int i=0; i<5; i++) {
    servos[i].attach(pins[i]);
  }
}

void loop() {
  while (Serial.available() > 0) {
    String entrada = Serial.readStringUntil('\n');
    procesarComando(entrada);
  }
}

void procesarComando(String comando) {
  int indices[5] = {0,0,0,0,0};
  int start=0;
  int commaIndex = comando.indexOf(',');

  int i = 0;
  while(commaIndex != -1 && i < 5) {
    String valStr = comando.substring(start, commaIndex);
    indices[i] = valStr.toInt();
    start = commaIndex + 1;
    commaIndex = comando.indexOf(',', start);
    i++;
  }

  // Último valor
  if(i < 5) {
    indices[i] = comando.substring(start).toInt();
  }

  for(int j=0; j<5; j++) {
    // Si el ángulo cambió, actualiza servo y adjunta si estaba desactivado
    if(indices[j] != angulos_previos[j]) {
      angulos_previos[j] = indices[j];
      if(indices[j] == 0) {
        // Posición abierta, detacherar servo para evitar movimiento
        servos[j].detach();
      } else {
        // Posición cerrada, attachar y mover
        if(!servos[j].attached()) {
          servos[j].attach(pins[j]);
        }
        servos[j].write(indices[j]);
      }
    }
  }
}
