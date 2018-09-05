/*
  Optical SP02 Detection (SPK Algorithm) using the MAX30105 Breakout
  By: Nathan Seidle @ SparkFun Electronics
  Date: October 19th, 2016
  https://github.com/sparkfun/MAX30105_Breakout

  This demo shows heart rate and SPO2 levels.

  It is best to attach the sensor to your finger using a rubber band or other tightening
  device. Humans are generally bad at applying constant pressure to a thing. When you
  press your finger against the sensor it varies enough to cause the blood in your
  finger to flow differently which causes the sensor readings to go wonky.

  This example is based on MAXREFDES117 and RD117_LILYPAD.ino from Maxim. Their example
  was modified to work with the SparkFun MAX30105 library and to compile under Arduino 1.6.11
  Please see license file for more info.

  Hardware Connections (Breakoutboard to Arduino):
  -5V = 5V (3.3V is allowed)
  -GND = GND
  -SDA = A4 (or SDA)
  -SCL = A5 (or SCL)
  -INT = Not connected

  The MAX30105 Breakout can handle 5V or 3.3V I2C logic. We recommend powering the board with 5V
  but it will also run at 3.3V.

  ## Changes May/June 2017 by RA

  * changed output formatting
    - TODO: re-integrate SpO2 calculations
  * fixed some Warnings
  * adapt to PlatformIO

*/
#include <Arduino.h>
#include <Wire.h>

#include <MAX30105.h>
//#include "spo2_algorithm.h"

MAX30105 particleSensor;

#define MAX_BRIGHTNESS 255

int32_t bufferLength; //data length
int32_t bufferPart; //data length of buffer part
int32_t spo2; //SPO2 value
int8_t validSPO2; //indicator to show if the SPO2 calculation is valid
int32_t heartRate; //heart rate value
int8_t validHeartRate; //indicator to show if the heart rate calculation is valid

uint32_t redValue;
uint32_t irValue;

void setup()
{
  Serial.begin(115200); // initialize serial communication at 115200 bits per second:
  while (!Serial);

  // Initialize sensor
  if (!particleSensor.begin(Wire, I2C_SPEED_FAST)) //Use default I2C port, 400kHz speed
  {
    Serial.println(F("## MAX30105 was not found. Please check wiring/power."));
    while (1);
  }

  #if require_input == 1
    Serial.println(F("## Attach sensor to finger with rubber band. Press any key to start."));
    while (Serial.available() == 0) ; //wait until user presses a key
    Serial.read();
  #else
    Serial.println(F("## Starting ..."));
  #endif

  // data header
  /*
  Serial.print(F("time_ms;"));
  Serial.print(F("RED;"));
  Serial.print(F("IR;"));
  Serial.print(F("HR;"));
  Serial.print(F("HRvalid;"));
  Serial.print(F("SPO2;"));
  Serial.println(F("SPO2valid"));
  */

  // used for 170501 data
  int8_t ledBrightness = 20; //Options: 0=Off to 255=50mA
  int8_t sampleAverage = 2; //Options: 1, 2, 4, 8, 16, 32
  byte ledMode = 2; //Options: 1 = Red only, 2 = Red + IR, 3 = Red + IR + Green
  int16_t sampleRate = 400; //Options: 50, 100, 200, 400, 800, 1000, 1600, 3200
  int16_t pulseWidth = 411; //Options: 69, 118, 215, 411
  //int16_t adcRange = 8192; //Options: 2048, 4096, 8192, 16384
  int16_t adcRange = 4096; //Options: 2048, 4096, 8192, 16384

  // check variant: speed decrease!
  //byte sampleRate = 800; //Options: 50, 100, 200, 400, 800, 1000, 1600, 3200
  //int pulseWidth = 215; //Options: 69, 118, 215, 411


  particleSensor.setup(ledBrightness, sampleAverage, ledMode, sampleRate, pulseWidth, adcRange); //Configure sensor with these settings
}

void loop()
{
  while (1)
  {
    while (particleSensor.available() == false) //do we have new data?
      particleSensor.check(); //Check the sensor for new data

    redValue = particleSensor.getRed();
    irValue = particleSensor.getIR();
    Serial.print(millis(), DEC);
    Serial.print(";");
    particleSensor.nextSample(); //We're finished with this sample so move to next sample

    //send samples and calculation result to terminal program through UART
    //Serial.print(F("red="));
    Serial.print(redValue, DEC);
    Serial.print(";");
    //Serial.print(F(", ir="));
    Serial.println(irValue, DEC);
  }
}

/*
Local Variables:
eval: (platformio-mode)
End:
*/
