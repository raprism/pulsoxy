/*

  Contains parts of:

  1) Optical SP02 Detection (SPK Algorithm) using the MAX30105 Breakout
     By: Nathan Seidle @ SparkFun Electronics
     Date: October 19th, 2016
     https://github.com/sparkfun/MAX30105_Breakout

  2) MQTT client arduino-mqtt
     https://github.com/256dpi/arduino-mqtt/tree/master/examples/ESP32DevelopmentBoard

  ## Changes May/June 2017 by RA

  * changed output formatting of measurements
    - TODO: optionally re-integrate SpO2 calculations
  * fixed some Warnings
  * adaptations to ESP32
  * adapt to PlatformIO

*/

#include <Arduino.h>

#include <MAX30105.h>
//#include "spo2_algorithm.h"

#include <WiFi.h>
#include <MQTTClient.h>

// ### definded DEBUG enables usage of serial line
//#define DEBUG
//
#include <debug.h>
#include <local.h>
#ifdef local_is_stub // stub/local.h
#endif

void connect();

// ### MAX3010x settings ###
MAX30105 sens;

// MAX3010x setup
int8_t ledBrightness = LED_BRIGHTNESS;
byte ledMode = LED_MODE;
int16_t pulseWidth = LED_PWIDTH;
int8_t sampleAverage = SMPL_AVG;
int16_t sampleRate = SMPL_RATE;
int16_t adcRange = ADC_RANGE;
// actual data
uint32_t redValue;
uint32_t irValue;

void setupSens();

// ### Wifi & MQTT settings
const char* ssid = SSID;
const char* passwd = PASSWD;
String server_a = SERVER_A;
String server_b = SERVER_B;
String server = server_a;

WiFiClient net;
MQTTClient client;

char msg[20];
String message;
uint32_t millisPrev;

void setup()
{
  #ifdef DEBUG
  Serial.begin(115200); // initialize serial communication at 115200 bits per second:
  //while (!Serial);
  #endif

  // ESP setup
  pinMode(LED_BUILTIN, OUTPUT);  // Initialize the LED pin as an output
  pinMode(KEY_BUILTIN, INPUT_PULLUP);  // Initialize the BUTTON 0 as an input

  // wifi & MQTT
  // the IP address for the shield:
  IPAddress ip(192, 168, IP_part3, IP_part4);
  IPAddress gateway(192, 168, IP_part3, 254);
  IPAddress subnet(255, 255, 255, 0);
  WiFi.mode(WIFI_STA);
  WiFi.config(ip, gateway, subnet);
  WiFi.begin(ssid, passwd);
  client.begin(server.c_str(), net);
  connect();

  // Initialize sensor
  if (!sens.begin(Wire, I2C_SPEED_FAST)) //Use default I2C port, 400kHz speed
  {
    debugln(F("## MAX30105 was not found. Please check wiring/power."));
    while (1);
  }
  #if require_input == 1
    debugln(F("## Attach sensor to finger with rubber band. Press any key to start."));
    while (Serial.available() == 0) ; //wait until user presses a key
    Serial.read();
  #else
    debugln(F("## Starting ..."));
  #endif
  setupSens();
}

void setupSens() {
  sens.softReset();
  //Configure sensor
  sens.setup(ledBrightness, sampleAverage, ledMode, sampleRate, pulseWidth, adcRange);
}

void connect() {
  debug("\nchecking wifi...");
  while (WiFi.status() != WL_CONNECTED) {
    debug(".");
    delay(1000);
  }
  debug("\nconnected as ");
  debugln(WiFi.localIP());
  debug(String("\nconnecting to MQTT server ") + server);
  while (!client.connect("arduino", "try", "try")) {
    debug(".");
    if (digitalRead(KEY_BUILTIN) == LOW)
    { // Check if button has been pressed
      while (digitalRead(KEY_BUILTIN) == LOW)
        ; // Wait for button to be released
      if (server == server_a) {
        server = server_b;
      } else {
        server = server_a;
      }
      client.begin(server.c_str(), net);
      debug(String("\nconnecting to MQTT server ") + server);
      debug(".");
    }
    delay(1000);
  }
  debug("\nconnected to ");
  debugln(server);
  client.subscribe(TOPIC_IN);
}

void getSensData() // msg is global variable
{
  redValue = sens.getRed();
  irValue = sens.getIR();
  snprintf(msg, 20, "%ld;%ld;%ld", millis(), redValue, irValue);
  //We're finished with this sample so move to next sample
  sens.nextSample();
}

void loop()
{

  client.loop();
  if (!client.connected()) {
    connect();
    setupSens();
  }

  message = "";
  for (int i = 0; i < 100; i++) {
    //millisPrev = millis();
    //sens.check();
    //delay(1);
    while(!sens.available()) {
      //debug("_");
      /*
      if (!sens.safeCheck((uint8_t)1000)) {
        sens.softReset();
      };
      */
      sens.check();
    }
    getSensData();
    debugln(msg);
    message.concat(String(msg) + "\n");
  }

  client.publish(TOPIC_OUT, message);

}

void messageReceived(String topic, String payload, char * bytes, unsigned int length)
{
  debug("Message arrived [");
  debug(topic);
  debug("] ");
  debug(payload);
  debugln();

  // Switch on the LED if an 1 was received as first character
  if (bytes[0] == '1') {
    digitalWrite(LED_BUILTIN, HIGH);   // Turn the LED on
  } else {
    digitalWrite(LED_BUILTIN, LOW);  // Turn the LED off
  }
}


/*
Local Variables:
eval: (platformio-mode)
End:
*/
