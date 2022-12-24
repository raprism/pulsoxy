#include <Arduino.h>

#include <MAX30105.h>
#include <WiFi.h>
#include <MQTTClient.h>


// error when defined before Wifi.h!!!
#include <local.h>
#ifdef local_is_stub // stub/local.h
  #error "Create lib/LOCAL/local.h! Use lib/stub/local.h as template."
#endif
// DEBUG defined in local.h?!
#include <debug.h>

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

void setup()
{
  Serial.begin(115200); // initialize serial communication at 115200 bits per second:
  //Serial.begin(9600);
  //while (!Serial);

  // ESP setup
  pinMode(LED_BUILTIN, OUTPUT);  // Initialize the LED pin as an output
  pinMode(KEY_BUILTIN, INPUT_PULLUP);  // Initialize the BUTTON 0 as an input

  // wifi & MQTT
  // the IP address for the shield:
  IPAddress ip(192, 168, IP_part3, IP_part4);
  IPAddress gateway(192, 168, IP_part3, IP_part4_gateway);
  IPAddress subnet(255, 255, 255, 0);
  WiFi.mode(WIFI_STA);
  // if not configured, DHCP should provide IP
  WiFi.config(ip, gateway, subnet);

  // init I2C with possibility to redefine pins
  #ifndef PIN_SDA
    #define PIN_SDA SDA
  #endif
  #ifndef PIN_SCL
    #define PIN_SCL SCL
  #endif
  Wire.begin(PIN_SDA, PIN_SCL);
  // Initialize sensor
  // external pullup's needed?
  // https://github.com/espressif/arduino-esp32/issues/741#issuecomment-409618371
  pinMode(SDA,INPUT_PULLUP);
  pinMode(SCL,INPUT_PULLUP);
  // check I2C
  Wire.beginTransmission(0x57);
  byte error = Wire.endTransmission();
  if (error == 0)
  {
    Serial.print("I2C device found at address 0x57");
    Serial.println("  !");
  }
  //  
  uint16_t cnt = 0;
  bool wire_okay = sens.begin(Wire, I2C_SPEED_FAST);
  //bool wire_okay = sens.begin();
  while ((wire_okay == false) && (cnt<100)) //Use default I2C port, 400kHz speed
  {
    if (!(cnt % 10)) {
      debugln(F("## MAX30105 was not found. Please check wiring/power."));
      debugln(sens.readPartID());
      sens.softReset();
    }
    debug(".");
    delay(100);
    wire_okay = sens.begin(Wire, I2C_SPEED_FAST);
    //wire_okay = sens.begin();
    cnt += 1;
  }

  // start WiFi and MQTT
  WiFi.begin(ssid, passwd);
  client.begin(server.c_str(), net);
  connect();

  #if require_input == 1
    debugln(F("## Attach sensor to finger with rubber band. Press any key to start."));
    while (Serial.available() == 0) ; //wait until user presses a key
    Serial.read();
  #else
    debugln(F("## Starting ..."));
  #endif
  setupSens();

  //
  debugln("Setup done.");
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
    delay(100);
  }
  debug("\nconnected to ");
  debugln(server);
  client.subscribe(TOPIC_IN);
}

void setupSens() {
  sens.softReset();
  debugln(sens.readPartID());
  debugln(sens.getRevisionID());
  //Configure sensor
  sens.setup(ledBrightness, sampleAverage, ledMode, sampleRate, pulseWidth, adcRange);
}

void getSensData() // msg is global variable
{
  redValue = sens.getRed();
  irValue = sens.getIR();
  snprintf(msg, 20, "%lu;%u;%u", millis(), redValue, irValue);
  //We're finished with this sample so move to next sample
  sens.nextSample();
}

bool b_doit = true;

void loop()
{

  client.loop();
  if (!client.connected()) {
    connect();
    setupSens();
  }

  message = "";
  // Switch on the LED if an 1 was received as first character
  if (b_doit) {
    digitalWrite(LED_BUILTIN, HIGH);   // Turn the LED on
    getSensData();
    //debugln(msg);
    message.concat(String(msg) + "\n");
    debug(message);
    client.publish(TOPIC_OUT, message);
  } else {
    digitalWrite(LED_BUILTIN, LOW);  // Turn the LED off
  }
  delay(1000);
  b_doit = !b_doit;

}
