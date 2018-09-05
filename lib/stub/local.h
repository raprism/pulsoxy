#define local_is_stub

#ifndef local_h
#define local_h
// wifi
#define SSID "SSID"
#define PASSWD "PASSWD"
// MQTT
#define SERVER "SERVER"
#define TOPIC_IN "inTopic"
#define TOPIC_OUT "outTopic"
// MAX30105
#define LED_BRIGHTNESS 20  // 0=Off to 255=50mA
#define LED_MODE 2         // 1 = Red only, 2 = Red + IR, 3 = Red + IR + Green
#define LED_PWIDTH 411     // 69, 118, 215, 411
#define SMPL_AVG 4         // 1, 2, 4, 8, 16, 32
#define SMPL_RATE 400      // 50, 100, 200, 400, 800, 1000, 1600, 3200
#define ADC_RANGE 4096     // 2048, 4096, 8192, 16384
#endif // local_h
