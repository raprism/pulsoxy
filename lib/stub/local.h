#define local_is_stub

#ifndef LOCAL_H
#define LOCAL_H

// general
// ### defined DEBUG enables usage of serial line
// #define DEBUG
// REQUIRE_INPUT 1 => sensor start only after key input
#define _REQUIRE_INPUT 0
// #define _REQUIRE_INPUT 1
#ifdef DEBUG
#define REQUIRE_INPUT _REQUIRE_INPUT
#else
#define REQUIRE_INPUT 0
#endif

// wifi
#define SERVER_PORT 1883
#define SSID "SSID"
#define PASSWD "PASSWD"
#define IP_part3 111
#define IP_part4 99
#define IP_part4_gateway 1

// MQTT
#define SERVER_A "192.168.111.22"
#define SERVER_B "192.168.111.33"
#define TOPIC_IN "inTopic"
#define TOPIC_OUT "outTopic"

// MAX30105
#define LED_BRIGHTNESS 20  // 0=Off to 255=50mA
#define LED_MODE 2         // 1 = Red only, 2 = Red + IR, 3 = Red + IR + Green
#define LED_PWIDTH 411     // 69, 118, 215, 411
#define SMPL_AVG 4         // 1, 2, 4, 8, 16, 32
#define SMPL_RATE 400      // 50, 100, 200, 400, 800, 1000, 1600, 3200
#define ADC_RANGE 4096     // 2048, 4096, 8192, 16384

#endif  // LOCAL_H
