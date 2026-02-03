# pulsoxy

Pulseoximeter live 'IoT' experiment for lecture

## short usage description

See the lecture text (sorry, so far only in German) `py/notebook.ipynb` for used hardware.

For full working experiment:

1. Program the microcontroller with code in `src`. Btw I had used platformio for this.

2. Test function with MAX sensor.

3. Test MQTT setup.

For tests without hardware the next step is by default configured 
to start local MQTT messaging from saved data.

4. or alt. 1.: `scripts/demo_oxy.sh` 

## TODO

- improved user documentation

___

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/raprism/pulsoxy)
