#! /bin/bash
pushd $(dirname $0)/../py

# # do you use Anaconda?
# . /opt/anaconda/bin/activate py36

# # By default loads test data.
# # If sensor and microcontroller MQTT is working
# # and subscribed (see mosquito.sh)
# # you can unset this variable (comment this line out)
export USE_MQTT_DEMO=1

if [ $USE_MQTT_DEMO ]; then
  python mqtt.py -p &
  export MQTT_PID=$!
fi

# # the live result display
echo use e.g. --dpi 50 --height 16 
python demo_oxy.py $@

if [ $USE_MQTT_DEMO ]; then
    kill $MQTT_PID
fi

# . deactivate

popd
