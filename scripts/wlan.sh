sudo nmcli radio wifi off
sudo rfkill unblock all
sudo systemctl restart hostapd
sudo systemctl status hostapd
iwconfig
ping 192.168.3.99
