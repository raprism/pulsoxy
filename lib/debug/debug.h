#ifndef DEBUG_H
#define DEBUG_H

template <typename... T> void debug (T... args) {
#ifdef DEBUG
  Serial.print(args...);
#endif
}

template <typename... T> void debugln (T... args) {
#ifdef DEBUG
  Serial.println(args...);
#endif
}

#endif
