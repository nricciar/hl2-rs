# HL2 Client on Teensy 4.1

![Waterfall](doc/waterfall.gif)

Proof of concept client using the `hl2` crate from this repo to connect to Hermes Lite 2
and stream the waterfall to a ILI9341 base LCD over SPI and stream demodulated audio to
the WM8731 audio codec board.

## Pins

Currently this teensy project interacts with three pieces of hardware. the Teensy ethernet
device with its own dedicated pins, the ILI9341 LCD display over SPI, and the WM8731 audio
codec board over I2C.

### ILI9341

```
//    SCK        13
//    MOSI       11
//    MISO       12
//    CS         10
//    DC          9
```

### WM8731

```
//    MikroE   Teensy 4
//    ------   --------
//     SCK        21
//     MISO        8
//     MOSI        7
//     ADCL       20
//     DACL       20
//     SDA        18
//     SCL        19
```
