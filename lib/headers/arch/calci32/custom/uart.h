/***************************************************************************************************
 * <uart.h>
 * 
 * Function and definitions used on UART communications in the 'calci32' architecture.
 * 
 * This library is part of the calcic compiler, written by @dabecart. 2026.
***************************************************************************************************/

#ifndef _CALCIC_CALCI32_UART_h
#define _CALCIC_CALCI32_UART_h

#include <stdint.h>

typedef struct UART_Handler {
    uint32_t CTRL;
    uint32_t STATUS;
    uint32_t DIN;
    uint32_t DOUT;
} UART_Handler;

typedef enum UART_Baudrate {
    UART_B9600    = 0b000,
    UART_B19200   = 0b001,
    UART_B38400   = 0b010,
    UART_B115200  = 0b011,
    UART_B230400  = 0b100,
    UART_B460800  = 0b101,
    UART_B921600  = 0b110,
    UART_B1000000 = 0b111,

    UART_BUNKNOWN = -1
} UART_Baudrate;

UART_Baudrate uart_getBaudrate(UART_Handler *huart);
int uart_setBaudrate(UART_Handler *huart, UART_Baudrate baudrate);

int uart_isEnabled(UART_Handler *huart);
int uart_setEnabled(UART_Handler *huart, int enabled);

// Main UART handler declaration. Use as a parameter for the function above.
extern UART_Handler *huart;

#endif // _CALCIC_CALCI32_UART_h