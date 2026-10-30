/***************************************************************************************************
 * <uart.h>
 * 
 * Function and definitions used on UART communications in the 'calci32' architecture.
 * 
 * This library is part of the calcic compiler, written by @dabecart. 2026.
***************************************************************************************************/

#ifndef _CALCIC_CALCI32_UART_h
#define _CALCIC_CALCI32_UART_h

#include <stddef.h>
#include <stdint.h>

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

typedef struct UART_RegCTRL {
    UART_Baudrate   baudrate    : 3;
    uint32_t                    : 13;
    uint32_t        enable      : 1;
    uint32_t                    : 15;
} UART_RegCTRL;

typedef struct UART_RegSTATUS {
    uint32_t        busy        : 1;
    uint32_t        rxReady     : 1;
    uint32_t                    : 30;
} UART_RegSTATUS;

typedef struct UART_Handler {
    UART_RegCTRL    CTRL;
    UART_RegSTATUS  STATUS;
    uint32_t        DIN;
    uint32_t        DOUT;
} UART_Handler;

UART_Baudrate uart_getBaudrate(volatile UART_Handler *huart);
int uart_setBaudrate(volatile UART_Handler *huart, UART_Baudrate baudrate);

int uart_isEnabled(volatile UART_Handler *huart);
int uart_enable(volatile UART_Handler *huart);
int uart_disable(volatile UART_Handler *huart);

int uart_receive(volatile UART_Handler *huart, char *buf, const size_t len);
int uart_transmit(volatile UART_Handler *huart, const char *buf, const size_t len);

// Main UART handler declaration. Use as a parameter for the function above.
volatile extern UART_Handler *huart;

#endif // _CALCIC_CALCI32_UART_h