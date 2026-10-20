/***************************************************************************************************
 * uart.c
 * 
 * Function and definitions used on UART communications in the 'calci32' architecture.
 * 
 * This library is part of the calcic compiler, written by @dabecart. 2026.
***************************************************************************************************/

#include <arch/calci32/custom/uart.h>
#include <stddef.h>

// External definition found in <uart.h>
UART_Handler *huart = (UART_Handler*) 0x01000000;

#define UART_CTRL_BAUDRATE_MASK 0x00000007U
#define UART_CTRL_BAUDRATE_BIT  0U

#define UART_CTRL_ENABLE_MASK   0x00010000U
#define UART_CTRL_ENABLE_BIT    16U

UART_Baudrate uart_getBaudrate(UART_Handler *huart) {
    if (huart == NULL) {
        return UART_BUNKNOWN;
    }

    return (UART_Baudrate) (huart->CTRL & UART_CTRL_BAUDRATE_MASK) >> UART_CTRL_BAUDRATE_BIT;
}

int uart_setBaudrate(UART_Handler *huart, UART_Baudrate baudrate) {
    if (huart == NULL) {
        return 0;
    }

    huart->CTRL = (huart->CTRL & (~UART_CTRL_BAUDRATE_MASK)) | (baudrate << UART_CTRL_BAUDRATE_BIT);
    return 1;
}

int uart_isEnabled(UART_Handler *huart) {
    if (huart == NULL) {
        return -1;
    }

    return (huart->CTRL & UART_CTRL_ENABLE_MASK) >> UART_CTRL_ENABLE_BIT;
}

int uart_setEnabled(UART_Handler *huart, int enabled) {
    if (huart == NULL) {
        return 0;
    }

    // Make sure that enabled is a single 0 or 1.
    enabled = enabled > 0;

    huart->CTRL = (huart->CTRL & (~UART_CTRL_ENABLE_MASK)) | (enabled << UART_CTRL_ENABLE_BIT);
    return 1;
}