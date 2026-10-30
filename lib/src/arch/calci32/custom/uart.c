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
volatile UART_Handler *huart = (volatile UART_Handler*) 0x01000000;

#define UART_CTRL_BAUDRATE_MASK 0x00000007U
#define UART_CTRL_BAUDRATE_BIT  0U

#define UART_CTRL_ENABLE_MASK   0x00010000U
#define UART_CTRL_ENABLE_BIT    16U

UART_Baudrate uart_getBaudrate(volatile UART_Handler *huart) {
    if (huart == NULL) {
        return UART_BUNKNOWN;
    }

    return huart->CTRL.baudrate;
}

int uart_setBaudrate(volatile UART_Handler *huart, UART_Baudrate baudrate) {
    if (huart == NULL) {
        return 0;
    }

    huart->CTRL.baudrate = baudrate;
    return 1;
}

int uart_isEnabled(volatile UART_Handler *huart) {
    if (huart == NULL) {
        return -1;
    }

    return huart->CTRL.enable;
}

int uart_enable(volatile UART_Handler *huart) {
    if (huart == NULL) {
        return 0;
    }

    huart->CTRL.enable = 1;
    return 1;
}

int uart_disable(volatile UART_Handler *huart) {
    if (huart == NULL) {
        return 0;
    }

    huart->CTRL.enable = 0;
    return 1;
}

int uart_receive(volatile UART_Handler *huart, char *buf, const size_t len) {
    if (huart == NULL || !huart->CTRL.enable) {
        return 0;
    }

    for (size_t counter = 0; counter < len; counter++) {
        // Wait for the RX_READY flag to be set.
        while (!huart->STATUS.rxReady) {}

        // Store the DIN register in the output buffer.
        buf[counter] = (uint8_t) huart->DIN;
    }

    return len;
}

int uart_transmit(volatile UART_Handler *huart, const char *buf, const size_t len) {
    if (huart == NULL || !huart->CTRL.enable) {
        return 0;
    }

    // Wait for the UART to be idle: check the BUSY flag.
    while (huart->STATUS.busy) {}

    for (size_t counter = 0; counter < len; counter++) {
        // Write the DOUT register.
        huart->DOUT = buf[counter];
        // The UART starts sending data immediately. Wait for the UART to be idle.
        while (huart->STATUS.busy) {}
    }

    return len;
}