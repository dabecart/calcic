/***************************************************************************************************
 * calci32_uart.c
 * 
 * Function and definitions used on UART communications in the 'calci32' architecture.
 * 
 * This library is part of the calcic compiler, written by @dabecart. 2026.
***************************************************************************************************/

#include <calci32_uart.h>
#include <calci32_irq.h>
#include <stddef.h>

// External definition found in <uart.h>
volatile UART_Handler *huart0 = (volatile UART_Handler*) 0x02000000;
void (*UART_IRQ_RxFinishedCallback)(volatile UART_Handler *huart) = NULL;
void (*UART_IRQ_TxFinishedCallback)(volatile UART_Handler *huart) = NULL;

// UART global IRQ handler. Function called by the IRQ when the UART IRQ line is toggled.
static void uart0_globalIRQHandler();

// Data used by the IRQ handlers.
typedef struct UART_IRQ_Data {
    uint8_t  *rxBuf;
    size_t   rxBufLen;
    uint32_t rxIndex;

    const uint8_t *txBuf;
    size_t        txBufLen;
    uint32_t      txIndex;
} UART_IRQ_Data;

UART_IRQ_Data data_uart0;

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

int uart_receive(volatile UART_Handler *huart, uint8_t *buf, const size_t len) {
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

int uart_transmit(volatile UART_Handler *huart, const uint8_t *buf, const size_t len) {
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

int uart_enableIRQ(volatile UART_Handler *huart) {
    if (huart == NULL || hirq == NULL) {
        return 0;
    }

    // Based on which huart is given, select the right IRQ line.
    int irqLine = -1;
    void (*callFunc)() = NULL;
    if (huart == huart0) {
        irqLine = CALCI32_IRQ_LINE_UART0;
        callFunc = uart0_globalIRQHandler;
    }

    if (!irq_setIRQCallFunction(hirq, irqLine, callFunc)) {
        // Failed to set the UART global IRQ callback.
        return 0;
    }

    if (!irq_enable(hirq, irqLine)) {
        // Failed to enable the IRQ line.
        return 0;
    }

    // Finish by enabling the UART.
    huart->CTRL.enable = 1;
    return 1;
}

int uart_disableIRQ(volatile UART_Handler *huart) {
    if (huart == NULL || hirq == NULL) {
        return 0;
    }

    // Disable the UART.
    huart->CTRL.enable = 0;

    // Based on which huart is given, select the right IRQ line.
    int irqLine = -1;
    if (huart == huart0) {
        irqLine = CALCI32_IRQ_LINE_UART0;
    }

    if (!irq_disable(hirq, irqLine)) {
        // Failed to disable the IRQ line.
        return 0;
    }

    return 1;
}

int uart_receiveIRQ(volatile UART_Handler *huart, uint8_t *buf, const size_t len) {
    if (huart == NULL || !huart->CTRL.enable || len <= 0) {
        return 0;
    }

    UART_IRQ_Data *irqData = NULL;
    if (huart == huart0) {
        irqData = &data_uart0;
    } else {
        // "huart" doesn't match any known UART handler.
        return 0;
    }

    // Received data from UART will be stored in "rxBuf".
    irqData->rxBuf = buf;
    irqData->rxBufLen = len;
    irqData->rxIndex = 0;

    return 1;
}

int uart_transmitIRQ(volatile UART_Handler *huart, const uint8_t *buf, const size_t len) {
    if (huart == NULL || !huart->CTRL.enable || len <= 0) {
        return 0;
    }

    UART_IRQ_Data *irqData = NULL;
    if (huart == huart0) {
        irqData = &data_uart0;
    } else {
        // "huart" doesn't match any known UART handler.
        return 0;
    }

    // Data to transmit by UART is stored in "txBuf".
    irqData->txBuf = buf;
    irqData->txBufLen = len;
    irqData->txIndex = 0;

    // Wait for the UART to be idle: check the BUSY flag.
    while (huart->STATUS.busy) {}

    // Write the DOUT register with the first byte to send.
    huart->DOUT = *buf;

    return 1;
}

// Template function used by all UARTx IRQ handlers.
static int uart_globalIRQHandler(volatile UART_Handler *huart, UART_IRQ_Data *data) {
    if (huart == NULL || data == NULL) {
        return 0;
    }

    // Check what triggered the IRQ.
    int isDoingRX = data->rxBuf != NULL && (data->rxIndex < data->rxBufLen);
    int isDoingTX = data->txBuf != NULL && (data->txIndex < data->txBufLen);

    // A read of the STATUS register clears some fields. Read once and use this for all checks.
    const UART_RegSTATUS uartStatus = huart->STATUS;

    if (isDoingRX && uartStatus.rxReady) {
        // UART is doing RX and has just received something: store it into the buffer.
        data->rxBuf[data->rxIndex] = (uint8_t) huart->DIN;

        data->rxIndex++;
        if (data->rxIndex == data->rxBufLen) {
            // RX finished. Trigger the RX finished callback.
            if (UART_IRQ_RxFinishedCallback) {
                UART_IRQ_RxFinishedCallback(huart);
            }
        }
    }

    if (isDoingTX && !uartStatus.busy) {
        // UART is doing TX and is not busy: a byte was sent and now the UART is awaiting the next.
        data->txIndex++;

        if (data->txIndex == data->txBufLen) {
            // TX finished. Trigger the TX finished callback.
            if (UART_IRQ_TxFinishedCallback) {
                UART_IRQ_TxFinishedCallback(huart);
            }
        } else {
            // TX not yet finished. Write the DOUT register with the next byte to send.
            huart->DOUT = data->txBuf[data->txIndex];
        }
    }

    return 1;
}

static void uart0_globalIRQHandler() {
    // UART0 has triggered an IRQ. Call the global IRQ handler with the UART0 handler.
    uart_globalIRQHandler(huart0, &data_uart0);
}
