/***************************************************************************************************
 * <calci32_irq.h>
 * 
 * Function and definitions to manage IRQs in the 'calci32' architecture.
 * 
 * This library is part of the calcic compiler, written by @dabecart. 2026.
***************************************************************************************************/

#ifndef _CALCIC_CALCI32_IRQ_h
#define _CALCIC_CALCI32_IRQ_h

#include <stddef.h>
#include <stdint.h>

#define CALCI32_IRQ_LINES 4

// IRQ lines to which each device is connected to.
#define CALCI32_IRQ_LINE_UART0 0

typedef struct IRQ_Handler {
    // Destination address of the current IRQ.
    uint32_t        IRQ;
    // Single bits enabled each interruption.
    uint32_t        CTRL;
    // Addresses of the interruption handlers associated to each IRQ line.
    void            (*IRQx[CALCI32_IRQ_LINES])(void);
} IRQ_Handler;

int irq_enable(volatile IRQ_Handler *hirq, int irqLine);
int irq_disable(volatile IRQ_Handler *hirq, int irqLine);
int irq_enableAll(volatile IRQ_Handler *hirq);
int irq_disableAll(volatile IRQ_Handler *hirq);

int irq_setIRQCallFunction(volatile IRQ_Handler *hirq, int irqLine, void (*callFunc)(void));

// Main IRQ handler declaration. Use as a parameter for the functions above.
volatile extern IRQ_Handler *hirq;

#endif // _CALCIC_CALCI32_IRQ_h