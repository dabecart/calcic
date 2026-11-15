/***************************************************************************************************
 * calci32_irq.c
 * 
 * Function and definitions to manage IRQs in the 'calci32' architecture.
 * 
 * This library is part of the calcic compiler, written by @dabecart. 2026.
***************************************************************************************************/

#include <calci32_irq.h>

volatile IRQ_Handler *hirq = (volatile IRQ_Handler*) 0x7F000000;

int irq_enable(volatile IRQ_Handler *hirq, int irqLine) {
    if (hirq == NULL || irqLine < 0 || irqLine >= CALCI32_IRQ_LINES) {
        return 0;
    }

    // Enable the IRQx.
    uint32_t mask = (1 << irqLine);
    hirq->CTRL = (hirq->CTRL & (~mask)) | mask;

    return 1;
}

int irq_disable(volatile IRQ_Handler *hirq, int irqLine) {
    if (hirq == NULL || irqLine < 0 || irqLine >= CALCI32_IRQ_LINES) {
        return 0;
    }

    uint32_t mask = ~(1 << irqLine);
    hirq->CTRL = (hirq->CTRL & mask);

    return 1;
}

int irq_enableAll(volatile IRQ_Handler *hirq) {
    if (hirq == NULL) {
        return 0;
    }

    // Set all CTRL bits to 1.
    hirq->CTRL = -1;

    return 1;
}

int irq_disableAll(volatile IRQ_Handler *hirq) {
    if (hirq == NULL) {
        return 0;
    }

    // Set all CTRL bits to 0.
    hirq->CTRL = 0;

    return 1;
}

int irq_setIRQCallFunction(volatile IRQ_Handler *hirq, int irqLine, void (*callFunc)(void)) {
    if (hirq == NULL || irqLine < 0 || irqLine >= CALCI32_IRQ_LINES) {
        return 0;
    }

    hirq->IRQx[irqLine] = callFunc;

    return 1;
}
