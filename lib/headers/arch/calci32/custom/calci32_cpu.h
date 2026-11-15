/***************************************************************************************************
 * <calci32_cpu.h>
 * 
 * Function and definitions to manage the 'calci32' architecture's CPU inner registers.
 * 
 * This library is part of the calcic compiler, written by @dabecart. 2026.
***************************************************************************************************/

#ifndef _CALCIC_CALCI32_CPU_h
#define _CALCIC_CALCI32_CPU_h

#include <stddef.h>
#include <stdint.h>

typedef struct CPU_RegFLAGS {
    uint32_t  zero        : 1;
    uint32_t  negative    : 1;
    uint32_t  carry       : 1;
    uint32_t  overflow    : 1;
    uint32_t  irqEnabled  : 1;
    uint32_t              : 0;
} CPU_RegFLAGS;

typedef struct CPU_Handler {
    CPU_RegFLAGS        FLAGS;
} CPU_Handler;

#define CPU_ENABLE_IRQ()   ((hcpu)->FLAGS.irqEnabled = 1U)
#define CPU_DISABLE_IRQ()  ((hcpu)->FLAGS.irqEnabled = 0U)

// Main CPU handler declaration.
volatile extern CPU_Handler *hcpu;

#endif // _CALCIC_CALCI32_CPU_h