/***************************************************************************************************
 * calci32_cpu.c
 * 
 * Function and definitions to manage the 'calci32' architecture's CPU inner registers.
 * 
 * This library is part of the calcic compiler, written by @dabecart. 2026.
***************************************************************************************************/

#include <calci32_cpu.h>

volatile CPU_Handler *hcpu = (volatile CPU_Handler *) 0x01000000;