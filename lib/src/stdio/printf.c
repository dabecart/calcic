/***************************************************************************************************
 * printf.c
 * 
 * This function is part of the <stdio.h> standard library.
 * 
 * This library is part of the calcic compiler, written by @dabecart. 2026.
***************************************************************************************************/

#define COMPILING_STDIO
#include <stdio.h>

int printf(const char *format, ...) {
    va_list args;
    va_start(args, format);
    int retCode = _printfToStream(stdout, format, args);
    va_end(args);
    return retCode;
}