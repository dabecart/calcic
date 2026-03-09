/***************************************************************************************************
 * snprintf.c
 * 
 * This function is part of the <stdio.h> standard library.
 * 
 * This library is part of the calcic compiler, written by @dabecart. 2026.
***************************************************************************************************/

#define COMPILING_STDIO
#include <stdio.h>
#include <stdarg.h>
#include <limits.h>

int sprintf(char *s, const char *format, ...) {
    va_list args;
    va_start(args, format);

    long strLen = _generateFormattedString(format, args, s, LONG_MAX);

    va_end(args);

    return strLen;
}