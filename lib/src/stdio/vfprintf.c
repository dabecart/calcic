/***************************************************************************************************
 * vfprintf.c
 * 
 * This function is part of the <stdio.h> standard library.
 * 
 * This library is part of the calcic compiler, written by @dabecart. 2026.
***************************************************************************************************/

#define COMPILING_STDIO
#include <stdio.h>

int vfprintf(FILE *stream, const char *format, va_list args) {
    return (int) _formatString(format, args, _writeFormattedStringToStream, stream, SIZE_T_MAX);
}