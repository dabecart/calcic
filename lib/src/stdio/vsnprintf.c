/***************************************************************************************************
 * vsnprintf.c
 * 
 * This function is part of the <stdio.h> standard library.
 * 
 * This library is part of the calcic compiler, written by @dabecart. 2026.
***************************************************************************************************/

#define COMPILING_STDIO
#include <stdio.h>

int vsnprintf(char *s, size_t n, const char *format, va_list args) {
    if(s == NULL) {
        n = 0;
    }

    // This str will be advanced when _writeFormattedStringToString is called.
    char *str = s;
    // The closing null-character is not counted.
    int retCode = (int) _formatString(format, args, _writeFormattedStringToString, &str, n);
    
    if(s != NULL && (retCode < n)) {
        // Add the null terminator. This write isn't counted in the 'retCode'.
        *str = 0;
    }

    return retCode;
}