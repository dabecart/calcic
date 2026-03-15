/***************************************************************************************************
 * gets.c
 * 
 * This function is part of the <stdio.h> standard library.
 * 
 * This library is part of the calcic compiler, written by @dabecart. 2026.
***************************************************************************************************/

#define COMPILING_STDIO
#include <stdio.h>

char *gets(char *s) {
    char *p = s;
    for(;;) {
        char c = fgetc(stdin);
        if(c == EOF) {
            // If an error occurs, return NULL.
            return NULL;
        }

        *p = c;
        p++;

        if(c == '\n') {
            // Substitute '\n' by a null terminator.
            *p = 0;
            break;
        }
    }
    return s;
}