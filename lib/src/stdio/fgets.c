/***************************************************************************************************
 * fgets.c
 * 
 * This function is part of the <stdio.h> standard library.
 * 
 * This library is part of the calcic compiler, written by @dabecart. 2026.
***************************************************************************************************/

#define COMPILING_STDIO
#include <stdio.h>

char *fgets(char *s, int n, FILE *stream) {
    size_t index = 0;

    // Read n-1 bytes at most.
    const size_t toRead = n - 1;
    while(index < toRead) {
        int status = fread(s + index, 1, 1, stream);
        if(status == 1) {
            if(s[index] == '\n') {
                // The new line is kept.
                break;
            }else {
                index++;
            }
        }
    }

    // Add the null terminator.
    s[index + 1] = 0;

    return s;
}
