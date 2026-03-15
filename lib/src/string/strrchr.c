/***************************************************************************************************
 * strrchr.c
 * 
 * This function is part of the <string.h> standard library.
 * 
 * This library is part of the calcic compiler, written by @dabecart. 2026.
***************************************************************************************************/

#include <string.h>

char *strrchr(const char *s, int c) {
    char toSearch = c;
    const char* match = NULL;
    
    for(;;) {
        if(*s == toSearch) {
            // Found a match!
            match = s;
        }

        // The terminating null character is is considered part of the string.
        if(*s == 0) {
            break;
        }

        // Next character...
        s++;
    }

    // Return the last match. If no match was found, NULL will be returned.
    return (char*) match;
}