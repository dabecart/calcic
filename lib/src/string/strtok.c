/***************************************************************************************************
 * strstr.c
 * 
 * This function is part of the <string.h> standard library.
 * 
 * This library is part of the calcic compiler, written by @dabecart. 2026.
***************************************************************************************************/

#include <string.h>

char *strtok(char *s1, const char *s2) {
    static char *stored = NULL;
    
    if(s1 != NULL) {
        stored = s1;
    }

    if(stored == NULL) {
        return NULL;
    }

    // Search for the first character that is not contained in s2, this will be the start of the 
    // returned token.
    size_t index = strspn(stored, s2);

    // Verify that [index] doesn't exceed the string.
    if(stored[index] == 0) {
        // If it does, stored only consists of s2 characters. Function returns NULL. Stored becomes 
        // NULL too.
        stored = NULL;
        return NULL;
    }

    // This stores the return value, it is the start of the token.
    char *ret = stored + index;
    // Now that we've got the start, search until we find a character which is contained in s2.
    index = strcspn(ret, s2);
    
    if(ret[index] == 0) {
        // If we reach the end of the string while parsing separators, we erase "stored" and return 
        // the start of the token.
        stored = NULL;
    }else {
        // Set the end terminator of the token.
        ret[index] = 0;
    
        // The next token starts from (ret + index + 1) onwards.
        stored = ret + index + 1;
    }

    return ret;
}
