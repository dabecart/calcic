/***************************************************************************************************
 * unget.c
 * 
 * This function is part of the <stdio.h> standard library.
 * 
 * This library is part of the calcic compiler, written by @dabecart. 2026.
***************************************************************************************************/

#define COMPILING_STDIO
#include <stdio.h>

int ungetc(int c, FILE *stream) {
    if((c == EOF) || (stream == NULL)) {
        return EOF;
    }

    if(stream->flags & UNBUFFERED_MODE) {
        // If the stream is unbuffered, only a single pushback is allowed.
        if(stream->flags & PUSHBACK_AVAILABLE) {
            // There was already a pushback, throw error.
            return EOF;
        }else { 
            stream->pushback = c;
            stream->flags |= PUSHBACK_AVAILABLE;
        }

    }else{
        // If the stream is buffered, we can push 'c' back to the buffer (but push it from the tail 
        // backwards).
        size_t status = push_back(stream, c);
        if(!status) {
            return EOF;
        }
    }

    return c;
}