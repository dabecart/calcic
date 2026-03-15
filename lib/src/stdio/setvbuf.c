/***************************************************************************************************
 * setvbuf.c
 * 
 * This function is part of the <stdio.h> standard library.
 * 
 * This library is part of the calcic compiler, written by @dabecart. 2026.
***************************************************************************************************/

#define COMPILING_STDIO
#include <stdio.h>
#include <stdlib.h>
#include <arch.h>

int setvbuf(FILE *stream, char *buf, int mode, size_t size) {
    if((stream == NULL) || (stream->buffer != NULL)) {
        return 1;
    }

    // Set the flags.
    stream->flags &= ~(LINE_BUFFERED_MODE | UNBUFFERED_MODE);
    if(mode == _IOLBF) {
        stream->flags |= LINE_BUFFERED_MODE;
    }else if(mode == _IONBF) {
        stream->flags |= UNBUFFERED_MODE;
    }else if(mode != _IOFBF) {
        // Invalid mode.
        return 1;
    }

    if(buf == NULL) {
        // Create a new buffer with the given size.
        stream->buffer = malloc(size);
    }else {
        // Use the given buffer as the buffer for the stream.
        stream->buffer = buf;
    }
    stream->bufSize = size;

    return 0;
}