/***************************************************************************************************
 * fflush.c
 * 
 * This function is part of the <stdio.h> standard library.
 * 
 * This library is part of the calcic compiler, written by @dabecart. 2026.
***************************************************************************************************/

#define COMPILING_STDIO
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <arch.h>

int fflush(FILE *stream) {
    if((stream == NULL) || (stream->buffer == NULL) || 
      !(stream->flags & WRITE_MODE) || (stream->flags & UNBUFFERED_MODE)){
        return EOF;
    }

    int status = 1;
    
    // Pop all the data from the buffer into a temporal linear array.
    size_t bufLen = stream->len;
    unsigned char *tempBuffer = malloc(bufLen);
    size_t toFlush = pop_N(stream, tempBuffer, bufLen);
    status &= (bufLen == toFlush);

    // Write the data.
    long bytesWritten = __arch_write_file(stream->fd, tempBuffer, toFlush);
    status &= (bytesWritten > 0) && (bytesWritten == toFlush);

    // Free the temporal array.
    free(tempBuffer);

    return status ? 0 : EOF;
}