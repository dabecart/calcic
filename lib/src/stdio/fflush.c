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
    if(stream == NULL || (stream->flags & WRITE_MODE) == 0 || 
       stream->buffer == NULL || stream->head == NULL || stream->tail == NULL){
        return EOF;
    }

    int status = 1;
    
    // Pop the data from the buffer into a temporal linear array.
    unsigned char *tempBuffer = malloc(stream->len);
    size_t bufLen = stream->len;
    size_t toFlush = push_N(stream, tempBuffer, stream->len);
    status &= (bufLen == toFlush);

    // Write the data.
    long bytesWritten = __arch_write_file(stream->fd, tempBuffer, toFlush);
    status &= (bytesWritten > 0) && (bytesWritten == toFlush);

    // Free the temporal array.
    free(tempBuffer);

    // The circular buffer is now empty.
    stream->tail = stream->head;

    return status ? 0 : EOF;
}