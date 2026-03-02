/***************************************************************************************************
 * fread.c
 * 
 * This function is part of the <stdio.h> standard library.
 * 
 * This library is part of the calcic compiler, written by @dabecart. 2026.
***************************************************************************************************/

#define COMPILING_STDIO
#include <stdio.h>
#include <stdlib.h>
#include <arch.h>
#include <string.h>

size_t fread(void *ptr, size_t size, size_t nmemb, FILE *stream) {
    if((size == 0) || (nmemb == 0) || (stream == NULL)) {
        return 0;
    }

    unsigned char *buf = ptr;

    size_t byteCount = size * nmemb;
    size_t elementsRead;

    if(stream->flags & UNBUFFERED_MODE) {
        // In unbuffered mode, fetch the data directly from the OS.
        long status = __arch_read_file(stream->fd, buf, byteCount);
        if(status < 0) {
            status = 0;
        }
        elementsRead = status / size;

    }else {
        // In buffered mode, first, fetch the data from the internal buffer. If there aren't enough 
        // bytes in the buffer, make a system call and fully fill the buffer again (if possible).
        
        size_t bytesFromBuffer = 0;
        if(byteCount >= stream->len) {
            // Empty the buffer and dump it to buff.
            bytesFromBuffer = pop_N(stream, buf, stream->len);
            byteCount -= bytesFromBuffer;
            buf += bytesFromBuffer;

            // Now, fetch as many bytes from the OS as possible.
            unsigned char* temp = malloc(BUFSIZ);
            long status = __arch_read_file(stream->fd, ptr, BUFSIZ);
            if(status < 0) {
                status = 0;
            }
            // Push all bytes into the circular buffer.
            push_N(stream, temp, status);
            free(temp);
        }

        // Fetch from the buffer 'byteCount' bytes.
        bytesFromBuffer += pop_N(stream, buf, byteCount);

        elementsRead = bytesFromBuffer / size;
    }

    return elementsRead;
}
