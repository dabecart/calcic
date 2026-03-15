/***************************************************************************************************
 * fwrite.c
 * 
 * This function is part of the <stdio.h> standard library.
 * 
 * This library is part of the calcic compiler, written by @dabecart. 2026.
***************************************************************************************************/

#define COMPILING_STDIO
#include <stdio.h>
#include <stdlib.h>
#include <arch.h>

static int dumpBufferToSystem(FILE *stream, unsigned char** temp) {
    if(*temp == NULL) {
        *temp = malloc(BUFSIZ);
        if(*temp == NULL) {
            // Error while allocating memory.
            return 0;
        }
    }

    size_t writeToOS = pop_N(stream, *temp, stream->len);
    long status = __arch_write_file(stream->fd, *temp, writeToOS);
    if(status < 0) {
        // Error during I/O.
        return 0;
    }

    return 1;
}

size_t fwrite(const void *ptr, size_t size, size_t nmemb, FILE *stream) {
    if((size == 0) || (nmemb == 0) || (stream == NULL)) {
        return 0;
    }

    const unsigned char *buf = ptr;

    size_t byteCount = size * nmemb;
    size_t elementsWritten;

    if(stream->flags & UNBUFFERED_MODE) {
        // In unbuffered mode, write the data directly to the OS.
        long status = __arch_write_file(stream->fd, buf, byteCount);
        if(status < 0) {
            status = 0;
        }
        elementsWritten = status / size;

    }else {
        // In buffered mode, we write data to the buffer first. If the buffer gets full, we output
        // it to the OS. This cycle continues until all bytes are written.

        if(stream->buffer == NULL) {
            // Initialize the circular buffer.
            if(!initBuffer(stream)) {
                return 0;
            }
        }
        
        size_t writtenBytes = 0;
        unsigned char* temp = NULL;
        
        if(stream->flags & LINE_BUFFERED_MODE) {
            // Iterate the input buffer searching for '\n'.
            for(; writtenBytes < byteCount; writtenBytes++) {
                // Push the latest character in the buffer.
                unsigned char lastChar = buf[writtenBytes];
                push(stream, lastChar);

                // If the buffer gets filled or if the last character was '\n', dump it to the OS.
                if((stream->len >= stream->bufSize) || (lastChar == '\n')) {
                    if(!dumpBufferToSystem(stream, &temp)) {
                        break;
                    }
                }
            }
        }else {
            // We may need to do multiple writes.
            while(writtenBytes < byteCount) {
                // Try to fill the buffer if there are enough input bytes.
                size_t toPush = stream->bufSize - stream->len;
                size_t remainingBytes = byteCount - writtenBytes;
                if(toPush > remainingBytes) {
                    toPush = remainingBytes;
                }
    
                writtenBytes += push_N(stream, buf + writtenBytes, toPush);
                
                // If the buffer gets filled, dump it to the OS.
                if(stream->len >= stream->bufSize) {
                    if(!dumpBufferToSystem(stream, &temp)) {
                        break;
                    }
                }
            }
        }

        if(temp != NULL) {
            free(temp);
        }

        elementsWritten = writtenBytes / size;
    }

    return elementsWritten;
}
