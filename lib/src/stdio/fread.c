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

        if(stream->buffer == NULL) {
            // Initialize the circular buffer.
            if(!initBuffer(stream)) {
                return 0;
            }
        }
        
        size_t bytesRead = 0;
        unsigned char* temp = NULL;
        
        // We may need to do multiple fetches.
        while(bytesRead < byteCount) {
            // If the number of bytes we need to pop is larger than the current number of bytes in 
            // the buffer...
            if((byteCount - bytesRead) >= stream->len) {
                // Empty the buffer and dump it to buff.
                bytesRead += pop_N(stream, buf + bytesRead, stream->len);
    
                // Now, fetch as many bytes from the OS as possible.
                if(temp == NULL) {
                    // Initialize the temporary lineal buffer.
                    temp = malloc(BUFSIZ);
                    if(temp == NULL) {
                        // An error occurred while allocating memory.
                        break;
                    }
                }

                long status = __arch_read_file(stream->fd, ptr, BUFSIZ);
                if(status < 0) {
                    // An error occurred during I/O.
                    break;
                }

                // Push all bytes into the circular buffer.
                push_N(stream, temp, status);
            }
            
            // Fetch from the buffer the remaining bytes. If there aren't enough bytes in the buffer
            // maybe we'll fetch them in the next iteration.
            bytesRead += pop_N(stream, buf + bytesRead, byteCount - bytesRead);
        }

        if(temp != NULL) {
            free(temp);
        }

        elementsRead = bytesRead / size;
    }

    return elementsRead;
}
