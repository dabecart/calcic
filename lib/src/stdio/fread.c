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
        // First check the pushback character (if any).
        if(stream->flags & PUSHBACK_AVAILABLE) {
            *buf = stream->pushback;
            buf++;
            byteCount--;
            
            // Clear the pushback flag.
            stream->flags &= ~PUSHBACK_AVAILABLE;
        }
        
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
            // Try to get what we can from the existing buffer.
            size_t available = pop_N(stream, buf + bytesRead, byteCount - bytesRead);
            bytesRead += available;

            if (bytesRead >= byteCount) {
                break;
            }

            // Buffer is empty, refill with a request to the OS.
            if (temp == NULL) {
                temp = malloc(BUFSIZ);
                if (!temp) {
                    // Could not allocate space for the temporal buffer.
                    break;
                }
            }

            long status = __arch_read_file(stream->fd, temp, BUFSIZ);
            if (status <= 0) {
                // There was an error reading bytes from the file.
                break;
            }

            // Push from the temp buffer to the circular buffer of 'stream'.
            push_N(stream, temp, (size_t)status);
        }

        if(temp != NULL) {
            free(temp);
        }

        elementsRead = bytesRead / size;
    }

    return elementsRead;
}
