/***************************************************************************************************
 * freopen.c
 * 
 * This function is part of the <stdio.h> standard library.
 * 
 * This library is part of the calcic compiler, written by @dabecart. 2026.
***************************************************************************************************/

#define COMPILING_STDIO
#include <stdio.h>
#include <arch.h>

FILE *freopen(const char *filename, const char *mode, FILE *stream) {
    if(stream == NULL) {
        return NULL;
    }

    int flags = _processModeString(mode);
    
    if(filename == NULL) {
        // This implementation doesn't allow reopening.
        return NULL;

    }else {
        // Close the given stream. Do not mind the errors.
        fflush(stream);
        __arch_close_file(stream->fd);

        // Open the file.
        int fd = __arch_open_file(filename, flags);
        if(fd < 0) { 
            return NULL;
        }

        // Modify the values of stream.
        stream->fd = fd;
        stream->flags = flags;
    }

    return stream;
}
