/***************************************************************************************************
 * fclose.c
 * 
 * This function is part of the <stdio.h> standard library.
 * 
 * This library is part of the calcic compiler, written by @dabecart. 2026.
***************************************************************************************************/

#define COMPILING_STDIO
#include <stdio.h>
#include <stdlib.h>
#include <arch.h>

int fclose(FILE *stream) {
    if(stream == NULL) {
        return EOF;
    }

    // Flush the file before closing it.
    int flushStatus = fflush(stream);

    int closeStatus = __arch_close_file(stream->fd);

    // Remove the pointer from the open files array.
    for(size_t index = stream->openFileIndex; index < FOPEN_MAX - 1; index++) {
        FILE *movingFile = files[index + 1];
        movingFile->openFileIndex--;
        files[index] = movingFile;
        
        if(movingFile == NULL) {
            // Reached the end of the files list.
            break;
        }
    }
    openFiles--;

    // Deallocate the buffer.
    if(stream->buffer != NULL) {
        closeBuffer(stream);
    }

    // Deallocate the pointer.
    free(stream);

    return ((flushStatus == 0) && (closeStatus == 0)) ? 0 : EOF;
}
