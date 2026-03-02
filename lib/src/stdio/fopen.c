/***************************************************************************************************
 * fopen.c
 * 
 * This function is part of the <stdio.h> standard library.
 * 
 * This library is part of the calcic compiler, written by @dabecart. 2026.
***************************************************************************************************/

#define COMPILING_STDIO
#include <stdio.h>
#include <stdlib.h>
#include <arch.h>

FILE *fopen(const char *filename, const char *mode) {
    if(filename == NULL || mode == NULL) {
        return NULL;
    }

    int flags = _processModeString(mode);
    int fd = __arch_open_file(filename, flags);
    if(fd < 0) { 
        return NULL;
    }

    FILE *file = (FILE*) malloc(sizeof(FILE));
    file->fd = fd;
    file->flags = flags;
    file->buffer = file->head = file->tail = NULL;
    file->bufSize = 0;
    file->openFileIndex = openFiles;

    // Add the file to the open files array.
    files[openFiles] = file;
    openFiles++;

    return file;
}
