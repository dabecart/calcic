/***************************************************************************************************
 * xdecl_stdio.c
 * 
 * Contains all declarations of global variables used by the stdio functions.
 * 
 * This library is part of the calcic compiler, written by @dabecart. 2026.
***************************************************************************************************/

#define COMPILING_STDIO
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

static FILE in  = {0, READ_MODE};
static FILE out = {1, WRITE_MODE|LINE_BUFFERED_MODE};
static FILE err = {2, WRITE_MODE|UNBUFFERED_MODE};

FILE *stdin = &in;
FILE *stdout = &out;
FILE *stderr = &err;

FILE *files[FOPEN_MAX] = {&in, &out, &err};
// We start with stdin, stdout and stderr.
size_t openFiles = 3;

int _processModeString(const char *mode) {
    int flags = 0;
    if(strcmp(mode, "r") == 0)         flags = READ_MODE;
    else if(strcmp(mode, "w") == 0)    flags = WRITE_MODE | TRUNCATE_MODE;
    else if(strcmp(mode, "a") == 0)    flags = WRITE_MODE | APPEND_MODE;
    else if(strcmp(mode, "rb") == 0)   flags = READ_MODE  | BINARY_MODE;
    else if(strcmp(mode, "wb") == 0)   flags = WRITE_MODE | TRUNCATE_MODE  | BINARY_MODE;
    else if(strcmp(mode, "ab") == 0)   flags = WRITE_MODE | APPEND_MODE    | BINARY_MODE;
    else if(strcmp(mode, "r+") == 0)   flags = READ_MODE  | WRITE_MODE;
    else if(strcmp(mode, "w+") == 0)   flags = READ_MODE  | WRITE_MODE     | TRUNCATE_MODE;
    else if(strcmp(mode, "a+") == 0)   flags = READ_MODE  | WRITE_MODE     | APPEND_MODE;
    else if(strcmp(mode, "r+b") == 0)  flags = READ_MODE  | WRITE_MODE     | BINARY_MODE;
    else if(strcmp(mode, "w+b") == 0)  flags = READ_MODE  | WRITE_MODE     | TRUNCATE_MODE | BINARY_MODE;
    else if(strcmp(mode, "a+b") == 0)  flags = READ_MODE  | WRITE_MODE     | APPEND_MODE   | BINARY_MODE;
    return flags;
}

int initBuffer(FILE *f) {
    f->buffer = malloc(BUFSIZ);
    if(f->buffer == NULL) {
        return 0;
    }

    f->bufSize = BUFSIZ;
    f->head = f->tail = f->buffer;
    return 1;
}

int closeBuffer(FILE *f) {
    if(f->buffer != NULL) {
        free(f->buffer);
        f->bufSize = 0;
        f->head = f->tail = NULL;
    }
    return 1;
}

size_t push(FILE *f, const unsigned char item) {
    if(f->len >= f->bufSize) {
        return 0;
    }

    *f->head = item;
    f->head++;
    if(f->head >= (f->buffer + f->bufSize)) {
        f->head = f->buffer;
    } 
    f->len++; 
    return 1;
}

size_t push_back(FILE *f, const unsigned char item) {
    if(f->len >= f->bufSize) {
        return 0;
    }

    f->tail--;
    if(f->tail < f->buffer) {
        f->tail = f->buffer + f->bufSize - 1;
    } 
    *f->tail = item;
    f->len++; 
    return 1;
}

size_t push_N(FILE *f, const unsigned char *items, size_t count) {
    size_t toPush = count;
    while(toPush > 0) {
        if(!push(f, *items)){
            break;
        }

        items++;
        toPush--;
    }
    return count - toPush;
}

size_t pop(FILE *f, unsigned char* outItem) {
    if(f->len < 1) {
        return 0;
    }

    *outItem = *f->tail;
    f->tail++;
    if(f->tail >= (f->buffer + f->bufSize)) {
        f->tail = f->buffer;
    } 
    f->len--;
    return 1;
}

size_t pop_N(FILE *f, unsigned char* outItems, size_t count) {
    size_t toPop = count;
    while(toPop > 0) {
        if(!pop(f, outItems)){
            break;
        }

        outItems++;
        toPop--;
    }
    return count - toPop;
}

