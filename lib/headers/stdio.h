/***************************************************************************************************
 * <stdio.h>
 * 
 * Functions and macros for performing input and output, according to Section 7.19 of the C99 
 * standard.
 * 
 * This library is part of the calcic compiler, written by @dabecart. 2026.
***************************************************************************************************/

#ifndef _CALCIC_STDIO_h
#define _CALCIC_STDIO_h

#include <stddef.h> // size_t, NULL

// TODO: L_tmpnam

#define EOF -1
#define FOPEN_MAX 10

#define SEEK_CUR 0
#define SEEK_END 1
#define SEEK_SET 2

// Fully buffered.
#define _IOFBF 0
// Line buffered.
#define _IOLBF 1
// Unbuffered.
#define _IONBF 2
// Size of the buffer inside FILE.
#define BUFSIZ 4096

typedef struct {
    int    fd;              // File descriptor.
    int    flags;
    size_t bufSize;         // Total size of the buffer.
    size_t openFileIndex;   // Index in the open files list.

    size_t  len;            // Number of bytes stored in the data buffer (stored bytes count).
    char    *head;          // Index to read from.
    char    *tail;          // Index to write to.
    char    *buffer;        // Data buffer.
} FILE;

typedef struct {
    unsigned long pos;
}fpos_t;

// xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
// File access functions.
// xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
int fclose(FILE *stream);
int fflush(FILE *stream);
FILE *fopen(const char *filename, const char *mode);
FILE *freopen(const char *filename, const char *mode, FILE *stream);
void setbuf(FILE *stream, char *buf);
int setvbuf(FILE *stream, char *buf, int mode, size_t size);

// xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
// Direct input/output functions.
// xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
size_t fread(void *ptr, size_t size, size_t nmemb, FILE *stream);
size_t fwrite(const void *ptr, size_t size, size_t nmemb, FILE *stream);

void clearerr(FILE *);
int feof(FILE *);
int ferror(FILE *);
int fgetc(FILE *);
int fgetpos(FILE *, fpos_t *);
char *fgets(char *, int, FILE *);
int fprintf(FILE *, const char *, ...);
int fputc(int, FILE *);
int fputs(const char *, FILE *);
int fscanf(FILE *, const char *, ...);
int fseek(FILE *, long, int);
int fsetpos(FILE *, const fpos_t *);
long ftell(FILE *);
int getc(FILE *);
int getchar(void);
char *gets(char *);
void perror(const char *);
int printf(const char *, ...);
int putc(int, FILE *);
int putchar(int);
int puts(const char *);
int remove(const char *);
int rename(const char *, const char *);
void rewind(FILE *);
int scanf(const char *, ...);
int sprintf(char *, const char *, ...);
int sscanf(const char *, const char *, ...);
FILE *tmpfile(void);
char *tmpnam(char *);
int ungetc(int, FILE *);
int vfprintf(FILE *, const char *, char *);
int vprintf(const char *, char *);
int vsprintf(char *, const char *, char *);

extern FILE *stderr, *stdin, *stdout;

// All global variables are inside xdecl_stdio.c
#ifdef COMPILING_STDIO
    #define READ_MODE           0x01
    #define WRITE_MODE          0x02
    #define TRUNCATE_MODE       0x04
    #define APPEND_MODE         0x08
    #define BINARY_MODE         0x10
    #define LINE_BUFFERED_MODE  0x20
    #define UNBUFFERED_MODE     0x40

    // Array of open files.
    extern FILE *files[FOPEN_MAX];
    // How many files are currently open.
    extern size_t openFiles;

    int _processModeString(const char *mode);

    // Circular buffer handling.
    int initBuffer(FILE *f);
    int closeBuffer(FILE *f);
    size_t push(FILE *f, const unsigned char item);
    size_t push_N(FILE *f, const unsigned char *items, size_t count);
    size_t pop(FILE *f, unsigned char* outItem);
    size_t pop_N(FILE *f, unsigned char* outItems, size_t count);

#endif

#endif // _CALCIC_STDIO_h