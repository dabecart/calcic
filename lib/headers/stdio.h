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

#ifdef COMPILING_STDIO
    #include <stdarg.h>
#endif

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

    int pushback;           // Used in the ungetc function.
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

// xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
// Character input/output functions.
// xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
int fgetc(FILE *stream);
char *fgets(char *s, int n, FILE *stream);
int fputc(int c, FILE *stream);
int fputs(const char *s, FILE *stream);
int getchar(void);
char *gets(char *s);
int putchar(int c);
int puts(const char *s);
int ungetc(int c, FILE *stream);

#define getc(stream) fgetc(stream)
#define putc(c,stream) fputc(c,stream)

// xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
// Formatted input/output functions.
// xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
int fprintf(FILE *stream, const char *format, ...);
int fscanf(FILE *stream, const char *format, ...);
int printf(const char *format, ...);
int scanf(const char *format, ...);
int snprintf(char *s, size_t n, const char *format, ...);
int sprintf(char *s, const char *format, ...);
int sscanf(const char *s, const char *format, ...);

void clearerr(FILE *stream);
int feof(FILE *stream);
int ferror(FILE *stream);
int fgetpos(FILE *stream, fpos_t *);
int fseek(FILE *stream, long, int);
int fsetpos(FILE *stream, const fpos_t *);
long ftell(FILE *stream);
void perror(const char *);
int remove(const char *);
int rename(const char *, const char *);
void rewind(FILE *stream);
FILE *tmpfile(void);
char *tmpnam(char *);
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
    #define PUSHBACK_AVAILABLE  0x80

    // Array of open files.
    extern FILE *files[FOPEN_MAX];
    // How many files are currently open.
    extern size_t openFiles;

    int _processModeString(const char *mode);

    // Circular buffer handling.
    int initBuffer(FILE *f);
    int closeBuffer(FILE *f);
    size_t push(FILE *f, const unsigned char item);
    size_t push_back(FILE *f, const unsigned char item);
    size_t push_N(FILE *f, const unsigned char *items, size_t count);
    size_t pop(FILE *f, unsigned char* outItem);
    size_t pop_N(FILE *f, unsigned char* outItems, size_t count);

    // Guess length of a formatted string.
    #define FORMATTED_STRING_LEN_GUESS 2048

    // Formatted string generation.
    long _generateFormattedString(const char *format, va_list args, char *out, size_t maxLen);
    long _printfToStream(FILE *stream, const char * format, va_list args);
#endif

#endif // _CALCIC_STDIO_h